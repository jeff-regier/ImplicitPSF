import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class Catalog:
    def __init__(self, fluxes, positions):
        assert len(fluxes.shape) == 2
        assert len(positions.shape) == 3
        self.fluxes = fluxes
        self.positions = positions

    @property
    def n_images(self):
        return self.fluxes.shape[0]

    @property
    def max_sources(self):
        return self.fluxes.shape[1]

    @property
    def device(self):
        return self.fluxes.device

    def size(self, dim):
        return self.fluxes.shape[dim]


class StarParamsDistribution(torch.distributions.Distribution):
    """Custom distribution for generating star parameters for an arbitrary number of stars."""

    arg_constraints = {}

    def __init__(
        self,
        n_images,
        max_sources,
        mean_sources,
        image_size,
        margin,
        min_sources=0,
        validate_args=None,
    ):
        super().__init__(validate_args=validate_args)
        self.n_images = n_images
        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        self.image_size = image_size
        self.margin = margin
        self.flux_dist = torch.distributions.Uniform(1000.0, 2000.0)

    def sample(self, sample_shape=torch.Size(), generator=None):
        """Generate star parameters: (flux1, x1, y1, flux2, x2, y2, ...) for each image."""
        if sample_shape:
            raise ValueError("sample_shape is not supported")

        dims = (self.n_images, self.max_sources)

        # objected positions within the image are from [-0.5, to image_size - 0.5]
        positions = torch.rand(dims + (2,), generator=generator) * self.image_size - 0.5
        # avoid potential weird boundary issues
        positions = positions.clamp(-0.5 + 1e-6, self.image_size - 0.5 - 1e-6)

        unit_fluxes = torch.rand(dims, generator=generator)
        fluxes = self.flux_dist.low + (self.flux_dist.high - self.flux_dist.low) * unit_fluxes

        poisson_dist = torch.distributions.Poisson(self.mean_sources)
        n_stars = poisson_dist.sample((self.n_images,))
        n_stars = torch.clamp(n_stars, min=self.min_sources, max=self.max_sources)
        star_indices = torch.arange(self.max_sources, device=fluxes.device)
        star_indices = star_indices.view(1, -1)
        observed_mask = star_indices < n_stars.unsqueeze(-1)
        fluxes = fluxes * observed_mask.float()

        return Catalog(fluxes, positions)


class StarImageDistribution(torch.distributions.Distribution):
    """Custom distribution for generating star field images with an arbitrary number of stars."""

    N_ELEC_PER_NMGY = 1000.0

    arg_constraints = {}  # No constraints on arguments

    def __init__(
        self,
        catalog,
        image_size=128,
        patch_size=16,
        sigma=0.5,
        background_intensity=1.0,
        shot_noise=True,
        patches_only=False,
        variable_psf=True,  # Flag to enable/disable variable PSF
        validate_args=None,
    ):
        super().__init__(validate_args=validate_args)
        self.catalog = catalog
        self.image_size = image_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.background_intensity = background_intensity
        self.shot_noise = shot_noise
        self.patches_only = patches_only
        self.variable_psf = variable_psf

    def _render_patches(self):
        """Render patches for each star in the catalog."""
        x_coords = torch.arange(self.patch_size, dtype=torch.float32, device=self.catalog.device)
        y_coords = torch.arange(self.patch_size, dtype=torch.float32, device=self.catalog.device)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing="xy")

        # Calculate star position within patch using astronomical convention
        # Star position within patch, accounting for centering offset
        array_positions = self.catalog.positions + 0.5

        # In _coadd_patches, we use patch_corner = floor(array_pos)
        # To center stars: patch_center + (fractional_offset - 0.5)
        # This puts integer astronomical positions at patch center (3.5 for 8x8 patch)
        patch_corners_canvas = array_positions.floor()
        fractional_offsets = array_positions - patch_corners_canvas
        patch_half_size = self.patch_size // 2
        patch_center = patch_half_size - 0.5  # 3.5 for 8x8 patch

        star_patch_x = patch_center + (fractional_offsets[..., 0] - 0.5)
        star_patch_y = patch_center + (fractional_offsets[..., 1] - 0.5)

        # Handle PSF variation based on flag
        if self.variable_psf:
            # Simple left/right PSF variation based on x-coordinate
            star_positions = self.catalog.positions  # Shape: (n_images, max_sources, 2)
            x_positions = star_positions[..., 0]  # Extract x coordinates

            # Stars in left half (x < image_size/2) get sigma
            # Stars in right half (x >= image_size/2) get 3*sigma
            is_right_half = x_positions >= (self.image_size / 2.0)
            sigma_multiplier = torch.where(is_right_half, 3.0, 1.0)
            spatially_varying_sigma = self.sigma * sigma_multiplier
        else:
            # Fixed PSF: same sigma for all stars
            # Create tensor with same shape as variable case
            shape = self.catalog.fluxes.shape  # (n_images, max_sources)
            spatially_varying_sigma = torch.full(shape, self.sigma, device=self.catalog.device)

        x_pos_exp = star_patch_x[..., None, None]
        y_pos_exp = star_patch_y[..., None, None]
        fluxes_exp = self.catalog.fluxes[..., None, None]
        sigma_exp = spatially_varying_sigma[..., None, None]  # Expand for broadcasting
        X_exp = X[None, None, :, :]
        Y_exp = Y[None, None, :, :]
        dx = X_exp - x_pos_exp
        dy = Y_exp - y_pos_exp
        sigma2 = sigma_exp**2
        norm = 1.0 / (2.0 * torch.pi * sigma2)
        exp_term = torch.exp(-0.5 * (dx**2 + dy**2) / sigma2)
        star_probs = norm * exp_term
        return fluxes_exp * star_probs

    def _coadd_patches(self, patches):
        """Fully vectorized patch overlay using scatter_add_."""
        # Canvas size calculation:
        # - Astronomical positions: [-0.5, image_size - 0.5] (with epsilon clamp)
        # - Array coordinates: [0, image_size] (with +0.5 offset)
        # - Centered patch corners: [-patch_half_size, image_size - patch_half_size]
        # - Patch extents: [patch_half_size, image_size + patch_half_size]
        # - Add offset to avoid negative indices: all coords + patch_half_size
        # - Final canvas range: [0, image_size + patch_size - 1]
        # - Required canvas size: image_size + patch_size
        canvas_size = self.image_size + self.patch_size
        images_shape = (self.catalog.size(0), 1, canvas_size, canvas_size)
        canvas = torch.zeros(images_shape, device=self.catalog.device)

        # Get patch coordinates for all valid patches
        n_images, max_sources, patch_h, patch_w = patches.shape

        # Create coordinate grids for patches
        patch_y, patch_x = torch.meshgrid(
            torch.arange(patch_h, device=patches.device),
            torch.arange(patch_w, device=patches.device),
            indexing="ij",
        )

        # Expand coordinates to match batch dimensions
        patch_y = patch_y[None, None, :, :].expand(n_images, max_sources, -1, -1)
        patch_x = patch_x[None, None, :, :].expand(n_images, max_sources, -1, -1)

        # Convert astronomical coordinates to canvas coordinates with centered patches
        # 1. Convert to array coordinates: astronomical + 0.5
        # 2. Center patches: subtract patch_half_size, then add it back to avoid negative indices
        # Net effect: patch_corner = floor(array_pos - patch_half_size + patch_half_size) = floor(array_pos)
        array_positions = self.catalog.positions + 0.5
        patch_corner_positions = array_positions.floor().long()

        global_y = patch_y + patch_corner_positions[..., 1:2, None]
        global_x = patch_x + patch_corner_positions[..., 0:1, None]

        # Flatten everything for scatter_add_
        img_indices = torch.arange(n_images, device=patches.device)[:, None, None, None].expand_as(
            patches
        )

        # Flatten all tensors
        flat_img_idx = img_indices.flatten()
        flat_y = global_y.flatten()
        flat_x = global_x.flatten()
        flat_patches = patches.flatten()

        # Convert to 3D indices for scatter_add_
        # We need to combine image index with spatial indices
        linear_idx = flat_img_idx * canvas_size * canvas_size + flat_y * canvas_size + flat_x

        # Flatten canvas completely for scatter_add_
        canvas_flat = canvas.view(-1)
        canvas_flat.scatter_add_(0, linear_idx, flat_patches)

        return canvas

    def render_expected_images(self):
        patches = self._render_patches()

        # Add patches to canvas
        canvas = self._coadd_patches(patches)

        # we should crop an additional patch_size // 2 to keep the flux density uniform,
        # but this is enough to keep the centroid density constant
        margin = self.patch_size // 2
        expected_images = canvas[:, :, margin:-margin, margin:-margin]

        # add background (in physical units)
        expected_images = expected_images + self.background_intensity

        return expected_images

    def sample(self, sample_shape=torch.Size(), generator=None):
        """Generate an image by rendering stars with Gaussian PSF and shot noise."""
        patches = self._render_patches()

        if self.patches_only:
            # Only return patches with background and noise
            expected_patches = patches + self.background_intensity
            expected_nelecs = expected_patches * self.N_ELEC_PER_NMGY
            if self.shot_noise:
                standard_noise = torch.normal(
                    mean=0.0, std=1.0, size=expected_nelecs.shape, generator=generator
                )
                observed_nelecs = expected_nelecs + standard_noise * expected_nelecs.sqrt()
            else:
                observed_nelecs = expected_nelecs
            observed_patches = observed_nelecs / self.N_ELEC_PER_NMGY
            return observed_patches, patches
        else:
            # Return full images as before
            expected_images = self.render_expected_images()
            expected_nelecs = expected_images * self.N_ELEC_PER_NMGY
            if self.shot_noise:
                standard_noise = torch.normal(
                    mean=0.0, std=1.0, size=expected_nelecs.shape, generator=generator
                )
                observed_nelecs = expected_nelecs + standard_noise * expected_nelecs.sqrt()
            else:
                observed_nelecs = expected_nelecs
            observed_image = observed_nelecs / self.N_ELEC_PER_NMGY
            return observed_image, patches


class StarDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for star data."""

    def __init__(
        self,
        n_samples=1024,
        image_size=128,
        max_sources=512,
        mean_sources=256,
        min_sources=0,
        patch_size=16,
        sigma=0.5,
        background_intensity=1.0,
        shot_noise=True,
        as_patches=False,
        batch_size=1,
        seed=42,
        variable_psf=True,  # Flag to enable/disable variable PSF
    ):
        super().__init__()
        # StarParamsDistribution parameters
        self.n_samples = n_samples
        self.max_sources = max_sources
        self.mean_sources = mean_sources
        self.min_sources = min_sources
        self.shot_noise = shot_noise
        self.as_patches = as_patches

        # StarImageDistribution parameters
        self.image_size = image_size
        self.patch_size = patch_size
        self.sigma = sigma
        self.background_intensity = background_intensity
        self.variable_psf = variable_psf

        # Data module parameters
        self.batch_size = batch_size
        self.seed = seed

    def setup(self, stage=None):
        print("Generating star data...")
        g = torch.Generator().manual_seed(self.seed)

        # Generate star parameters
        star_params_dist = StarParamsDistribution(
            n_images=self.n_samples,
            max_sources=self.max_sources,
            mean_sources=self.mean_sources,
            min_sources=self.min_sources,
            image_size=self.image_size,
            margin=self.patch_size // 2,
        )
        catalog = star_params_dist.sample(generator=g)

        # Generate images from star parameters
        image_dist = StarImageDistribution(
            catalog=catalog,
            image_size=self.image_size,
            patch_size=self.patch_size,
            sigma=self.sigma,
            background_intensity=self.background_intensity,
            patches_only=self.as_patches,
            variable_psf=self.variable_psf,
        )
        images, patches = image_dist.sample(generator=g)

        # Create dataset with tensors only
        if self.as_patches:
            # images is now processed patches with background and noise
            processed_patches = images

            # Convert positions from image coordinates to patch-relative coordinates
            # The patches are created by centering around each star, so we need to compute
            # where the star is within its own patch
            array_positions = catalog.positions + 0.5  # Convert to array coordinates
            patch_corners = array_positions.floor()  # Patch corner positions
            fractional_offsets = array_positions - patch_corners  # Star position within patch
            patch_center = self.patch_size // 2 - 0.5  # 3.5 for 8x8 patch
            patch_positions = patch_center + (fractional_offsets - 0.5)

            dataset = TensorDataset(processed_patches, catalog.fluxes, patch_positions)
        else:
            dataset = TensorDataset(images, catalog.fluxes, catalog.positions)

        # Use a fixed generator for random_split
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=g
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
