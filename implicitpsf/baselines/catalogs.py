"""Input catalogs for the PIFF and PSFEx baselines.

Both baselines receive exactly the fitting stars we choose (reserved stars removed);
their internal selection is disabled. Our x_pixel/y_pixel are 0-based SEP pixel-center
coordinates; FITS catalogs are 1-based, so +1 on write.

PSFEx cannot run from a plain table: it requires a SExtractor FITS_LDAC catalog with
VIGNET stamps. We synthesize one from the extracted cutouts (masked pixels set to
-1e30, the SExtractor bad-pixel sentinel), avoiding a SExtractor run entirely.
"""

import numpy as np
from astropy.io import fits

VIGNET_BAD = -1e30


def write_piff_catalog(x_pixel, y_pixel, flux, out_path):
    """FITS bintable with 1-based x, y columns for PIFF's input stage."""
    columns = [
        fits.Column(name="x", format="D", array=np.asarray(x_pixel, dtype=np.float64) + 1.0),
        fits.Column(name="y", format="D", array=np.asarray(y_pixel, dtype=np.float64) + 1.0),
        fits.Column(name="flux", format="D", array=np.asarray(flux, dtype=np.float64)),
    ]
    table = fits.BinTableHDU.from_columns(columns)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(out_path, overwrite=True)


def write_psfex_ldac(
    cutouts,
    valid_pixels,
    x_pixel,
    y_pixel,
    flux,
    flux_err,
    snr,
    fwhm_pixels,
    image_header,
    out_path,
    gain=1.0,
    background_dev=1.0,
    saturation=1e9,
):
    """Synthesize a FITS_LDAC catalog (LDAC_IMHEAD + LDAC_OBJECTS) for PSFEx.

    Args:
        cutouts: (n, k, k) background-subtracted stamps
        valid_pixels: (n, k, k) bool, False -> VIGNET bad-pixel sentinel
        x_pixel, y_pixel: 0-based centroids
        flux, flux_err, snr: per-star photometry
        fwhm_pixels: scalar exposure FWHM (fills FLUX_RADIUS for PSFEx's bookkeeping)
        image_header: astropy header of the science image (for LDAC_IMHEAD)
        out_path: destination .fits path
        gain, background_dev, saturation: exposure stats for the SEX* keywords
            psfex insists on reading from the IMHEAD
    """
    n_stars, patch, _ = cutouts.shape
    vignets = np.where(valid_pixels, cutouts, VIGNET_BAD).astype(np.float32)

    header = image_header.copy()
    header["SEXGAIN"] = float(gain)
    header["SEXBKGND"] = 0.0  # stamps are background-subtracted
    header["SEXBKDEV"] = float(background_dev)
    header["SEXSATLV"] = float(saturation)
    # psfex's reader scans for an "END     " card; omitting it crashes the binary
    cards = [card.image for card in header.cards] + ["END" + " " * 77]
    header_cards = np.array([cards], dtype="S80")
    imhead_column = fits.Column(
        name="Field Header Card",
        format=f"{80 * header_cards.shape[1]}A",
        dim=f"(80,{header_cards.shape[1]})",
        array=header_cards,
    )
    imhead = fits.BinTableHDU.from_columns([imhead_column])
    imhead.name = "LDAC_IMHEAD"

    flux_radius = np.full(n_stars, fwhm_pixels / 2.0, dtype=np.float32)
    columns = [
        fits.Column(name="X_IMAGE", format="D", array=np.asarray(x_pixel) + 1.0),
        fits.Column(name="Y_IMAGE", format="D", array=np.asarray(y_pixel) + 1.0),
        fits.Column(name="FLUX_APER", format="E", array=np.asarray(flux, dtype=np.float32)),
        fits.Column(name="FLUXERR_APER", format="E", array=np.asarray(flux_err, np.float32)),
        fits.Column(name="SNR_WIN", format="E", array=np.asarray(snr, dtype=np.float32)),
        fits.Column(name="FLUX_RADIUS", format="E", array=flux_radius),
        fits.Column(name="ELONGATION", format="E", array=np.ones(n_stars, dtype=np.float32)),
        fits.Column(name="FLAGS", format="I", array=np.zeros(n_stars, dtype=np.int16)),
        fits.Column(
            name="VIGNET",
            format=f"{patch * patch}E",
            dim=f"({patch},{patch})",
            array=vignets,
        ),
    ]
    objects = fits.BinTableHDU.from_columns(columns)
    objects.name = "LDAC_OBJECTS"

    fits.HDUList([fits.PrimaryHDU(), imhead, objects]).writeto(out_path, overwrite=True)

    # astropy null-pads string cells on disk; psfex requires FITS-style space padding
    # (its header scan segfaults on "END\\0\\0..."), so patch the IMHEAD bytes in place
    with fits.open(out_path) as hdul:
        data_offset = hdul["LDAC_IMHEAD"].fileinfo()["datLoc"]
    card_bytes = b"".join(card.ljust(80).encode("ascii") for card in cards)
    with open(out_path, "r+b") as handle:
        handle.seek(data_offset)
        handle.write(card_bytes)
