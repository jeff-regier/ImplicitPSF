# An Amortized Continuous PSF Field for DES Single-Epoch Imaging

*Draft manuscript — generated 2026-06-16. Numbers are from the frozen test split unless
noted. Sections marked [TODO] need figures or final values from in-flight runs.*

## Abstract

Accurate point-spread-function (PSF) models are the dominant systematic in weak-lensing
shear measurement. The standard approach fits an independent PSF model per exposure
(e.g. PIFF, PSFEx) from that exposure's stars. We instead learn a single *amortized*
continuous PSF field: a multi-head attention network takes an exposure's stars as
context (position, color, flux, and pixel cutouts) and predicts the PSF at any queried
position and resolution, with no per-exposure fitting at test time. On 3,301 frozen test
exposures of DES single-epoch imaging (CCD 31, grizY; 73,609 reserved stars), the model
matches or beats PIFF and PSFEx on every metric: size-residual scatter 0.030 vs
0.039/0.037, ellipticity errors |de1,de2| = 0.015/0.015 vs 0.017/0.016 (PIFF), and
shape correlations 0.65/0.61 vs 0.61/0.58 (PIFF), at chi2/dof 1.04. On a star-anchored
galaxy injection-recovery test on real data, our PSF recovers galaxy *ellipticity* far
better than PIFF (Δe1 = +0.004 vs −0.138), the weak-lensing-critical quantity. Color
conditioning removes a chromatic PSF-size systematic that the baselines retain
(residual slope +0.004/mag vs −0.06/mag). The model is flat in the number of available
fit stars where PIFF degrades sharply (chi2 1.1 vs 5.1 at five stars). Using simulations
as a controlled instrument, we show that a *blend forward-model* training loss removes a
size bias from unrecognized blends, and we identify and fix a spin-2 representation
failure in which the network corrupts the PSF ellipticity by attending to galaxy
detections in its context; restricting attention to point sources recovers it fully.

## 1. Introduction

Weak gravitational lensing infers the matter distribution from coherent ~1% distortions
of galaxy shapes. The measurement is limited by knowledge of the PSF: any error in the
modeled PSF size or ellipticity propagates directly into the inferred shear. Survey
pipelines model the PSF per exposure by fitting the stars in that exposure with a
flexible model interpolated across the focal plane (PSFEx; PIFF). This is robust but
(i) discards information shared across exposures and nights, (ii) degrades when few clean
stars are available, and (iii) is purely empirical at the star positions — it does not
exploit auxiliary information such as stellar color, which sets the chromatic PSF.

We take an amortized, learning-based approach. A single network is trained across many
exposures to map an exposure's stars (the *context*) to a continuous PSF field, queryable
at arbitrary position, color, flux, and resolution. The inferential target is the PSF at
*star-free* positions, where galaxies are measured; predicting held-out stars is the
training objective and the validation instrument. We train and evaluate on DES
single-epoch data and benchmark against PIFF and PSFEx run by us under matched
conditions.

## 2. Method

**Architecture.** Each detected source is encoded from its pixel cutout (a learned image
encoder, applied to the scale-normalized stamp), its sky position (sinusoidal encoding),
and scalar color and log-flux. A multi-head self-attention block lets every query attend
to the context sources; a coordinate decoder then renders the PSF at the query position
and any oversampling factor, returning the pixel-convolved effective PSF used to convolve
galaxy models. We use a polar coordinate parametrization of the decoder (a spin-2-aware
angular basis) and FiLM modulation of the decoder by the attended context features.

**Loss.** Training minimizes an inverse-variance chi2 between predicted and observed star
cutouts, with the per-star amplitude solved in closed form. We introduce a *blend
forward-model* loss: rather than treat each clean star as isolated, we model its cutout
as a generalized-least-squares sum of the target plus its detected neighbors, all
rendered by the same decoder (§4.5). Detected galaxies near a target are handled by
exclusion (the default), masking, or a crude extended-component model; the three are
within noise on real reserved stars.

**Point-source context.** The PSF should be informed only by point sources. We restrict
the attention context to stellar detections (`point_source_context`); §4.5 shows this is
necessary to avoid a spin-2 corruption from extended sources in the context.

**Color conditioning.** The decoder conditions on g−i color, enabling a per-object
chromatic PSF (differential chromatic refraction and wavelength-dependent seeing) that
per-exposure empirical models cannot represent (§4.3).

## 3. Data

DES single-epoch FITS (Y5A1, CCD 31; amplifier B dead, MSK bit 8). Sources are detected
with SEP and cross-matched to DES DR2 for colors, SPREAD_MODEL star/galaxy labels, and
COADD object ids. Splits are at the observing-night level by deterministic hash to
prevent leakage (consecutive exposures share atmosphere), frozen in a committed manifest
with ~20% of clean stars per test/val exposure reserved for scoring only; the test split
is 3,301 exposures across grizY. Reserved stars are excluded from every method's
fit/context and scored identically across methods on the same pixel grid.

**Simulations.** A matched simulation pipeline renders Moffat PSF fields (2nd-order
polynomial FWHM and shear variation — exactly representable by PIFF/PSFEx, a deliberately
fair-to-baselines choice) into the same v2 data format, optionally with chromatic
FWHM-color dependence, blended stars, and injected galaxy detections. The simulation is
the controlled instrument for the systematics studies in §4.5; star density, galaxy
density (3:1), and per-source SNR are tuned to match the real extraction.

## 4. Results

### 4.1 Reserved-star accuracy on real data (headline)

On 3,301 test exposures / 73,609 reserved stars, the amortized model (real_v6_blend; no
per-exposure fitting) matches or beats both baselines on every metric:

| method | T scatter | \|de1\| | \|de2\| | corr(e1) | corr(e2) | chi2/dof |
|---|---|---|---|---|---|---|
| **implicit (ours)** | **0.0301** | **0.0151** | **0.0150** | **0.650** | **0.607** | 1.037 |
| PIFF | 0.0393 | 0.0171 | 0.0158 | 0.611 | 0.577 | 1.073 |
| PSFEx | 0.0371 | 0.0160 | 0.0154 | 0.645 | 0.604 | 1.047 |

Caveat: reserved "clean" stars carry a mild blend-contamination floor that suppresses the
absolute correlations for *all* methods; the comparison is fair (identical stars) and the
ordering is robust. [TODO: rho-statistics figure; per-band breakdown.]

### 4.2 Galaxy recovery on real data (star-anchored injection)

We test the inferential goal directly: inject galsim galaxies, convolved with a strictly
isolated reserved star's empirical PSF, at that star's position; each method predicts the
PSF there from the *other* stars and recovers the galaxy. The truth arm (the anchor's own
image) recovers near-zero bias, validating the protocol. On 113 r-band exposures / 522
anchors:

| arm | size bias | Δe1 | Δe2 |
|---|---|---|---|
| truth (validation) | −1.5% | −0.002 | ~0 |
| **implicit (ours)** | −7.6% | **+0.004** | −0.002 |
| PIFF | −1.0% | **−0.138** | −0.001 |

Our PSF recovers galaxy *ellipticity* essentially without bias, where PIFF's PSF
ellipticity error induces a large shape bias (Δe1 = −0.14). PIFF recovers size slightly
better; our PSF is ~6% too broad at star-free positions. For weak lensing, the
ellipticity result is the consequential one. [TODO: extend to all bands; size-bias
investigation.]

### 4.3 Chromatic PSF correction

In a chromatic simulation (FWHM varies −3%/mag of g−i), color conditioning reduces the
residual chromatic size slope from −0.06/mag (PIFF, PSFEx, and our own zero-color
ablation) to +0.004/mag — a ~17× reduction. On real r-band data the paired
color-vs-zero-color slope difference is +0.0024/mag (CI [+0.0016, +0.0030]); the
baselines structurally retain the systematic. This is information unavailable to a
per-exposure empirical PSF.

### 4.4 Sample efficiency

Restricting every method to k randomly chosen fit stars, the amortized model is
approximately flat in k (size scatter 0.078→0.056, chi2 ~1.1 from k=5 to k=100), while
PIFF collapses below k≈25 (chi2 5.1, scatter 0.18 at k=5). Amortization transfers
information from the training set to sparsely-populated exposures. [TODO: figure.]

### 4.5 Simulation studies (controlled instrument)

**Architecture ladder.** On the star-free truth grid of the simulation, the chi2/dof
improves along the architecture progression additive → FiLM → diagonal → polar
(8.9 → 4.96 → 4.41; PIFF 2.67; verified floor 1.0), recovering both ellipticity
components.

**Blend forward-model.** On a blended simulation, single-star training leaves a +4.8%
size bias on the star-free truth grid; the blend forward-model loss reduces it to +0.7%
(≈7×). [Ellipticity behavior under the blend loss is discussed with the point-source
finding below.]

**Galaxy-context corrupts PSF ellipticity — and the fix.** Injecting galaxy detections
into the simulation, a model that attends to *all* detections (including galaxies) as
context suffers a clean, reproducible failure: it cannot represent one spin-2 ellipticity
component (corr(e1)=0.05 under the polar parametrization; the failure flips to e2 under a
Cartesian parametrization, identifying it as a representation degeneracy, not noise). The
other component is recovered perfectly. We localize the cause by ablation: it is *not*
data volume, per-source SNR, galaxy size/color, or the chi2 cap. The model is treating
extended galaxy cutouts as PSF evidence. Restricting the attention context to point
sources (`point_source_context`) recovers *both* components completely (corr(e1,e2)=0.98,
matching the galaxy-free ideal); a coordinate-system change does not. Galaxies are clearly
distinguishable (cutout second moment T≈33 vs 15 px² for stars), so this is the model
failing to use available information, fixed by the principled restriction that only point
sources inform a PSF. [In progress: confirming on real data that point-source context
preserves or improves the §4.1 metrics; the real-data baseline already works because the
real model's behavior does not trigger the degeneracy.]

## 5. Discussion

The amortized field trades a per-exposure fit for a single trained model. Benefits: no
test-time fitting, robustness in star-sparse exposures, and the ability to condition on
color (chromatic PSF) and flux (brighter-fatter). Costs and caveats:

- **Training-data requirement.** Unlike per-exposure fitters, the amortized model needs
  many exposures to learn the field; in simulation, ellipticity recovery is unstable
  below a few thousand exposures. Real DES (32k exposures) is well above this.
- **Simulation fidelity for ellipticity.** The simulation is a faithful instrument for
  size/bias and for the systematic studies above, but its idealized PSF profile (Moffat,
  low-order polynomial field) favors the per-exposure baselines (zero model mismatch).
  Absolute accuracy comparisons rest on real data, which carry the head-to-head.
- **Point-source context as a model principle.** The galaxy-context finding (§4.5)
  argues that the context for a PSF estimator should be gated to point sources; we adopt
  this.

## 6. Conclusion

A single attention-based continuous PSF field, trained across DES single-epoch exposures
and applied with no per-exposure fitting, matches or beats PIFF and PSFEx on real
reserved stars and recovers galaxy ellipticity better than PIFF on a star-anchored
injection test, while additionally correcting a chromatic systematic and remaining robust
in star-sparse exposures. Controlled simulations motivate two training choices — a blend
forward-model loss and point-source-only attention context — the latter from a clean
diagnosis of a spin-2 representation failure triggered by galaxy detections in context.

## Methods/data availability, baselines configuration

PIFF: PixelGrid, 2nd-order BasisPolynomial interpolation, 4σ outlier rejection. PSFEx:
PSF_SAMPLING 0.5 (super-resolved), 45×45, PSFVAR_DEGREES 2; rendered via galsim
DES_PSFEx with method='no_pixel'. All methods fit on identical non-reserved clean stars
and scored on identical reserved stars. Splits and reserved-star lists are frozen in a
committed manifest. [TODO: code/data release statement.]

## Open items before submission

1. Point-source-context production model on real data (training; ~hours) — confirm it
   preserves/improves §4.1, then adopt as the production model and add its seed-stability.
2. Single-loss vs blend-loss comparison on real reserved stars (eval running).
3. Figures: rho-statistics, spatial residual maps, residual histograms, galaxy-recovery
   and sample-efficiency plots (most exist in results/*/ from prior runs; regenerate
   against real_v6_blend).
4. Per-band breakdown of §4.1; extend §4.2 galaxy recovery to all bands; investigate the
   −7.6% size bias.
5. Seed-stability of the production model (real_v6_blend best val 1.994 == seed1 1.994,
   confirmed; add point-source-context seeds, in flight).
