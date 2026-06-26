# What changed: the size-deficit reframe (for author read-through + DESC re-review)

The internal-review version framed the recovered-galaxy-size deficit as a PSF under-concentration
to be **fixed by a Monte Carlo EM** (per-star contaminant inference + retraining). The current
version reframes the same deficit as a **PSF query-flux effect** and demotes the MCEM to a
Discussion cross-check. Nothing else in the paper changed (reserved-star headline, ρ-stats,
chromatic, sample efficiency, blend loss, galaxy-context, ensembling are all untouched).

## The finding that drove the change

The model conditions on source flux. Trained on stars that carry faint unresolved neighbours, it
learned that **fainter stars are more fractionally contaminated** — so its predicted PSF *narrows
toward the clean PSF as the query flux rises*. Galaxy recovery had been rendering the PSF at the
**median star flux**, where it is contamination-broadened. Rendering at the **clean high-flux limit**
recovers it. Evidence:

- **Sim, truth-anchored, no brighter-fatter (the decisive proof):** δEE vs *known* truth goes
  −0.0078 (median flux) → −0.0014 (clean flux), i.e. essentially onto truth, with no cleaning.
- **Real, direct PSF size (HSM T) vs bright clean reserved stars, n=260/50 exp:** δT/T +13.4%
  (median) → +0.0±0.4% (10⁶); clean window broad and flat 5×10⁵–6×10⁶ (BF +0.2%/decade). → tab:fluxsize.
- **Real galaxy recovery, 16 matched held-out exp, n=316:** de-confounded implicit−truth −5.1%
  (median) → −0.7% (clean), to the protocol floor. → fig:galrec, sec:galrec text.

## Why the MCEM was demoted (residual experiment)

The MCEM-cleaned sim model reaches δEE −0.0006 at its natural flux; the flux query reaches −0.0014
— **the same truth, within seed scatter**. The MCEM does not beat the flux-query floor, and the two
do not stack (cleaning + high-flux query over-corrects to +0.0034). So the simple query is the
operative correction; the MCEM is an honest principled cross-check that confirms the mechanism.

## Section-by-section edits

- **Abstract:** deficit reframed as query-flux; MCEM sentence removed.
- **Method:** the entire `\subsection{Contamination correction by Monte Carlo EM}` (model, latents,
  MCEM, sampler-validation) **deleted**.
- **§Galaxy recovery (sec:galrec):** the deficit paragraph rewritten as the query-flux effect;
  added **tab:fluxsize** (δT/T vs query flux); tab:galrec caption notes its −7.6% is the median-flux
  render; **fig:galrec regenerated** with a "Neural PSF (clean flux)" arm showing the recovery.
- **§Simulation studies (sec:sim):** the MCEM paragraph trimmed to the truth-anchored flux-query
  result (heading renamed "The size deficit reproduces in simulation and is a query-flux effect");
  the (λ,α)-inference / SBC / diagnostics content removed.
- **Discussion (iv):** rewritten around the query-flux mechanism; one honest paragraph retains the
  MCEM as a principled cross-check that "reaches the same truth as the flux query … without
  improving on the simpler query." (Fixed a now-backwards claim that the flux–size trend "does not
  propagate"; dropped a config-confounded "higher-capacity encodings reduce it" claim.)
- **Conclusion:** MCEM clause removed; query-flux prescription stated.

## Honest caveats a referee will probe (for discussion before submission)

1. **DESC internal review predates this reframe** — the acknowledged internal review was of the
   MCEM-headline version; the reframed paper should be re-reviewed.
2. **Real-data validity rests on bright stars being clean.** The real δT/galaxy results compare to
   bright stars, which are themselves somewhat contaminated (and selection can leave bright survivors
   with masked contamination). The *simulation* (known truth, no BF) is what closes this — worth an
   explicit sentence if desired.
3. **Galaxy-recovery N = 16 exposures** (n=316) is solid for the effect size but thinner than the
   32-exp reserved-star tables; a larger-N run is in progress to remove the asymmetry.
4. Minor: 3 now-unused MCEM bib entries (booth/levine/delyon) can be dropped; final
   author/affiliation/funding check; target-journal formatting.
