# ROGII — NN Research Plan v2 (target: beat WARP 11.3 → PF 7 → Tucker 5.2)

## Where we are (established facts — don't re-litigate)
- `TVT = smooth_surface(MD) − Z`. Z is KNOWN in eval. Oracle: `surface_deg2 − Z = 3.9` on holdout.
- The wall = the **eval-zone surface trend** (it slope-CHANGES vs the known zone; not extrapolable: 43–73).
- **Global** surface reconstruction is ILL-CONDITIONED (GBM coeffs → 22; small slope err × long lever blows up).
- **Local** tracking is the only viable regime: WARP (per-point dTVT + continuity) = **11.3** (our best), but
  GR self-similarity blocks precise per-point localization → capped.
- Naive "line-oracle + high-freq(−Z)" = 7.3 (worse than 6.6): wrong decomposition. The right target is a
  **smooth surface predicted LOCALLY**, minus full Z.
- Tucker: **5ft, single model, per-well only** → a much better LOCAL tracker than ours exists.

## Methodology (keep — it works)
1. **Cheap premise-test first** (oracle/correlation, no training) before building any NN.
2. Build → validate on the **honest whole-well holdout** (flat 14.7, line 6.6, oracle 3.9). Fast on Kaggle T4.
3. Submit to Kaggle ONLY if holdout < 7.096 AND stable. Public LB is not the judge.
4. Every run → `research/RESULTS.jsonl`; every finding → memory.

## Hypotheses to test (prioritized by EV)

### H1 ★ Surface-space WARP done right (physics-regularized local tracker)
The surface-space model drifted to 30 ONLY because its zero-output prior = const-surface (105, bad) and
weak smoothness. Fix: predict `surface = known_linear_surface_prior + small NN correction`, output
`TVT = surface − Z`, with STRONG surface-smoothness penalty (surface is smooth by physics) and the
correction bounded/low-DOF. Good prior (near the answer) + smoothness + exact −Z. Premise already proven
(oracle 3.9); this is an optimization/regularization fix, high EV. **Build first.**

### H2 ★ Known-zone self-reference ruler (break GR self-similarity)
Match eval GR to the well's OWN known-zone GR (same tool/borehole → far less self-similar than a foreign
typewell), exploiting Eagle Ford rhythmic bedding (±15ft cycles repeat). Premise-test: cross-correlate
eval-GR windows vs known-GR windows; does it localize the within-cycle phase better than typewell (which
gave ~30/179)? If yes → a self-reference matcher could sharpen local tracking.

### H3 Change-point / piecewise-linear tracker (match human annotation)
The wall is slope-CHANGES (faults). TVT is human-drawn piecewise-linear (64.5% straight, quantized dips).
Model = detect breakpoints (from GR/trajectory events) + fit linear segments, output PL surface − Z.
Structurally matches both the physics (piecewise dip) and the labels. Premise-test: can any GR/trajectory
event detector localize the true breakpoints better than chance (earlier corr was 0.08–0.29 — retry with
learned multi-scale features)?

### H4 Multi-scale / shape GR features (better local discrimination)
Raw GR value is self-similar; multi-scale (wavelet) shape + derivative patterns may discriminate. Add as
channels to WARP. Cheap to add; test if it lowers WARP below 11.

### H5 Diverse WARP ensemble (the decorrelation win, reapplied)
Our only real LB gain (7.230→7.096) was a decorrelated ensemble. Train K WARP models (diff seeds,
augmentations, feature subsets) → average. Each ~11, decorrelated → maybe ~9-10. Structured, low-risk.

### H6 Richer trajectory features
Inclination, azimuth, dogleg, X/Y curvature — where the 3D path bends, the geology relation changes.
Underused (we only use dZ/dMD). Cheap to add to WARP; test.

### H7 Formation/Geology auxiliary (low odds, but untested in NN)
Typewell Geology as a datum-pinning signal + auxiliary formation-prediction head. GR→formation is 42.6%
(weak) and geology-Viterbi failed — low odds, but it's the one non-GR/non-trajectory signal. Test last.

## Premise-test update (just run)
- "line-oracle + high-freq(−Z)" = 7.3 (worse than 6.6): the winning decomposition is `smooth surface −
  full Z` (oracle 3.9), NOT line+wiggle.
- H1's known-linear-surface prior is itself 43–73 (eval slope ≠ known slope) → anchoring to it won't fix
  the drift. So the crux is NOT surface parameterization but **better local GR discrimination** (WARP's
  11 comes from local tracking; to beat it we must sharpen the per-point GR signal).
- **Reprioritized: start with H5 (ensemble, safe 11→~10) + H4 (multi-scale features) + H2 (self-reference
  matcher), then H3. H1 demoted** (surface reparam alone doesn't beat the slope wall).

## Milestones
M1: beat WARP 11.3 (any H) → M2: reach ~7 (match PF, submittable) → M3: <7 → 5.2 (top).

## Honest odds
H1 (surface-regularized) and H5 (ensemble) are the safest to move 11→~9-10. Reaching 7 needs a real
localization gain (H2/H3/H4). Reaching 5.2 likely needs Tucker's specific trick — but H1+H2+H4 stacked is
our best shot at closing the gap. Every localization signal still derives from GR/trajectory; the crux is
whether learned multi-scale/self-reference matching beats our current ~30 GR-alignment floor.
