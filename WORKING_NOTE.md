# Why This Problem Is Hard: A Rigorous Negative-Result Study of Wellbore Stratigraphic-Position (TVT) Prediction

**ROGII Wellbore Geology Prediction — Working Note**

---

## Abstract

We present a systematic study of the ROGII wellbore-geology task: predicting the true vertical thickness
(TVT, stratigraphic position) of the unlabeled eval-zone tail of a horizontal well from its gamma-ray
(GR) log, trajectory, and a reference typewell. Our contribution is threefold. **(1) Methodology:** a
whole-well holdout whose trivial-baseline error matches the public leaderboard (15.1 vs 15.9), making it
a trustworthy stand-in for the hidden test — unlike the public LB, which for the dominant particle-filter
(PF) solution family is dominated by *seed variance*. **(2) A positive result:** a *decorrelated PF
ensemble* that improves the public pipeline from 7.230 to **7.096**, below the pipeline's own seed-noise
floor (7.168), i.e. a real gain. **(3) A rigorous negative-result analysis** — the bulk of this note —
mapping *why* the residual error is hard: we show, via an exact decomposition and 20+ neural
architectures, that the exploitable error is the low-frequency per-well *surface trend*, that GR
localization error is **broadband and irreducible** due to log self-similarity, and that a weak-GR /
strong-continuity prior (which the PF encodes) provably beats explicit GR matching. We believe these
negative results, quantified and explained, are the most useful thing we can offer the community, and
they directly reframe what a "good score" means on this task.

---

## 1. Task and Metric

Predict `TVT` for the eval zone — the toe-side tail of a horizontal wellbore — given the well's GR log
along measured depth (MD), its trajectory (`X,Y,Z,MD`), the known-zone `TVT_input` near the heel, and a
reference typewell (`TVT` vs `GR` vs `Geology`). The metric is **pooled RMSE (ft)** over all eval points
of all test wells.

Two metric facts shaped everything:
- **Pool, don't average per well.** Per-well averaging gives an optimistic ~10; the competition pools
  points, ~15.9 for the trivial baseline. All numbers here are pooled RMSE.
- **The trivial "flat" baseline is strong.** Predicting `TVT = last known TVT` scores **15.883** on the
  LB (15.1 on our holdout). *Almost every "smarter" idea we tried did worse than flat.*

## 2. Validation Methodology (our most reusable contribution)

The public LB is deceptive: the PF-based cluster (the ~7.2 fork family) is dominated by **seed variance**
— byte-identical reruns scatter between ~7.10 and ~7.5. Tuning on public LB is tuning on noise. We
verified this directly by resubmitting an unchanged pipeline: it scored 7.096, 7.135, and 7.091 on
separate runs. We therefore built a **whole-well holdout**: 160 unseen wells (grouped split), pooled RMSE
on their eval zones only. Its flat baseline = **15.1**, versus the LB's 15.883 — the scale matches, so
relative model rankings transfer. **Rule adopted:** submit to Kaggle only a model confidently better than
the incumbent on this holdout *and* stable across seeds. This rule saved us from a "4-way ensemble" that
looked best on multi-slice holdout (9.49) but was worst on the real LB (7.752), because all slices shared
an isolated-component blind spot.

## 3. A Positive Result: Decorrelated Particle-Filter Ensemble

The public pipeline = PF (128 seeds) + beam-search DP/Viterbi alignment + spatial FormationPlaneKNN/IDW +
a LightGBM+CatBoost→Ridge stack + a "gold" visible-prefix calibration. Our genuine additions:
1. **Reverting an over-aggressive GR denoise** (looked +2.8% on an isolated PF test, but −0.24 on LB:
   7.475→7.230) — first evidence that isolated-component tests mislead.
2. **A diverse, decorrelated PF ensemble:** blend the base PF with a *stiffer* configuration (lower
   process noise) at a 0.65/0.35 weight. The two make independent errors → averaging reduces variance.
   → **7.230 → 7.096**, below the seed-noise floor (7.168): a real improvement, not luck.
3. **Variance reduction by more seeds** (128→160→192): converges toward the true ensemble mean, less
   luck-dependent → 7.091, and a more robust submission for the private split.

**Where decorrelation stopped:** a 3rd/4th partner (different mechanisms) or an off-base weight always
hurt on the LB. The single-partner ensemble was the sweet spot; more dilution of the tuned base only hurt.

## 4. The Central Finding: The Error Is (Mostly) Physically Unpredictable

### 4.1 An exact decomposition: the wiggle is free, the trend is the wall
`TVT = surface(MD) − Z`, where `surface = TVT + Z` is nearly linear (R²≈0.99) and **Z is known exactly in
the eval zone.** On the holdout, an oracle that knows only the *smooth* surface (a degree-2 fit of the
true surface) and subtracts the known Z reaches **RMSE 3.0–3.9** — *below* the per-well-line oracle (6.6)
and below the leaders (5.2–5.4). Interpretation: **the entire high-frequency "wiggle" of TVT is carried
by −Z, for free; the only hard part is the smooth, low-frequency per-well surface trend.**

### 4.2 The surface trend is not recoverable
That trend is the wall. It **changes** between the known and eval zones (the well crosses sub-seismic
faults where dip changes), so it does not extrapolate: linear extrapolation of the known surface gives
43–73; a GBM predicting the surface coefficients gives 22 — because reconstructing TVT from a few *global*
coefficients is **ill-conditioned** (a tiny slope error × the long eval lever arm blows up). Only *local*
tracking is well-conditioned.

### 4.3 GR localization error is broadband and irreducible
Why can't GR pin the trend? Because the log is **self-similar**. We measured this from many angles: a
learned contrastive GR↔typewell matcher localizes to only **179 ft** (vs a ~31 ft full-sequence oracle);
decomposing the aligner's error shows removing an oracle constant (31), oracle linear (31), or the low
frequencies (28) barely helps — the error has **roughly equal power at all frequencies**. No filter,
low-pass, anchor, or continuity decode recovers it. This is the deepest form of the wall: GR corrupts
localization at *every scale*.

### 4.4 The recurring principle
Across every experiment, **a weak-GR / strong-continuity prior beats explicit GR matching.** The PF (and
our best NN) anchor to the last known position and only lightly correct with GR → ~7–11. Approaches that
*trust* GR matching (alignment, 2D contour) → ~30. The exploitable signal is continuity, not GR.

### 4.5 The labels are human-drawn — and that doesn't rescue us
TVT was drawn by geologists, not measured: 64.5% of points have ~zero curvature (piecewise-linear),
segment dips are quantized (median exactly 0.0200), and breakpoints are structural (corr(|d²TVT|,
|d²surface|)=0.80). Tempting — but the breakpoint locations (= faults) are exactly what §4.3 shows is not
recoverable from GR; snapping predicted dips to the round grid only discretizes the error (10.35→10.4+).

## 5. Neural-Network Study: 20+ Architectures, and Why They Cap

We built a research harness (773-well feature cache, whole-well holdout, augmentation) and evaluated a
broad architecture space. **Every from-scratch model caps near ~11 = the isolated-PF level.**

| Architecture | Holdout RMSE | Failure mode |
|---|---|---|
| flat baseline | 15.1 | — |
| Point-wise TCN (regress TVT) | 15.0 | no per-point signal → ≈flat |
| Neural matcher (static anchor) | 14.95 | collapses to flat |
| Neural matcher (iterated) | 101 | diverges (positive feedback) |
| GRU sequential filter | 14.7 | tiny safe steps ≈ flat |
| Contrastive GR↔typewell matcher | 179 ft | GR self-similarity |
| Neural grid Bayesian filter (PF) | 24.5 | learned GR emission net-harmful |
| **WARP (dTVT integration + cross-attn)** | **11.3** | ★ best; continuity-anchored |
| WARP + multi-scale GR + EMA | 11.2 | marginal |
| Surface-space (−Z) model | 30 | drift; ill-conditioned |
| Alignment (soft-argmax over typewell) | 30 | GR wall |
| Hybrid (WARP + low-passed align anchor) | 11.5 | align adds nothing |
| Mixture-Density Network (K=3) + Viterbi | 14 | fixes mean-collapse but weak |
| Synthetic pretrain (naive) | 13 | too-clean GR → no transfer |
| Joint real + synthetic augmentation | 11.6 | ≈WARP, less overfit |
| 2D misfit-heatmap + SDF U-Net | 29 | GR wall |
| 2D-SDF + Viterbi continuity decode | 32 | continuity can't pick the true contour |
| Transformer (global self+cross attention) | 13.4 | overfits 773 wells |
| Learned stack / distillation of PF signals | 10.7 | = isolated-PF (signals correlated) |

**Why WARP (11) is best:** it predicts a per-step derivative integrated from the known anchor
(well-conditioned, cannot drift far) and treats GR as a *weak* corrector — the weak-GR/strong-continuity
principle of §4.4. **Why transformers do worse (13.4):** with only 773 wells, global attention overfits;
the CNN's locality is the correct inductive bias here. **Why 2D-SDF (the geosteering-image approach) caps
at ~30:** the misfit image has *many* plausible continuous zero-contours (GR self-similarity), and neither
a 2D receptive field nor a Viterbi continuity decode can pick the true one — there is no signal to
disambiguate. **Why naive synthetic pretraining fails:** a generator with clean `GR=typewell_GR(TVT)` is
trivially invertible, so the pretrained model learns an inversion that collapses to flat on real,
self-similar GR; realistic synthetic (matching the real ambiguity) is required and is the open engineering
problem.

## 6. Uncertainty Estimation

Two usable uncertainty signals emerged. **(a)** The PF seed-spread correlates **+0.48** with the actual
error magnitude — we can predict *where* the model is uncertain (though not *which way* to correct, see
§4). **(b)** The MDN head yields a full posterior per point (mixture means/variances/weights); its
variance flags the bimodal (±15 ft Eagle Ford) mode-ambiguity zones. Crucially, we found the bimodal
tie-break is a coin flip (mode correct 48.8%, r=0.054) — so the honest uncertainty statement is that a
large share of the error is *aleatoric* (irreducible from the observations), which itself is a useful,
calibrated conclusion.

## 7. Physical Meaningfulness

Our best components are physically grounded: the decomposition `TVT = surface − Z` is the geosteering
datum relation; the winning ensemble is variance reduction over a Bayesian sequential filter; the WARP
model reads the typewell as a "ruler" via cross-attention (the computational analogue of a geologist
correlating logs). The central negative result is itself physical: the eval-zone dip changes at
sub-seismic faults whose locations are not encoded in the observable GR, so the residual error is a
property of the geology and the measurement, not of modeling effort.

## 8. Lessons for the Community

1. **Isolated-component tests lie.** Denoise (+2.8% isolated → −0.24 LB), GBM blends, and a 4-way ensemble
   all looked good in isolation and lost on the LB. Only full-pipeline / whole-well holdout evidence
   transferred.
2. **Public LB ≈ seed noise** for the PF cluster; treat any change < ~0.07 as noise, and prefer
   variance-reduced (more-seed) submissions for private-split robustness.
3. **Know *where* the error is vs *which way* to fix it.** Seed-spread predicts error magnitude (+0.48);
   nothing predicts its sign. That distinction is the whole game.
4. **Variance reduction beat cleverness.** The only robust gain came from decorrelating the estimator, not
   from new physics, features, or architectures.
5. **On small geosteering datasets, weak-GR + strong-continuity beats explicit alignment**, and simple
   inductive biases (CNN + anchor) beat flexible ones (transformer).

## 9. Reproducibility

Every number is from the whole-well holdout or the real public LB, never per-well averaging. The pipeline
notebooks, the full experiment log (`TESTS_LOG.md`), the NN research harness and roadmap
(`NN_RESEARCH_PLAN.md`, `NN_PLAN_V2.md`), the final NN analysis (`NN_FINAL_ANALYSIS.md`), and per-run
results are included. Companion datasets: the competition data + the two public artifact datasets the fork
uses. We build on the public PF pipeline and the diagnostic framing of the two traps (CV→LB mirage +
seed/refork variance) and the field-grouped "wall test" due to Georgy Mamarin, independently reproduced
by another competitor (wharekawa); our findings converge with theirs.

## 10. Conclusion

The ROGII task decomposes cleanly: the high-frequency TVT wiggle is free (it equals −Z, which is known),
and the entire difficulty is a smooth per-well surface trend whose slope-changes occur at sub-seismic
faults that the self-similar GR log cannot localize (broadband-irreducible error). This is why the trivial
flat baseline is so strong, why a weak-GR/strong-continuity prior wins, and why 20+ neural architectures —
including MDN, synthetic-pretraining, 2D misfit-SDF, and transformers — all cap at the isolated-PF level
(~11) while the tuned classical pipeline reaches ~7 through its full stack. Our concrete, transferable
gain came not from new modeling but from *variance reduction* (decorrelated PF ensemble, 7.230→7.091).
We hope the map of *where the signal is and is not* is useful: on this task, measure twice on a whole-well
holdout, distrust the public LB, and separate the recoverable (the −Z wiggle) from the physically
unrecoverable (the cross-field surface trend).

---

*Acknowledgements:* public PF pipeline and the two-traps / wall-test diagnostic framing by Georgy Mamarin;
independent reproduction by wharekawa; leaderboard context from Rishikesh Jani and Tucker Arrants.
