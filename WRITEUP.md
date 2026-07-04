# ROGII Wellbore Geology Prediction — Solution & Research Write-up

**A rigorous map of *why this problem is hard*, an honest account of ~30 experiments, and the one
change that produced a genuine leaderboard gain.**

TL;DR — We started from the public PF pipeline (Public LB ≈ 7.23), added a *decorrelated
particle-filter ensemble* that pushed us to **7.096** (below the pipeline's own seed-noise floor,
so a real gain — not luck), and then spent most of our effort answering a harder question:
**how much of the remaining error is actually learnable?** Our answer, confirmed ~9 independent ways
and by 7 neural architectures: **most of it is not.** The eval-zone TVT is dominated by a drift whose
*direction* is physically absent from the observable data. We think this is the most useful thing we
can share, because it reframes what "a good score" means on this task.

---

## 1. The task and the metric
Predict `TVT` (true vertical thickness / stratigraphic position) for the unlabeled **eval zone** — the
toe-side tail of a horizontal wellbore — given the well's GR log + trajectory and a reference typewell.
Score = **pooled RMSE (ft)** over all eval points of all test wells.

Two metric facts shaped everything we did:
- **Pool, don't average per well.** Per-well averaging gives optimistic ~10; the competition pools
  points, which is ~15.9 for the trivial baseline. We report pooled RMSE throughout.
- **The trivial "flat" baseline is strong.** Predicting `TVT = last known TVT` scores **15.883** on
  the LB (15.1 on our local holdout). *Almost every "smarter" idea we tried did worse than flat.*

## 2. Validation methodology (our most reusable contribution)
Public LB here is deceptive: the PF-based cluster (the ~7.2 fork family) is dominated by **seed
variance** — byte-identical reruns scatter between ~7.17 and ~7.5. Tuning on Public LB is tuning on
noise. So we built a **trustworthy local holdout**:
- 160 whole, unseen wells (grouped split, seed 42); pooled RMSE on their eval zones only.
- **Sanity anchor:** our holdout's flat baseline = **15.1**, vs the LB's 15.883 — the scale matches,
  so relative model rankings on the holdout are trustworthy.

**Rule we adopted (after a painful lesson, see §6): submit to Kaggle only a model that is confidently
better than the incumbent on this holdout AND stable across folds/seeds.** Public LB is not the judge.

## 3. What actually moved the needle: a decorrelated PF ensemble
The public pipeline = particle filter (128 seeds) + beam-search DP/Viterbi alignment +
spatial FormationPlaneKNN/IDW + a LightGBM+CatBoost→Ridge stack + a "gold" visible-prefix calibration.
Our **genuine additions**:
1. Reverting an over-aggressive GR denoise step (it *looked* +2.8% on an isolated PF test but was
   **−0.24 on the real LB**: 7.475 → 7.230). First hard evidence that isolated component tests mislead.
2. A **diverse, decorrelated PF ensemble**: blend the base PF with a *stiffer* PF configuration
   (lower process noise, `MOM/VN/PN = 0.999/0.001/0.002`) at a 0.65/0.35 weight inside the selector.
   The two configurations make *independent* errors, so averaging reduces variance.
   → **Public LB 7.230 → 7.096.** Crucially this is **below the documented seed-noise floor (7.168)**,
   so it is a real improvement, not a lucky seed.

Why this transferred when other things didn't: it is a **GBM-independent, structural** change to the
position estimator. Changes that touched the GBM's role (blends, denoise) were redundant with the
existing stack and hurt; variance-reduction on the PF itself carried over cleanly.

### 3b. Where decorrelation stopped
We pushed the same idea further — a 3rd partner decorrelated by a *different* mechanism
(GR-sensitivity), and a 4-way "kitchen-sink" blend. On our 3-slice holdout the 4-way looked best
(9.49 vs 9.57). **On the real LB it was the worst (7.752).** Lesson repeated: even multi-slice holdout
ranking can fail to transfer when all slices share a blind spot (here: isolated-PF, no full pipeline).
The single-partner ensemble (7.096) was the sweet spot; more dilution of the tuned base only hurt.

## 4. The central finding — "the wall"
Why is flat so hard to beat? Because the eval-zone TVT behaves like a **near-random walk around the
last known value, whose drift *direction* is not present in the observable data.** We confirmed this
from ~9 independent angles:

| # | Probe | Result | Meaning |
|---|---|---|---|
| 1 | Flat baseline | 15.1 | strong; drift has small predictable mean |
| 2 | Linear / dip / surface extrapolation | 37–85 | trends do **not** persist → worse than flat |
| 3 | Inter-well drift correlation w/ nearest neighbor | −0.08 | neighbors don't tell you the direction |
| 4 | Bimodal tie-break (±15 ft Eagle Ford rhythmic bedding) | r=0.054, mode correct 48.8% | a coin flip |
| 5 | PF residual vs any observable | all \|corr\| < 0.15 | we know *where* error is (seed-spread), not *which way* |
| 6 | Surface breakpoints vs GR shift | 0.082 | fault locations not in GR |
| 7 | Breakpoints vs formation-column curvature | 0.293 (and formation cols are removed in test) | weak & unavailable |
| 8 | Oracle GR window registration | ~31 ft | pure GR matching is weak (GR is self-similar) |
| 9 | Learned contrastive GR↔typewell matcher (localization) | **179 ft** | without a continuity prior, GR localizes *nowhere* |

The picture is consistent: **the exploitable signal is the continuity prior (stay near the last
position); GR only nudges locally, and the sub-seismic fault drift that dominates the error is
physically not observable.** The particle filter's ~7 comes almost entirely from continuity + a
carefully tuned (deliberately *weak*) GR likelihood + the beam/GBM/calibration stack — not from strong
GR localization.

### 4b. The labels are human-drawn — and that doesn't rescue us
TVT was drawn by geologists in steering software, not measured. We verified the human signature:
**64.5%** of points have ~zero curvature (long straight segments = piecewise-linear), segment dips are
**quantized** to round values (median exactly 0.0200 dTVT/dMD), and breakpoints are structural
(corr(|d²TVT|, |d²surface|)=0.80). Tempting — but the *breakpoint locations* (= sub-seismic faults)
are exactly the thing probe #6/#7 show is **not recoverable** from observable GR. Snapping predicted
dips to the round grid only discretizes our error (10.35 → 10.42–12.15, all worse). Real structure,
no post-hoc handle.

## 5. Can a neural network beat the tuned PF? (7 architectures, honest negatives)
The leading solutions (≈5.6) are reportedly regularized CNNs that generalize well on local holdout.
We took that seriously and built a proper research harness (773-well feature cache, whole-well holdout,
augmentation: GR calibration jitter / noise / channel-dropout). Every from-scratch architecture we
tried **fails to beat flat** on the honest holdout:

| Architecture | Holdout pooled RMSE | Failure mode |
|---|---|---|
| flat baseline | 15.1 | — |
| Dilated TCN, point-wise TVT-delta regression | 15.0 | no per-point signal → predicts ≈flat |
| Neural matcher, static continuity anchor | 14.95 | collapses to flat (anchor window shrinks to ~10 ft) |
| Neural matcher, iterated re-centering | 101 | diverges (positive feedback to wrong matches) |
| GRU sequential state-estimator (autoregressive obs) | 14.7 | fits known zone, tiny safe steps on eval ≈ flat |
| Contrastive GR↔typewell window matcher | 179 (localization) | GR self-similarity, no continuity |
| **Differentiable grid Bayesian filter (neural PF, GPU)** | **24.5** | learned GR emission is **net-harmful** |

The last one is the punchline. A discretized, end-to-end-trained particle filter (full posterior over
256 typewell positions, learned continuity transition + learned GR emission, renormalized each step)
is the *right* architecture — multi-hypothesis, can't collapse or diverge. It **does** track position,
yet converges to flat **from above (24.5)**: turning the GR emission *off* (pure diffusion) would give
≈flat=15, which is **better**. In other words, on this data a learned GR likelihood **subtracts value**.
The production PF wins (7.096) only because its Gaussian likelihood is hand-tuned to be *just weak
enough* to help without misleading — a razor's edge a from-scratch model doesn't find — plus the full
classical stack around it.

**Takeaway for the community:** the ceiling for public-data methods on this task appears to be set by
physics, not modeling effort. A strong score is mostly a well-tuned continuity prior with a very light
GR touch; heavier "learning" of the GR→position map tends to hurt.

## 6. Lessons (the ones we'd want to read first)
1. **Isolated component tests lie.** Denoise (+2.8% isolated → −0.24 LB), GBM blends, and a 4-way
   ensemble all looked good in isolation and lost on the LB. Only *full-pipeline / whole-well holdout*
   evidence transferred.
2. **Public LB ≈ seed noise** for the PF cluster. We treated any change < ~0.07 as noise.
3. **Know where the error is vs. which way to fix it.** We can *predict error magnitude* (seed-spread
   correlates +0.48 with error) but not its *sign*. That distinction is the whole game here.
4. **Variance reduction beats cleverness.** The only robust gain came from decorrelating the estimator,
   not from new physics or features.

## 7. Final submissions & why
We selected **7.096 (decorrelated PF ensemble)** + the **robust base (7.220)** as our two finals —
a performance pick plus a maximally-different insurance pick. Given the competition is 74% private and
*designed* to punish Public-LB overfitting, we expect this disciplined pair to hold (or climb) on
private while over-optimized entries regress.

## 8. Reproducibility
- Full pipeline notebooks, every experiment logged in `TESTS_LOG.md`, the NN research harness and
  roadmap (`NN_RESEARCH_PLAN.md`), and per-run results (`research/RESULTS.jsonl`) are included.
- Companion datasets required: the competition data + the two public artifact datasets the fork uses.
- Every number above is from the honest whole-well holdout or the real Public LB, never per-well
  averaging.

*If you take one thing from this write-up: on this task, measure twice on a whole-well holdout, distrust
Public LB, and respect the wall — a lot of "improvement" is the estimator's own seed variance.*
