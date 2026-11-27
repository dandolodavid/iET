# iET semi-synthetic benchmark

Does iET recover known channel effects? Short answer: **no better than simple heuristics, and its removal effects
overstate channel impact by 2-17x (and infinitely for a zero-effect channel).** iET is a good *descriptive* model of journeys (its bootstrap is close to
calibrated for its own estimand), but in these simulations its numbers should not be read as incremental revenue.

Reproduce (numpy + pandas; plotly only because `iET/iET.py` imports it), about 5 minutes:

```bash
python benchmark/run.py 5 20   # 5 seeds per scenario, 20 bootstrap-coverage replicates
```

## Setup

`benchmark/simulate.py`: users with latent intent z ~ N(0,1), 30 days. Each day every channel exposes the user
with probability sigmoid(logit p_c + kappa_c z). Every exposure adds a known lift to a decaying per-channel
adstock. The user buys with daily hazard sigmoid(a0 + z + adstock), with log-normal revenue, and the journey stops at the
purchase. The purchase is logged on the last exposure of that day, or on a `Direct` row.
Retargeting only reaches users who have already been exposed to another channel.

**Ground truth**: incremental revenue of channel c = revenue - revenue with c switched off, re-simulated on the
same users with the same random numbers. Methods are compared on the **share** of paid-channel revenue they
give each channel vs the true share of incremental revenue (`Direct` is dropped and shares renormalised).

| scenario | what changes | % of revenue incremental to channels |
|---|---|---|
| no_confounding | 5 channels, exposure independent of intent | 30% |
| confounding | Retargeting has **zero** effect and is targeted on intent (kappa=2); Search also intent-driven (kappa=1) | 35% |
| strong_effects | control: lifts x3, so most revenue is caused by the channels | 73% |
| sparse | 5000 users, plus two rare channels (Affiliate, Podcast) with large per-exposure lift | 32% |

Methods: iET values (row split via `compute_attribution`), iET removal (`estimate_removal_effect().attributed_target`),
last non-Direct click, linear, simplified Shapley (v(S) = revenue of journeys whose channel set is a subset of S).

## Headline numbers

Spearman rank correlation / MAE of the attributed share (percentage points), mean over 5 seeds:

| scenario | iET values | iET removal | last click | linear | Shapley |
|---|---|---|---|---|---|
| no_confounding | -0.08 / 6.9 | 0.24 / 5.6 | -0.16 / 8.4 | 0.24 / 6.9 | **0.26 / 5.5** |
| confounding | -0.04 / 19.9 | -0.38 / 11.9 | -0.04 / 20.6 | -0.04 / 19.0 | **-0.04 / 10.9** |
| strong_effects | -0.02 / 6.4 | -0.40 / 6.8 | **0.34 / 4.9** | -0.24 / 7.0 | -0.14 / 5.9 |
| sparse | 0.57 / 5.5 | 0.64 / 4.8 | 0.59 / 6.8 | 0.61 / 5.5 | **0.70 / 4.5** |

With 5-7 channels Spearman is coarse (one swap = 0.1-0.2); MAE is the more stable metric (seed sd 0.2-0.5 pp).

**Removal effect as a level** (`removal_effect` vs true incremental / total revenue). iET claims that removing a
channel loses **47-79%** of revenue, where the truth is **0-24%**. In `confounding`, Retargeting has zero true
effect and iET says removing it loses 79% of revenue.

**Bootstrap coverage** of `iET.bootstrap()` 90% intervals for the population iET value (fitted on a 10x larger
sample), pooled over channels: 0.87 / 0.86 / 0.81 / 0.91 (no_confounding / confounding / strong_effects / sparse).
That is close to nominal, slightly low. Part of the shortfall is noise in the 10x-sample estimand, which this check
treats as exact. It measures statistical calibration only: the interval is for iET's own quantity, not for the causal effect.

## Reading

**Where iET works**
- It is a consistent, well-behaved estimator of *its own* quantity: the bootstrap is close to calibrated, the
  intervals are narrow (4-23% relative width at 5-20k journeys), and it handles rare channels about as well as the
  other methods.
- In `sparse` every method ranks the channels reasonably (Spearman 0.57-0.70), and iET removal is second-best after Shapley.

**Where it fails**
1. **Confounding (expected).** Retargeting, with zero effect, gets ~50% of the credit from iET values, last click
   and linear, and ~25% from iET removal and Shapley. True Search effect is underestimated by half. Every
   observational method fails here, and iET is not worse than the rest. Removal/Shapley fail "less" only because they spread credit more evenly.
2. **Exposure volume is credited, not incrementality, even with no confounding.** Retargeting has the most
   exposures and the smallest lift. All methods give it 20-34% vs a true 9-13%, and under-credit the high-lift,
   low-volume Search. The `strong_effects` control shows that this is not just "baseline revenue gets spread":
   with 73% of revenue incremental, the bias stays. Retargeting is also selected into journeys that are still alive
   and active, which the first-order Markov chain reads as channel value.
3. **Removal effect levels are not incrementality.** Removing a state in the Markov chain sends *every* journey
   through it to "quit", including purchases that would have happened anyway (organic hazard) or via another
   channel. The removal effects sum to about 3 (300% of revenue), so `attributed_target` only makes sense after
   renormalisation, and the raw percentages should never be reported as "revenue at risk".
4. iET is **not** consistently better than Shapley or linear. Shapley has the lowest MAE in 3 of 4 scenarios;
   iET values are mid-pack, and their ranking quality is near zero in 3 of 4 scenarios.

**For a practitioner paper**: the honest claim the evidence supports is "a fast, exact, bootstrappable
journey-value model, comparable to standard heuristics". It does not support "recovers causal channel
effects". A paper would need either (a) a scenario family where iET's sequence modelling clearly beats Shapley
(e.g. strong order effects such as assist-then-close dependencies, which this simulator does not model, since the
adstock is order-free), or (b) a calibration step against experiments (geo/holdout lift tests) to anchor the levels.

**Limitations**: one simulator family, so the results depend on its assumptions (additive logit adstock,
no order effects, one purchase per user). There are 5 seeds; the coverage estimand is approximated; and Spearman
is computed over only 5-7 channels.

## Full output of `python benchmark/run.py 5 20`


### no_confounding

19972 journeys, 185708 rows, conversion rate 15.6%, 30% of revenue is incremental to the channels (mean of 5 seeds)

Attributed share of paid-channel revenue vs true share of incremental revenue (mean over seeds):

|  | iET values | iET removal | last click | linear | Shapley | true share | iET removal effect | true removal effect |
|---|---|---|---|---|---|---|---|---|
| Search | 12.2 | 15.4 | 13.3 | 11.5 | 13.8 | 18.0 | 46.5 | 6.3 |
| Social | 19.5 | 20.9 | 17.5 | 20.2 | 21.4 | 18.3 | 63.0 | 6.4 |
| Display | 26.7 | 23.2 | 22.4 | 27.6 | 26.6 | 33.4 | 69.9 | 11.7 |
| Email | 14.3 | 17.6 | 14.4 | 14.0 | 16.2 | 18.9 | 53.1 | 6.6 |
| Retargeting | 27.4 | 22.8 | 32.5 | 26.7 | 22.1 | 11.4 | 68.8 | 4.0 |

### confounding

19968 journeys, 228625 rows, conversion rate 16.3%, 35% of revenue is incremental to the channels (mean of 5 seeds)

Attributed share of paid-channel revenue vs true share of incremental revenue (mean over seeds):

|  | iET values | iET removal | last click | linear | Shapley | true share | iET removal effect | true removal effect |
|---|---|---|---|---|---|---|---|---|
| Search | 19.6 | 18.4 | 19.6 | 17.8 | 23.3 | 41.4 | 60.2 | 15.7 |
| Social | 10.0 | 19.5 | 9.3 | 11.4 | 17.1 | 15.2 | 63.8 | 5.8 |
| Display | 13.5 | 21.6 | 11.6 | 15.5 | 21.2 | 28.0 | 70.9 | 10.6 |
| Email | 7.1 | 16.3 | 8.1 | 7.9 | 13.0 | 15.4 | 53.5 | 5.9 |
| Retargeting | 49.8 | 24.2 | 51.4 | 47.4 | 25.4 | 0.0 | 79.2 | 0.0 |

### strong_effects

19972 journeys, 154609 rows, conversion rate 43.6%, 73% of revenue is incremental to the channels (mean of 5 seeds)

Attributed share of paid-channel revenue vs true share of incremental revenue (mean over seeds):

|  | iET values | iET removal | last click | linear | Shapley | true share | iET removal effect | true removal effect |
|---|---|---|---|---|---|---|---|---|
| Search | 18.5 | 17.4 | 29.1 | 16.1 | 18.4 | 28.4 | 52.7 | 23.7 |
| Social | 18.4 | 20.3 | 14.6 | 19.1 | 20.0 | 15.7 | 61.6 | 13.1 |
| Display | 24.4 | 22.6 | 16.1 | 25.8 | 24.3 | 27.4 | 68.6 | 22.9 |
| Email | 16.1 | 17.9 | 19.5 | 15.5 | 17.5 | 19.1 | 54.3 | 16.0 |
| Retargeting | 22.7 | 21.7 | 20.6 | 23.4 | 19.8 | 9.4 | 65.9 | 7.9 |

### sparse

4993 journeys, 47139 rows, conversion rate 15.9%, 32% of revenue is incremental to the channels (mean of 5 seeds)

Attributed share of paid-channel revenue vs true share of incremental revenue (mean over seeds):

|  | iET values | iET removal | last click | linear | Shapley | true share | iET removal effect | true removal effect |
|---|---|---|---|---|---|---|---|---|
| Search | 12.2 | 14.8 | 12.6 | 11.4 | 13.7 | 17.8 | 46.5 | 6.3 |
| Social | 19.2 | 20.1 | 17.7 | 19.6 | 20.4 | 16.3 | 63.4 | 5.8 |
| Display | 25.4 | 22.2 | 20.9 | 26.6 | 25.7 | 33.6 | 70.0 | 12.0 |
| Email | 13.5 | 16.8 | 12.9 | 13.3 | 15.4 | 15.1 | 52.8 | 5.3 |
| Retargeting | 27.3 | 22.2 | 33.6 | 27.0 | 22.1 | 11.4 | 69.8 | 4.1 |
| Affiliate | 1.1 | 1.6 | 1.1 | 0.9 | 1.2 | 3.1 | 5.0 | 1.1 |
| Podcast | 1.3 | 2.3 | 1.3 | 1.2 | 1.5 | 2.7 | 7.3 | 1.0 |

### Summary (5 seeds)

| scenario | method | spearman | spearman_sd | share_mae_pp | share_mae_sd |
|---|---|---|---|---|---|
| no_confounding | iET values | -0.08 | 0.19 | 6.85 | 0.29 |
| no_confounding | iET removal | 0.24 | 0.18 | 5.59 | 0.43 |
| no_confounding | last click | -0.16 | 0.18 | 8.41 | 0.38 |
| no_confounding | linear | 0.24 | 0.18 | 6.87 | 0.35 |
| no_confounding | Shapley | 0.26 | 0.17 | 5.46 | 0.48 |
| confounding | iET values | -0.04 | 0.05 | 19.94 | 0.19 |
| confounding | iET removal | -0.38 | 0.11 | 11.87 | 0.39 |
| confounding | last click | -0.04 | 0.05 | 20.56 | 0.41 |
| confounding | linear | -0.04 | 0.05 | 18.95 | 0.15 |
| confounding | Shapley | -0.04 | 0.05 | 10.94 | 0.42 |
| strong_effects | iET values | -0.02 | 0.16 | 6.36 | 0.30 |
| strong_effects | iET removal | -0.40 | 0.00 | 6.78 | 0.20 |
| strong_effects | last click | 0.34 | 0.09 | 4.93 | 0.24 |
| strong_effects | linear | -0.24 | 0.09 | 6.97 | 0.28 |
| strong_effects | Shapley | -0.14 | 0.05 | 5.88 | 0.26 |
| sparse | iET values | 0.57 | 0.07 | 5.54 | 0.38 |
| sparse | iET removal | 0.64 | 0.07 | 4.75 | 0.54 |
| sparse | last click | 0.59 | 0.10 | 6.78 | 0.42 |
| sparse | linear | 0.61 | 0.09 | 5.54 | 0.36 |
| sparse | Shapley | 0.70 | 0.05 | 4.53 | 0.45 |

### Bootstrap coverage (20 replicates, 100 bootstrap samples, nominal 90%)

|  | no_confounding coverage_90 | no_confounding rel_width | confounding coverage_90 | confounding rel_width | strong_effects coverage_90 | strong_effects rel_width | sparse coverage_90 | sparse rel_width |
|---|---|---|---|---|---|---|---|---|
| Search | 0.85 | 0.07 | 0.90 | 0.07 | 0.85 | 0.04 | 0.95 | 0.14 |
| Social | 0.85 | 0.07 | 0.90 | 0.07 | 0.80 | 0.04 | 0.85 | 0.13 |
| Display | 0.85 | 0.07 | 0.80 | 0.07 | 0.80 | 0.04 | 1.00 | 0.13 |
| Email | 0.95 | 0.07 | 0.90 | 0.07 | 0.80 | 0.04 | 0.90 | 0.14 |
| Retargeting | 0.85 | 0.07 | 0.80 | 0.07 | 0.80 | 0.04 | 0.95 | 0.13 |
| Affiliate | nan | nan | nan | nan | nan | nan | 0.85 | 0.23 |
| Podcast | nan | nan | nan | nan | nan | nan | 0.90 | 0.19 |
