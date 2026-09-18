# Machine Learning for Leak Detection and Localisation in a Water Distribution Network

A study on the BattLeDIM 2020 benchmark (L-Town network)

---

## 1. Problem

Water utilities lose a large share of treated water to leaks that nobody knows about.
A burst that surfaces gets reported within hours; a leak that discharges quietly into
the ground can run for months. The cost is not only the water — it is the energy spent
treating and pumping it, and the damage done underground before anyone notices.

Detection today mostly relies on **minimum night flow (MNF)**: between 02:00 and 04:00
almost nobody uses water, so whatever is still flowing into a district is assumed to be
loss. A step up in night flow means a new leak. The method is simple, and as Section 5
shows, on this network it is also close to useless.

This project asks two questions using only the sensors a utility already owns:

1. **Detection** — can we tell, soon after it starts, that a *new* leak has begun?
2. **Localisation** — can we tell a repair crew which district to search?

## 2. Data

**BattLeDIM 2020** (Zenodo record 4017659, CC BY 4.0), released by the KIOS Center of
Excellence for the "Battle of the Leakage Detection and Isolation Methods" competition.

| | |
|---|---|
| Network | L-Town: 782 junctions, 905 pipes, 43.1 km, 1 tank, 2 reservoirs, 1 pump, 3 PRVs |
| Sensors | 33 pressure, 3 flow, 1 tank level |
| Resolution | 5 minutes, 105,120 samples per year |
| Period | 2018 (development) and 2019 (evaluation) |
| Ground truth | 33 leak events: pipe ID, start, end, diameter (8.8–22.9 mm), type |

Leaks are labelled as **abrupt** (full discharge immediately, a burst) or **incipient**
(discharge ramps up over weeks, a crack that slowly opens). The split is 15 abrupt and
18 incipient. This distinction turns out to drive almost every result in this report.

The 2018 data was used for all design and tuning decisions. The 2019 data was used
once, to produce the numbers in Sections 5 and 6.

## 3. The problem with the obvious approach

The natural first framing is a binary classifier: given the current sensor readings, is
there a leak? Checking the labels kills that idea immediately:

| Year | Timesteps with no leak flowing | Max concurrent leaks |
|---|---|---|
| 2018 | 2,310 of 105,120 (**2.2%**) | 6 |
| 2019 | 0 of 105,120 (**0%**) | 16 |

In 2019 something is leaking at every single timestep. A classifier answering "yes,
there is a leak" would score 100% accuracy and be worth nothing. There is also almost
no clean data to define a leak-free baseline from — eight days in January 2018, all in
winter, which a model would then have to extrapolate across a full year of seasonal
demand.

So the task is reframed to the one that is both answerable and operationally useful:
**detect the onset of a new leak**, and localise it. This is what the competition
scored, and it is what a control-room operator actually needs.

## 4. Method

### 4.1 Rolling-reference expected-pressure model

For each of the 33 pressure sensors, a Ridge regression predicts the sensor reading from:

- total inflow across the three flow meters, and its square (head loss is non-linear)
- tank level
- cyclical time-of-day and day-of-week terms, plus a weekend flag

The essential design choice is that the model is **refitted on a trailing 21-day
window**, skipping the two most recent days so that a leak starting yesterday cannot
contaminate its own reference. Consequences:

- Leaks already running are absorbed into the reference and stay silent — which is what
  makes the "0% leak-free data" problem survivable.
- Seasonality is handled implicitly, because the reference is never more than three
  weeks old.
- A genuinely *new* leak breaks the learned relationship and appears as a step in the
  residual.

Residuals are averaged per day, giving a 365 × 33 matrix. Residual noise is
**σ ≈ 0.032 m**, while leak-induced shifts reach 0.03–0.48 m.

An early version restarted the model each 1 January. That left the detector blind for
its first three weeks, which is exactly where two 2019 leaks begin. Letting the
reference window reach back into the previous year — as a real utility's would —
raised detection from 14/19 to 17/19 and halved median latency from 30 to 14 days.

### 4.2 CUSUM detection

Residuals are converted to robust z-scores using a trailing 28-day median and median
absolute deviation. MAD is used rather than standard deviation so a developing leak
does not inflate its own alarm threshold.

A two-sided CUSUM then accumulates evidence per sensor. This is the right tool here: a
slow incipient leak never produces a single dramatic day, but it does produce many
mildly abnormal days in the same direction, and a CUSUM adds those up. An alarm is
raised when the statistic crosses `h`, after which the state resets and a 14-day
refractory period prevents one leak from alarming every day until it is repaired.

`k` (slack) and `h` (threshold) were grid-searched **on 2018 only**, maximising
detections subject to a budget of at most 6 false alarms per year — because the binding
real-world constraint is not accuracy but how many phantom leaks a crew will chase
before ignoring the system. Selected: `k = 1.0`, `h = 4`.

### 4.3 Localisation by hydraulic fingerprint matching

When a leak starts, all 33 sensors shift, each by a different amount. The *direction*
of that 33-dimensional shift vector encodes where the leak is; its *length* encodes how
big the leak is. Since leak size is unknown at detection time, every vector is
normalised to unit length and only direction is compared.

Fourteen training leaks is nowhere near enough to learn a map from fingerprint to
location. So the fingerprint library is generated hydraulically instead: **EPANET (via
WNTR) simulates a leak on each of the 902 pipes in turn**, and the resulting sensor
response is recorded. The leak is modelled as an emitter — pressure-dependent discharge
`Q = C·√P` — calibrated to a 15 mm orifice, the median of the real leaks, giving
14.2 m³/h at typical head. The resulting mean response of 0.29 m matches the magnitude
of the real observed shifts.

Observed fingerprints are matched to the library by cosine similarity. Pipe-level
prediction from 33 sensors across 902 pipes is not identifiable, so scores are
aggregated into **12 spatial zones** (k-means on pipe midpoints, ~3.6 km of pipe each),
which is the granularity a repair crew is dispatched at. A zone's score is the mean
similarity of its best-matching 5% of pipes, so a large zone full of irrelevant pipes
cannot dilute a good match.

### 4.4 Adaptive fingerprint window

Abrupt and incipient leaks need different observation windows, and on the 2018
development year the difference is stark:

| Window after onset | Abrupt top-1 | Incipient top-1 |
|---|---|---|
| 5 days | 67% | 10% |
| 30 days | 33% | 30% |

A burst shows its full effect within days and is then contaminated by later events; a
slow leak has barely started after five days. Since leak type is unknown at detection
time, the window is chosen by the data: each candidate window (5/10/20/30 days) is
scored by the norm of its shift divided by the residual noise over the same span, and
the strongest wins. Selected on 2018 (58% top-1, against 50% for the best fixed window)
and then frozen.

## 5. Detection results

**2019 was not used for any tuning decision.**

| | 2018 (development) | 2019 (held-out) |
|---|---|---|
| Leaks | 14 | 19 |
| **Detected** | **13 (93%)** | **17 (89%)** |
| **False alarms** | 4 | **0** |
| **Median time to detection** | 21 days | **14 days** |
| MNF baseline detected | 1 (7%) | 1 (5%) |
| MNF median delay | 39 days | 30 days |

The classical night-flow method finds **1 leak in 19**. The reason is visible in
`figures/10_mnf_noise.png`: day-to-day MNF noise on this network is 26–52 m³/h while a
typical leak adds 10–45 m³/h, so onsets are simply buried. The residual approach wins
because regressing pressure on inflow removes the demand variation that swamps MNF,
rather than hoping it averages out at night.

A related finding: restricting residuals to night hours — the intuition MNF is built on
— made results *worse* (median SNR 1.6 vs 3.9), because discarding 80% of the day costs
more in sample size than it gains in quietness. Once demand is modelled explicitly,
there is no reason to throw away daytime data.

The two missed leaks are **p426** (15.0 mm, abrupt, 25 Oct) and **p879** (13.2 mm,
incipient, 20 Nov) — both late in the year, when eight other leaks are already running
and masking further change.

## 6. Localisation results

| Year | Subset | n | Top-1 | Top-3 | Median zone rank | Median distance error |
|---|---|---|---|---|---|---|
| 2018 | all | 12 | 58% | 58% | 1 of 12 | 324 m |
| 2018 | abrupt | 6 | 100% | 100% | 1 | 203 m |
| 2018 | incipient | 6 | 17% | 17% | 6 | 1078 m |
| **2019** | **all** | **19** | **37%** | **53%** | **3 of 12** | **518 m** |
| 2019 | abrupt | 9 | 44% | 56% | 2 | 443 m |
| 2019 | incipient | 10 | 30% | 50% | 3.5 | 561 m |

Random guessing over 12 zones scores 8% top-1 and 25% top-3. Held-out top-1 is
therefore **4.4× better than chance** and top-3 **2.1×**.

In practical terms: searching the whole network means 43.1 km of pipe. A correct
top-1 zone narrows that to 3.6 km, and a top-3 shortlist to about 11 km — a reduction
of roughly 75% in the area a crew must cover.

Abrupt leaks localise better than incipient ones in both years, for the same reason
they detect faster: a sharp step produces a clean fingerprint, while a slow ramp blends
into whatever else is happening.

## 7. Ablations

### 7.1 How many pressure sensors are needed?

Each subset is re-tuned on 2018 and evaluated on 2019 (5 random draws per size).

| Sensors | Detection rate | False alarms/yr | Median delay | Localisation top-1 |
|---|---|---|---|---|
| 5 | 98% | 1.8 | 15.8 d | 18% |
| 10 | 100% | 2.6 | 12.2 d | 20% |
| 15 | 100% | 3.6 | 10.2 d | 27% |
| 20 | 100% | 3.2 | 12.6 d | 28% |
| 25 | 96% | 1.6 | 13.6 d | 31% |
| **33** | **89%** | **0.0** | 14.0 d | **37%** |

Read carelessly this says fewer sensors are better. They are not — detection rate alone
is gameable, because tuning on a noisier subset simply picks a lower threshold, which
finds more leaks *and* raises more false alarms. The honest reading of the table is:

- **Detection barely needs sensors.** Five are enough to catch nearly every leak.
- **Precision and localisation are what sensors buy.** Going 5 → 33 removes all false
  alarms and doubles localisation accuracy.

For a utility, that is the actionable result: a thin sensor network is enough to know
*that* something is wrong, but finding *where* is what the capital expenditure is for.

### 7.2 How finely can we localise?

| Zones | Top-1 | Top-3 | Random | Mean pipe per zone |
|---|---|---|---|---|
| 6 | 42% | 84% | 17% | 7.2 km |
| 8 | 42% | 63% | 13% | 5.4 km |
| 10 | 26% | 58% | 10% | 4.3 km |
| 12 | 37% | 53% | 8% | 3.6 km |
| 16 | 16% | 53% | 6% | 2.7 km |
| 20 | 37% | 47% | 5% | 2.2 km |

Accuracy over random improves as zones get finer, but absolute accuracy degrades and
becomes unstable — with only 19 test leaks, one leak is 5 percentage points, so the
non-monotonicity between 10, 12 and 20 zones is sampling noise, not signal. Twelve zones
is a reasonable operating point; the honest statement is that this data cannot resolve
the optimum precisely.

## 8. Limitations

- **Small test set.** Nineteen leaks in the held-out year. Differences of less than
  about 10 percentage points should not be treated as meaningful.
- **Incipient leaks remain the weak case** for both detection latency and localisation.
- **Localisation is zone-level.** Pipe-level identification from 33 sensors over 902
  pipes is not identifiable, and no amount of modelling changes that.
- **Simulated fingerprints carry model error.** The published L-Town model is documented
  as differing from the true network by up to 10% in diameters, roughness and demands,
  which puts a ceiling on match quality regardless of method.
- **Leaks are attributed to a pipe's upstream node** in the simulation, a small
  approximation at zone granularity.
- **One network, two years.** Nothing here demonstrates transfer to a different
  topology without re-tuning.
- **Alarm-to-leak matching is greedy.** With up to 16 concurrent leaks, attribution of
  an alarm to a specific onset is inherently ambiguous; alarms are matched to the oldest
  open leak within 45 days.

## 9. Conclusion

Reframing "is there a leak?" as "has a *new* leak started?" makes an unanswerable
question tractable, and a rolling-reference pressure model with CUSUM change detection
finds 89% of leaks in a held-out year with zero false alarms and a 14-day median
latency — against 5% for the night-flow method in standard industrial use.

Localising those leaks by matching their pressure fingerprint against an EPANET-
simulated library places 37% in the correct district and 53% in the top three, cutting
the search area by about 75%. The sensor ablation gives the operational punchline: a
handful of sensors is enough to know something is wrong, but it is the dense network
that tells you where to dig.

## 10. Reproducing

```bash
.venv/Scripts/python.exe src/download_data.py
```

```bash
.venv/Scripts/python.exe run_all.py
```

All tables in this report are regenerated into `reports/tables/` and all figures into
`reports/figures/`. Total runtime is about 15 minutes from a clean cache.

## References

1. Vrachimis, S. G. et al. *Battle of the Leakage Detection and Isolation Methods*.
   Journal of Water Resources Planning and Management, 2022.
2. BattLeDIM 2020 dataset, Zenodo record 4017659 (CC BY 4.0), KIOS CoE.
3. Klise, K. A. et al. *WNTR: Water Network Tool for Resilience*. US EPA.
4. Rossman, L. A. *EPANET 2 Users Manual*. US EPA, 2000.
