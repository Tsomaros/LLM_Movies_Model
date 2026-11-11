# Data summary

## Films per split

Counts from the database; all splits.

| split | films |
| --- | --- |
| train | 565901 |
| calibration | 52781 |
| validation | 95216 |
| test | 281690 |
| holdout | 57017 |
| later | 28744 |

## Open splits only

Everything below covers the open splits (train, calibration, validation) only. Locked films are never shown.

Films in the dataset (open splits, with at least one target): 34354

| target | films | share |
| --- | --- | --- |
| with revenue | 17312 | 50.4% |
| with score | 28907 | 84.1% |

## Coverage

| field | films | coverage |
| --- | --- | --- |
| writers | 32085 | 93.4% |
| producers | 29636 | 86.3% |
| composer | 24205 | 70.5% |
| distributor | 20826 | 60.6% |
| source_material (not original) | 7822 | 22.8% |
| release_type | 32302 | 94.0% |
| tagline | 21192 | 61.7% |
| competition (above 0) | 24265 | 70.6% |
| installment | 3526 | 10.3% |
| collaborations (above 0) | 11707 | 34.1% |
| director_age | 29145 | 84.8% |
| lead_age | 31441 | 91.5% |
| source_strength | 5739 | 16.7% |

## Score films by min_votes

Open-split films with a vote_average and at least this many votes: the table for the vote-threshold decision.
The threshold table comes from the database; `with score` comes from the parquet (at the current min_votes).

| min_votes | train | calibration | validation | total |
| --- | --- | --- | --- | --- |
| 10 | 68684 | 6329 | 10848 | 85861 |
| 20 | 41764 | 4256 | 7197 | 53217 |
| 50 | 22158 | 2511 | 4238 | 28907 |
| 100 | 14056 | 1652 | 2801 | 18509 |

Current setting: min_votes = 50

## Outcome-word audit (open splits only)

Films whose tagline or synopsis matches the pattern (word-bounded, case-insensitive), with up to 5 example film ids. Report only: nothing in the data is changed.

| pattern | films | example ids |
| --- | --- | --- |
| box office | 8 | 1284, 14376, 25631, 31151, 37757 |
| gross, grossed, grossing | 14 | 9012, 9336, 12228, 15394, 17190 |
| dollar figure | 223 | 161, 163, 380, 429, 1538 |
| Oscar | 56 | 1430, 2260, 8359, 9689, 10555 |
| Academy Award | 24 | 6643, 10806, 15012, 23383, 25126 |
| award-winning | 50 | 1282, 1818, 10942, 11004, 11466 |
| blockbuster | 12 | 10152, 11353, 22520, 49633, 76180 |
| Golden Globe | 4 | 35830, 61667, 414001, 478503 |
| record-breaking | 4 | 19016, 24960, 78192, 85910 |
| smash hit | 2 | 24664, 198391 |
