# EchoJudge evaluation report (split: dev)

Labels are instructed flaws over each recording's speech extent (not listener-verified). Time-weighted metrics compare detected spans with labelled spans; file-level metrics ask whether a category fired at all.

## Mode: none (10 participant recordings)

| Category | Seg P | Seg R | Seg F1 | File P | File R | File F1 | File TP/FP/FN/TN |
|---|---|---|---|---|---|---|---|
| rushed_pacing | 0.68 | 0.37 | 0.48 | 0.67 | 1.00 | 0.80 | 2/1/0/7 |
| slow_pacing | 1.00 | 0.02 | 0.04 | 1.00 | 0.50 | 0.67 | 1/0/1/8 |
| excessive_pauses | 0.97 | 0.50 | 0.66 | 0.67 | 1.00 | 0.80 | 2/1/0/7 |
| monotone_delivery | n/a | 0.00 | 0.00 | n/a | 0.00 | 0.00 | 0/0/2/8 |

Clean controls: 0/2 files had at least one scored flag; flags per minute {}; mean score 100.0; cross-speaker clean files with a flag: 0/0.

Score ranking: flawed recording scored below its clean control in 7/8 pairs; mean flawed score 94.81.
Repeatability: max |score diff| on a repeated run = 0.0.
