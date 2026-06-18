# Datasets Differences

> **Verified against the actual files in `Downloads\graduation-project-2\DataSets` on 2026-06-18.**
> All five filenames correctly match their content (column count, row count, and label distribution all check out).

All labelled files share the same **11-column schema**:

```
session_id, gps_date, gps_time, latitude, longitude, velocity,
course, satellites_in_view, satellites_used, hdop, label
```

> The `Nolabel` file drops the `label` column (10 columns).

---

## Overview Table

| File | Rows (data) | Labels | Role |
|------|-------------|--------|------|
| `GPS_Data_All_Normal_40K.csv` | 40,546 | all 0 (normal) | Original raw GPS recording — no spoofing injected yet. `course` has some missing values (3,511 empty rows). |
| `GPS_Data_Mixed_40K.csv` | 40,546 | 29,408 normal / 11,138 spoofed (~27.5%) | Same data as above but with spoofing attacks injected and `course` fully filled in (0 missing). ← **This is the training file** (`DATASET_FILE`) |
| `GPS_Data_All_Normal_7K.csv` | 7,423 | all 0 (normal) | A second, smaller raw recording — no spoofing. `course` has a few missing values (106 empty rows). |
| `GPS_Data_Mixed_7K.csv` | 7,423 | 5,345 normal / 2,078 spoofed (~28%) | The smaller recording with spoofing injected and `course` fully filled in (0 missing). ← **The true-label file** (`TRUE_LABEL_FILE`) for scoring the blind test |
| `GPS_Data_Mixed_Nolabel_7K.csv` | 7,423 | none (10 columns, no `label`) | Same rows as `GPS_Data_Mixed_7K` but with the `label` column removed. ← **The unlabeled file** (`UNLABELED_FILE`) the model predicts on blindly |

---

## How They Pair Up

There are basically **two recording sessions**, each in multiple versions:

### Big dataset (40,546 rows)
- `GPS_Data_All_Normal_40K.csv` = raw, clean (all normal)
- `GPS_Data_Mixed_40K.csv` = + attacks injected → **used for training**

### Small dataset (7,423 rows)
- `GPS_Data_All_Normal_7K.csv` = raw, clean (all normal)
- `GPS_Data_Mixed_7K.csv` = + attacks injected → **answer key** for the blind test
- `GPS_Data_Mixed_Nolabel_7K.csv` = same rows but answer hidden → **what the model is tested on**

---

## What the Script Actually Uses (3 of the 5)

```
DATASET_FILE     = GPS_Data_Mixed_40K.csv            ← train + evaluate
UNLABELED_FILE   = GPS_Data_Mixed_Nolabel_7K.csv     ← blind predict
TRUE_LABEL_FILE  = GPS_Data_Mixed_7K.csv             ← score the blind predictions
```

The two raw "clean" files (`GPS_Data_All_Normal_40K.csv` and `GPS_Data_All_Normal_7K.csv`) are
**not used** by the script — they're the pre-spoofing source material.

The design is sound: it trains on the big 40,546-row set and tests honestly on a completely
separate 7,423-row set the model never saw.

> ✅ **Note:** the script `Code/PROJECT_2.py` (lines 54–56) is already wired to these new
> filenames and reads them from the sibling `DataSets/` folder via `DATA_DIR`. No edits needed
> to run it.

---

## Verification Details

| File | Columns | Data rows | Label = 0 | Label = 1 | `course` missing |
|------|---------|-----------|-----------|-----------|------------------|
| `GPS_Data_All_Normal_40K.csv` | 11 | 40,546 | 40,546 | 0 | 3,511 |
| `GPS_Data_Mixed_40K.csv` | 11 | 40,546 | 29,408 | 11,138 | 0 |
| `GPS_Data_All_Normal_7K.csv` | 11 | 7,423 | 7,423 | 0 | 106 |
| `GPS_Data_Mixed_7K.csv` | 11 | 7,423 | 5,345 | 2,078 | 0 |
| `GPS_Data_Mixed_Nolabel_7K.csv` | 10 | 7,423 | — (no label) | — | 0 |
