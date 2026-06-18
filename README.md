# GPS Spoofing Detection — Graduation Project

A machine-learning system that detects **GPS spoofing attacks** in human-movement GPS streams
(e.g. a wearable GPS bracelet). It learns from labelled GPS recordings, then flags each reading as
**normal** or **spoofed**.

The pipeline follows four stages: **Train → Evaluate → Predict → Score**.

---

## Results (latest run, 2026-06-18)

| Metric | Random-split test | Time-based test | **Blind test** (separate recording) |
|--------|------------------|-----------------|-------------------------------------|
| Accuracy | 99.42% | 99.53% | **99.58%** |
| Precision | 99.10% | 99.17% | 100% (spoofed) |
| Recall | 98.78% | 99.02% | 99% (spoofed) |
| F1-score | 98.94% | 99.09% | 99% (spoofed) |

The blind-test accuracy (≈ held-out accuracy) shows the model **generalizes** to a completely
separate recording it never trained on — not just memorizing.

Blind-test confusion matrix (7,423 rows):

```
              Predicted Normal   Predicted Spoofed
Normal             5341                 4
Spoofed              27              2051
```

---

## Folder structure

```
graduation-project-2/
├── Code/
│   ├── PROJECT_2.py            ← MAIN script: train → evaluate → predict → score
│   ├── GPS_Detector_Workflow.py  ← Tkinter GUI (Train / Predict / Evaluate tabs)
│   ├── GPS_Detector_Dashboard.py                  ← Light-mode GUI (imports the main engine)
│   └── GPS_Spoofing_Generator.py        ← Generates the spoofed datasets from clean recordings
├── DataSets/
│   ├── GPS_Data_All_Normal_40K.csv        ← raw, all normal (40,546 rows)
│   ├── GPS_Data_Mixed_40K.csv             ← + spoofing injected → TRAINING file
│   ├── GPS_Data_All_Normal_7K.csv         ← raw, all normal (7,423 rows)
│   ├── GPS_Data_Mixed_7K.csv              ← + spoofing injected → answer key for blind test
│   └── GPS_Data_Mixed_Nolabel_7K.csv      ← same rows, label removed → blind-test input
└── Documents/
    ├── Datasets Differences.md            ← dataset comparison (verified counts)
    ├── GPS Spoofing Model Explained.md    ← the 4 ML steps mapped to code line numbers
    ├── TRAIN_EVALUATE_PREDICT_SCORE.md    ← the 4 stages, plain-language + code
    └── PKL_AND_TWO_PREDICTION_PATHS.md    ← what .pkl files are + the two prediction paths
```

---

## Datasets

All labelled files share an 11-column schema:
`session_id, gps_date, gps_time, latitude, longitude, velocity, course, satellites_in_view, satellites_used, hdop, label`
(`label`: 0 = normal, 1 = spoofed). The `Nolabel` file drops `label` (10 columns).

| File | Rows | Normal | Spoofed | Role |
|------|------|--------|---------|------|
| `GPS_Data_All_Normal_40K.csv` | 40,546 | 40,546 | 0 | Raw source (not used by the model) |
| `GPS_Data_Mixed_40K.csv` | 40,546 | 29,408 | 11,138 | **Training** (`DATASET_FILE`) |
| `GPS_Data_All_Normal_7K.csv` | 7,423 | 7,423 | 0 | Raw source (not used by the model) |
| `GPS_Data_Mixed_7K.csv` | 7,423 | 5,345 | 2,078 | **Answer key** for scoring (`TRUE_LABEL_FILE`) |
| `GPS_Data_Mixed_Nolabel_7K.csv` | 7,423 | — | — | **Blind-test input** (`UNLABELED_FILE`) |

See `Documents/Datasets Differences.md` for full verification details.

---

## How to run

**Requirements:** Python 3.x with `scikit-learn`, `pandas`, `numpy`, `joblib`, `matplotlib`, `seaborn`.

```bash
pip install scikit-learn pandas numpy joblib matplotlib seaborn
```

**Run the full pipeline:**

```bash
cd Code
python PROJECT_2.py
```

This will:
1. **Train** an ensemble (Random Forest + Neural Network + Extra Trees) on `GPS_Data_Mixed_40K.csv`.
2. **Evaluate** it on a held-back 25% split (and a time-based split).
3. **Predict** blindly on `GPS_Data_Mixed_Nolabel_7K.csv` (no labels).
4. **Score** those predictions against `GPS_Data_Mixed_7K.csv`.

The script reads the CSVs from the sibling `DataSets/` folder automatically
(`DATA_DIR` is set near the top of the file — edit it there if your data lives elsewhere).

**GUI alternative:**

```bash
cd Code
python GPS_Detector_Workflow.py
```

---

## Outputs

Created under `Code/human_detection_outputs/`:

- `models/human_ensemble_model.pkl` — the trained model
- `models/human_preprocessing.pkl` — the imputer / scaler / normalizer used at training time
- `human_gps_predictions_*.csv` — predictions over the full training set
- `unlabeled_predictions_*.csv` — blind predictions on the no-label file
- `human_model_report_*.txt` — text summary of performance
- `plots/` — confusion matrices, model-performance bar chart, feature-importance chart

---

## How it works (high level)

The model doesn't look at raw coordinates alone. From each GPS stream it engineers ~30 features
that capture **physical consistency**, such as:

- **Speed residual** — does the reported `velocity` match the distance actually covered between points?
- **Satellite behavior** — `satellites_used` vs `satellites_in_view`, and sudden changes.
- **Course vs bearing** — does the reported heading match the direction implied by the coordinates?
- **HDOP / acceleration / rolling windows** — short-term context that exposes attack windows.

Spoofing tends to break these physical relationships, which is what the ensemble learns to detect.
See `Documents/` for the full, plain-language explanation tied to exact code line numbers.
