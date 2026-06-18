# GPS Spoofing Model — Train, Evaluate, Predict, Score

> A plain-language guide to the four core machine-learning steps in
> `PROJECT_2.py`, with the exact line numbers for each step.

**Think of the model like a student learning to spot fake GPS readings.**

---

## 1. Train 🎓 (teach the model)

**Training** = showing the model lots of examples **with the answers included**, so it learns the patterns.

In your project, the script feeds it `GPS_Data_Mixed_40K.csv` — 40,546 rows where each row is already marked `0` (normal) or `1` (spoofed). The model studies these and learns rules like:

> *"When the reported speed doesn't match the distance between points, and satellites behave oddly → it's probably spoofed."*

📍 **In the code:** `model.fit(X_train, y_train)` — `X` is the GPS features, `y` is the known answers.

**Exact lines:**
| What | Line(s) |
|------|---------|
| The 3 models are defined (RandomForest, MLP, ExtraTrees) | `158–188` |
| Build feature matrix `X` and answers `y` | `419–420` |
| Train each individual model | `446` (`model.fit(X_train, y_train)`) |
| Build the soft-voting ensemble | `462–467` |
| Train the ensemble | `468` (`self.ensemble_model.fit(X_train, y_train)`) |

---

## 2. Evaluate 📊 (test the student on a quiz it can see the answers to)

**Evaluation** = checking how well the model learned, **using data where you also know the answers** — so you can compare its guesses to the truth.

The trick: you **hide part of the training data** from the model during learning (the script holds back 25%), then test on that hidden part. The model never "saw" those rows while training, so it's a fair quiz.

This tells you things like *"95% accurate."*

📍 **In the code:** the 75/25 `train_test_split`, then comparing predictions to the true labels.

**Exact lines:**
| What | Line(s) |
|------|---------|
| 75/25 split (hold back 25% for the quiz) | `423–428` (`train_test_split`, `TEST_SIZE = 0.25`) |
| Predict on the held-back test set | `448` (`test_pred = model.predict(X_test)`) |
| Accuracy of each model | `450–451` |
| Ensemble predictions on test set | `471` |
| Precision / Recall / F1 | `474–476` |
| Accuracy stored | `479–480` |
| Confusion matrix | `493` |
| Classification report | `497` |
| Extra: time-based split (more realistic) | `507–562` |

---

## 3. Predict 🔮 (use the trained model on new, unknown data)

**Prediction** = the model's actual job — looking at GPS data and **guessing** normal vs spoofed.

The difference from evaluation: here you may **not know the real answer**. This is what happens in the real world / deployment.

In your project, this is the **blind test** on `GPS_Data_Mixed_Nolabel_7K.csv` — the file with the `label` column removed. The model has *no answer key*; it just outputs its guesses:

```
prediction = spoofed,  confidence = 92%
```

📍 **In the code:** `model.predict(X)`.

**Exact lines:**
| What | Line(s) |
|------|---------|
| Blind-prediction function | `841–873` (`run_blind_prediction`) |
| Load the saved trained model | `846–847` |
| Read the unlabeled file (no answers) | `853` (`pd.read_csv(UNLABELED_FILE)`) |
| Make the guesses | `859` (`pred = model.predict(X_norm)`) |
| Add confidence % | `860, 864` |
| Save the blind predictions to CSV | `866–868` |

> Note: the script *also* predicts on the full training dataset to save an output file — that happens in `predict_all_and_save` at lines `663–667`.

---

## 4. Score the predictions ✅ (grade the guesses against the real answers)

**Scoring** = taking the model's blind predictions and **comparing them to the true answers** to see how many it got right.

In your project, after the model guesses blindly on the no-label file (7,423 rows), the script opens `GPS_Data_Mixed_7K.csv` (the **answer key** — same 7,423 rows but *with* labels, of which **2,078 are actually spoofed**) and compares:

> *"Of the rows the model called 'spoofed', how many of the 2,078 truly-spoofed rows did it catch?"* → e.g. **94% accuracy** (illustrative — the real number comes from running the script).

📍 **In the code:** `score_blind_predictions()` → `accuracy_score(y_true, y_pred)`.

**Exact lines:**
| What | Line(s) |
|------|---------|
| Scoring function | `877–897` (`score_blind_predictions`) |
| Open the answer-key file | `879` (`pd.read_csv(TRUE_LABEL_FILE)`) |
| True answers vs model guesses | `888–889` |
| Final accuracy | `891` (`accuracy_score(y_true, y_pred)`) |
| Confusion matrix + report | `894, 896` |

---

## How They All Fit Together

```
TRAIN      →  learn from GPS_Data_Mixed_40K.csv (with answers)
EVALUATE   →  quiz on held-back 25% (answers known)  → "how well did it learn?"
PREDICT    →  guess on GPS_Data_Mixed_Nolabel_7K.csv (answers hidden)
SCORE      →  grade those guesses vs GPS_Data_Mixed_7K.csv (real answers) → honest final %
```

### The key distinction people get confused by:

| Term | Does it know the answer? | Purpose |
|------|--------------------------|---------|
| **Train** | Yes — *learns* from answers | Teach the model |
| **Evaluate** | Yes — checks against answers | Measure learning quality |
| **Predict** | No — just guesses | The real job |
| **Score** | Yes — compares afterward | Grade the guesses |

**Evaluate vs Score** are both "grading," but:
- **Evaluate** uses a slice of the *same* dataset (quick self-check).
- **Score** uses a *completely separate* dataset the model never touched — a much more honest, real-world test. That's why your project does both. 👍

---

## Where It All Runs From

The whole sequence is kicked off at the bottom of the script:

| Step | Line(s) |
|------|---------|
| `run()` orchestrates train + evaluate + predict | `732–743` |
| `main()` starts training | `747–750` |
| `__main__` block: train → blind predict → score | `901–915` |
