# Train → Evaluate → Predict → Score

*A plain-language guide to how `PROJECT_2.py` works, with the exact place in the code for every step.*

Think of the model like a student learning to spot fake GPS readings.

---

## The 4 stages at a glance

```
TRAIN      →  learn from GPS_Data_Mixed_40K.csv (with answers)
EVALUATE   →  quiz on held-back 25% (answers known)        → "how well did it learn?"
PREDICT    →  guess on GPS_Data_Mixed_Nolabel_7K.csv (answers hidden)
SCORE      →  grade those guesses vs GPS_Data_Mixed_7K.csv (real answers) → honest final %
```

| Term     | Does it know the answer?     | Purpose                  |
|----------|------------------------------|--------------------------|
| Train    | Yes — learns from answers    | Teach the model          |
| Evaluate | Yes — checks against answers | Measure learning quality |
| Predict  | No — just guesses            | The real job             |
| Score    | Yes — compares afterward     | Grade the guesses        |

**Evaluate vs Score** are both "grading," but:
- **Evaluate** uses a slice of the *same* dataset (quick self-check).
- **Score** uses a *completely separate* dataset the model never touched — a more honest, real-world test.

The three data files are defined here:

```python
# PROJECT_2.py : lines 54–56
DATASET_FILE   = os.path.join(BASE, "GPS_Data_Mixed_40K.csv")          # training data (has labels)
UNLABELED_FILE = os.path.join(BASE, "GPS_Data_Mixed_Nolabel_7K.csv") # blind test (no answer key)
TRUE_LABEL_FILE= os.path.join(BASE, "GPS_Data_Mixed_7K.csv")         # the answer key for scoring
```

---

## 1. Train 🎓 (teach the model)

**What it is:** showing the model lots of examples *with the answers included*, so it learns the patterns —
e.g. *"when reported speed doesn't match the distance between points, and satellites behave oddly → probably spoofed."*

**Where in the code:**

The labelled data is loaded and the known answers (`label`) are turned into numbers `0`/`1`:

```python
# load_data()  : lines 212, 225–234
df = pd.read_csv(csv_file)
...
df["label_numeric"] = df["label"].astype(int)     # 0 = normal, 1 = spoofed
```

The features (X) and the known answers (y) are pulled out:

```python
# train_and_evaluate()  : lines 419–420
X = df[self.feature_names].copy()   # the GPS features
y = df["label_numeric"].copy()      # the known answers
```

The actual learning happens here — each model studies X with its answers y:

```python
# train_and_evaluate()  : line 446  (inside the model loop, lines 444–459)
model.fit(X_train, y_train)         # <-- this is "training"
```

And the three models are combined into one voting "committee" that also trains:

```python
# train_and_evaluate()  : lines 462–468
self.ensemble_model = VotingClassifier(...)
self.ensemble_model.fit(X_train, y_train)   # <-- ensemble training
```

> The three students on the committee are defined in `HumanGPSDetector.__init__`,
> lines **159–188**: a Random Forest, a Neural Network (MLP), and Extra Trees.

---

## 2. Evaluate 📊 (quiz it on answers you can see)

**What it is:** checking how well the model learned, using data where you *also* know the answers.
The trick: hide part of the training data (25%) during learning, then test on that hidden part —
the model never "saw" those rows, so it's a fair quiz.

**Where in the code:**

The 75 / 25 hold-back split. `TEST_SIZE = 0.25` is set at line 63:

```python
# train_and_evaluate()  : lines 423–433
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,        # hold back 25% as the "quiz"
    random_state=RANDOM_STATE,
    stratify=y,                 # keep the same normal/spoofed ratio in both halves
)
X_train_raw = X.iloc[train_idx]
X_test_raw  = X.iloc[test_idx]   # <-- the held-back rows the model won't learn from
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]
```

After training, the model guesses on the held-back quiz and we compare to the truth:

```python
# train_and_evaluate()  : lines 447–451
train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)
train_acc = accuracy_score(y_train, train_pred) * 100
test_acc  = accuracy_score(y_test,  test_pred)  * 100   # <-- "how well did it learn?"
```

The ensemble's evaluation (accuracy, precision, recall, F1) is computed here:

```python
# train_and_evaluate()  : lines 470–491
test_ens = self.ensemble_model.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_ens, ...)
self.performance = {
    "test_accuracy": accuracy_score(y_test, test_ens),
    "precision": precision, "recall": recall, "f1_score": f1,
}
```

> **Bonus second quiz:** `evaluate_time_based_split()` (lines **507–562**) repeats the
> evaluation but splits by *time order* instead of randomly — a more realistic test for a
> GPS stream. Called at line **503**.

---

## 3. Predict 🔮 (use the model on new, unknown data)

**What it is:** the model's actual job — looking at GPS data and guessing normal vs spoofed,
on a file where you may **not** know the real answer. This is "deployment."

**Where in the code:**

The blind test loads the *no-label* file and the saved model, rebuilds the same features, and guesses:

```python
# run_blind_prediction()  : lines 846–864
model = joblib.load(.../"human_ensemble_model.pkl")   # reuse the trained model
...
udf = pd.read_csv(UNLABELED_FILE)                     # the file with NO answer key
udf = build_features_unlabeled(udf)                   # same features as training (lines 770–837)
...
pred = model.predict(X_norm)                          # <-- pure guessing, no answers
prob = model.predict_proba(X_norm)                    # confidence for each guess
udf["prediction"]  = np.where(pred == 1, "spoofed", "normal")
udf["confidence"]  = [prob[i, p] * 100 for i, p in enumerate(pred)]
```

> Example output row: `prediction = spoofed, confidence = 92%`.

> There is also a *non-blind* prediction over the full training file in
> `predict_all_and_save()` (lines **663–730**) — that one is mainly for saving an output CSV
> and the model `.pkl` files (lines 699–709).

---

## 4. Score the predictions ✅ (grade the blind guesses)

**What it is:** taking the model's blind predictions and comparing them to the *true* answers
to see how many it got right — the honest, final number.

**Where in the code:**

After the blind guesses, this opens the answer-key file and compares:

```python
# score_blind_predictions()  : lines 877–897
true_df = pd.read_csv(TRUE_LABEL_FILE)                 # the answer key (same rows, WITH labels)
true_df["label_numeric"] = true_df["label"].astype(int)
...
y_true = true_df["label_numeric"]                      # real answers
y_pred = udf["prediction_numeric"].astype(int)         # the model's blind guesses

acc = accuracy_score(y_true, y_pred)                   # <-- the honest final %
print(f"Blind-test Accuracy: {acc * 100:.2f}%")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["Normal", "Spoofed"]))
```

---

## How it all runs together

The bottom of the file wires the four stages in order:

```python
# __main__  : lines 901–915
detector = main()                       # 1) TRAIN + 2) EVALUATE  (main() -> detector.run())

if os.path.exists(UNLABELED_FILE):
    udf, blind_path = run_blind_prediction()   # 3) PREDICT (blind)
    if os.path.exists(TRUE_LABEL_FILE):
        score_blind_predictions(udf)           # 4) SCORE
```

And `detector.run()` itself (lines **732–743**) chains the pipeline:

```python
# run()  : lines 733–738
df = self.load_data(csv_file)      # read + label the training data   (TRAIN data prep)
df = self.create_features(df)      # build the GPS features
self.analyze_data_quality(df)      # sanity checks / class separation
self.plot_stream_behavior(df)      # report plots
self.train_and_evaluate(df)        # TRAIN + EVALUATE
self.predict_all_and_save(df)      # predict on full set + save model
```

---

### Quick reference: line numbers

| Stage    | Function                      | Key lines |
|----------|-------------------------------|-----------|
| Train    | `train_and_evaluate`          | 419–420, **446**, 462–468 |
| Evaluate | `train_and_evaluate`          | 423–433, **447–451**, 470–491 |
| Evaluate | `evaluate_time_based_split`   | 503, 507–562 |
| Predict  | `run_blind_prediction`        | 846–864 |
| Score    | `score_blind_predictions`     | 877–897 |
| Orchestration | `run` / `__main__`       | 732–743, 901–915 |
