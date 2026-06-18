# What is a `.pkl` file? And why are there two prediction paths?

*A plain-language explainer for `PROJECT_2.py`.*

---

## What is a `.pkl` file?

`.pkl` = **"pickle"** file. Pickling is Python's way of **saving a live object to disk** so you
can load it back later exactly as it was.

Here's the idea: after your model spends time *learning* (training), all that learned knowledge
lives in memory (RAM). The moment the script ends, that memory is wiped — the model is gone.
A `.pkl` file is a **freeze-frame** of the trained model saved to your hard drive, so you don't
have to retrain from scratch every time.

> **Analogy:** training is like a student studying for months. The `.pkl` file is taking a
> **perfect photograph of their brain** so that next time you just load the photo instead of
> making them study all over again.

### In the code

It **saves** two `.pkl` files (lines 699–709):

```python
joblib.dump(self.ensemble_model, model_path)   # freezes the trained model
joblib.dump({...preprocessing...}, prep_path)  # freezes the data-prep settings
```

1. `human_ensemble_model.pkl` — the trained model (the "brain").
2. `human_preprocessing.pkl` — the imputer / scaler / normalizer (the settings used to clean the
   data before feeding it to the model). These **must** match what training used, or predictions
   will be garbage.

Then later, the blind test **loads them back** instead of retraining (lines 846–851):

```python
model = joblib.load(".../human_ensemble_model.pkl")
preprocessing = joblib.load(".../human_preprocessing.pkl")
```

---

## The two prediction paths

Both call `model.predict(...)`, but they predict on **different files** for **different reasons**.

| | **Path A: `predict_all_and_save`** (lines 663–730) | **Path B: `run_blind_prediction`** (lines 841–873) |
|---|---|---|
| **Predicts on** | The **full training file** (`GPS_Data_Mixed_40K.csv`) — data the model already learned from | A **separate file** (`GPS_Data_Mixed_Nolabel_7K.csv`) the model has **never seen** |
| **Does the model know the answers?** | Yes — these rows had labels during training | No — this file has the label column removed |
| **Main purpose** | **Housekeeping**: save the output CSV and **save the `.pkl` model files** | **The honest test**: simulate real-world deployment |
| **Is the score trustworthy?** | ❌ No — it's grading the student on the exact questions they studied. Looks great but means little | ✅ Yes — brand-new questions, so the score is honest |

### Why have both?

- **Path A exists mostly as a side effect.** Its real job is to **save the trained model to those
  `.pkl` files** (and write a predictions CSV for the report). The "predictions" it makes are on
  data the model already memorized, so they're not a fair measure of real performance.
  **Don't trust that number as your accuracy.**

- **Path B is the one that matters for your results.** It loads the saved `.pkl`, runs on a fresh
  file with no answers, then `score_blind_predictions` grades it against the hidden answer key.
  *That's* your honest accuracy.

### The flow

```
Path A trains & SAVES the .pkl  ───►  Path B LOADS the .pkl and does the real test
```

- Path A = the "save the brain photo" step.
- Path B = the "load the brain photo and give it a fair exam" step.
