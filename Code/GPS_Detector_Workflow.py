# -*- coding: utf-8 -*-
"""
GPS Spoofing Detector GUI - Train / Predict / Evaluate Edition

ضع هذا الملف في نفس المجلد مع:
    PROJECT_2.py

تشغيل:
    python GPS_Detector_Workflow.py

الفكرة:
1) Train Model: اختر داتا فيها label ودرب الموديل واحفظ pkl.
2) Predict Unlabeled: اختر pkl + داتا بدون label، واعمل prediction واحفظ CSV.
3) Evaluate: اختياري، قارن ملف predictions مع ملف true labels إذا موجود.

ملاحظة: نتائج التدريب تظهر داخل صفحة Train Model، ونتائج التنبؤ تظهر داخل صفحة Predict Unlabeled.
"""
#  GUI 2 (recommended) — Train / Predict / Evaluate workflow
import os
import sys
import io
import contextlib
import threading
import warnings
from datetime import datetime

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)

try:
    from PROJECT_2 import (
        HumanGPSDetector,
        haversine_m,
        bearing_deg,
        circular_diff_deg,
        add_time_delta_preserve_order,
    )
except ImportError as e:
    print("ERROR: Cannot import from PROJECT_2.py")
    print("Put GPS_Detector_Workflow.py and PROJECT_2.py in the same folder.")
    print(e)
    sys.exit(1)

warnings.filterwarnings("ignore")

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

COLORS = {
    "bg": "#ffffff",
    "surface": "#ffffff",
    "card": "#f8fafc",
    "border": "#e2e8f0",
    "accent": "#2563eb",
    "green": "#16a34a",
    "yellow": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "text": "#0f172a",
    "muted": "#475569",
    "dim": "#94a3b8",
    "tbl_odd": "#ffffff",
    "tbl_even": "#f1f5f9",
    "tbl_fg": "#0f172a",
    "tbl_head": "#f1f5f9",
}

NAV_W = 240
ROLLING_WINDOW = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "human_detection_outputs")
AUTO_MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
AUTO_PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
LATEST_MODEL_PATH = os.path.join(AUTO_MODEL_DIR, "latest_gps_detector_model.pkl")


TRAIN_REQUIRED_COLS = [
    "session_id", "gps_date", "gps_time", "latitude", "longitude",
    "velocity", "course", "satellites_in_view", "satellites_used", "hdop", "label",
]

UNLABELED_REQUIRED_COLS = [
    "session_id", "gps_date", "gps_time", "latitude", "longitude",
    "velocity", "course", "satellites_in_view", "satellites_used", "hdop",
]

LABEL_MAP = {
    "normal": 0,
    "legitimate": 0,
    "real": 0,
    "0": 0,
    "fake": 1,
    "spoofed": 1,
    "attack": 1,
    "1": 1,
}


# ---------------------------------------------------------------------
# Console redirect
# ---------------------------------------------------------------------
class Redirector:
    def __init__(self, *boxes):
        self._boxes = boxes
        self._lock = threading.Lock()

    def write(self, s: str):
        with self._lock:
            for box in self._boxes:
                try:
                    box.insert("end", s)
                    box.see("end")
                    box.update_idletasks()
                except Exception:
                    pass

    def flush(self):
        pass


class TeeCapture:
    """Write to the GUI console and keep a full copy for the page summary."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str):
        for stream in self.streams:
            try:
                stream.write(text)
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


# ---------------------------------------------------------------------
# Small reusable widgets
# ---------------------------------------------------------------------
class StatCard(ctk.CTkFrame):
    def __init__(self, master, title, color, **kw):
        super().__init__(master, fg_color=COLORS["card"], corner_radius=12,
                         border_width=1, border_color=COLORS["border"], **kw)
        ctk.CTkFrame(self, fg_color=color, height=3, corner_radius=3).pack(fill="x")
        ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=14, pady=(10, 0))
        self._value = ctk.CTkLabel(
            self, text="—", font=ctk.CTkFont(size=28, weight="bold"),
            text_color=COLORS["text"],
        )
        self._value.pack(anchor="w", padx=14, pady=(2, 12))

    def set(self, value):
        self._value.configure(text=str(value))


# ---------------------------------------------------------------------
# Validation / feature engineering helpers
# ---------------------------------------------------------------------
def normalize_label_series(s: pd.Series) -> pd.Series:
    """Convert labels to 0/1."""
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce")
        return out.astype("Int64")
    return s.astype(str).str.strip().str.lower().map(LABEL_MAP).astype("Int64")


def validate_columns(df: pd.DataFrame, required_cols, file_kind: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{file_kind} missing required columns: {missing}")


def validate_gps_values(df: pd.DataFrame):
    """Return warnings for suspicious values. Does not stop prediction."""
    warnings_list = []
    check_df = df.copy()

    for col in ["latitude", "longitude", "velocity", "course", "satellites_in_view", "satellites_used", "hdop"]:
        if col in check_df.columns:
            check_df[col] = pd.to_numeric(check_df[col], errors="coerce")

    if check_df["latitude"].isna().any() or check_df["longitude"].isna().any():
        warnings_list.append("Some latitude/longitude values are non-numeric or missing.")

    bad_lat = (~check_df["latitude"].between(-90, 90)).sum()
    bad_lon = (~check_df["longitude"].between(-180, 180)).sum()
    if bad_lat:
        warnings_list.append(f"Latitude out of range rows: {int(bad_lat)}")
    if bad_lon:
        warnings_list.append(f"Longitude out of range rows: {int(bad_lon)}")

    if (check_df["velocity"] < 0).sum() > 0:
        warnings_list.append("There are negative velocity values.")
    if (check_df["hdop"] < 0).sum() > 0:
        warnings_list.append("There are negative hdop values.")
    if (check_df["satellites_used"] > check_df["satellites_in_view"]).sum() > 0:
        warnings_list.append("Some rows have satellites_used > satellites_in_view.")

    return warnings_list


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare labeled dataframe when we do not call HumanGPSDetector.load_data()."""
    df = df.copy().reset_index(drop=True)

    if "Data Type" in df.columns and "label" not in df.columns:
        df["label"] = df["Data Type"]

    validate_columns(df, TRAIN_REQUIRED_COLS, "Training dataset")

    df["label_numeric"] = normalize_label_series(df["label"])
    if df["label_numeric"].isna().any():
        bad = int(df["label_numeric"].isna().sum())
        raise ValueError(f"Could not convert {bad} label values to 0/1.")

    df["label_numeric"] = df["label_numeric"].astype(int)
    normal_count = int((df["label_numeric"] == 0).sum())
    spoof_count = int((df["label_numeric"] == 1).sum())
    if normal_count == 0 or spoof_count == 0:
        raise ValueError("Training needs both classes: normal=0 and spoofed=1.")

    df["label_text"] = df["label_numeric"].map({0: "normal", 1: "spoofed"})
    df["timestamp"] = pd.NaT
    df = add_time_delta_preserve_order(df)
    return df


def build_features_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as training, but without label."""
    df = df.copy().reset_index(drop=True)
    validate_columns(df, UNLABELED_REQUIRED_COLS, "Unlabeled dataset")

    numeric_cols = [
        "latitude", "longitude", "velocity", "course",
        "satellites_in_view", "satellites_used", "hdop",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.NaT
    df = add_time_delta_preserve_order(df)

    df["sat_count"] = df["satellites_in_view"]
    df["sat_locks"] = df["satellites_used"]
    df["sat_ratio"] = df["sat_locks"] / (df["sat_count"] + 1e-6)
    df["sat_discrepancy"] = (df["sat_count"] - df["sat_locks"]).abs()

    median_dt = df["time_delta"].median()
    if pd.isna(median_dt) or median_dt <= 0:
        median_dt = 1.0
    df["time_delta"] = df["time_delta"].fillna(median_dt).clip(lower=0.5, upper=5.0)

    df["prev_lat"] = df.groupby("session_id")["latitude"].shift(1)
    df["prev_lon"] = df.groupby("session_id")["longitude"].shift(1)
    first_rows = df["prev_lat"].isna() | df["prev_lon"].isna()
    df.loc[first_rows, "prev_lat"] = df.loc[first_rows, "latitude"]
    df.loc[first_rows, "prev_lon"] = df.loc[first_rows, "longitude"]

    df["distance_m"] = haversine_m(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])
    df["coord_speed"] = df["distance_m"] / df["time_delta"]
    df["speed_residual"] = (df["velocity"] - df["coord_speed"]).abs()
    df["velocity_diff"] = df.groupby("session_id")["velocity"].diff().abs().fillna(0)
    df["acceleration"] = df["velocity_diff"] / df["time_delta"]

    df["bearing_from_coords"] = bearing_deg(df["prev_lat"], df["prev_lon"], df["latitude"], df["longitude"])
    df.loc[df["distance_m"] < 0.7, "bearing_from_coords"] = np.nan

    df["course_filled"] = df["course"]
    df["course_filled"] = df["course_filled"].fillna(df["bearing_from_coords"])
    df["course_filled"] = df.groupby("session_id")["course_filled"].ffill().bfill().fillna(0)

    df["prev_course"] = df.groupby("session_id")["course_filled"].shift(1).fillna(df["course_filled"])
    df["course_change"] = circular_diff_deg(df["course_filled"], df["prev_course"])

    df["course_bearing_diff"] = circular_diff_deg(
        df["course_filled"].fillna(0),
        df["bearing_from_coords"].fillna(df["course_filled"]),
    )
    df.loc[df["distance_m"] < 0.7, "course_bearing_diff"] = 0

    df["hdop_diff"] = df.groupby("session_id")["hdop"].diff().abs().fillna(0)
    df["is_stationary"] = (df["velocity"] < 0.25).astype(int)
    df["is_fast_human"] = (df["velocity"] > 2.8).astype(int)

    rolling_base_cols = [
        "velocity", "coord_speed", "speed_residual", "sat_ratio",
        "sat_discrepancy", "hdop", "course_change", "course_bearing_diff",
    ]
    for col in rolling_base_cols:
        roll = df.groupby("session_id")[col].rolling(ROLLING_WINDOW, min_periods=1)
        df[f"{col}_mean_{ROLLING_WINDOW}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_std_{ROLLING_WINDOW}"] = roll.std().reset_index(level=0, drop=True).fillna(0)

    return df


def detect_attack_types(feature_df: pd.DataFrame, predictions: np.ndarray):
    """Simple rule-based explanation for predicted spoofed rows."""
    attack_types = {
        "gradual_drag": {"count": 0, "indicators": []},
        "freeze": {"count": 0, "indicators": []},
        "replay": {"count": 0, "indicators": []},
        "fake_walking": {"count": 0, "indicators": []},
        "geofence_evasion": {"count": 0, "indicators": []},
        "signal_manipulation": {"count": 0, "indicators": []},
    }

    spoof_idx = np.where(np.asarray(predictions).astype(int) == 1)[0]
    for idx in spoof_idx:
        if idx >= len(feature_df):
            continue
        row = feature_df.iloc[idx]
        speed_residual = row.get("speed_residual", 0)
        coord_speed = row.get("coord_speed", 0)
        velocity = row.get("velocity", 0)
        distance_m = row.get("distance_m", 0)
        course_bearing_diff = row.get("course_bearing_diff", 0)
        velocity_diff = row.get("velocity_diff", 0)
        hdop_diff = row.get("hdop_diff", 0)
        sat_discrepancy = row.get("sat_discrepancy", 0)
        sat_ratio = row.get("sat_ratio", 1)
        hdop = row.get("hdop", 0)

        if pd.isna(speed_residual): speed_residual = 0
        if pd.isna(coord_speed): coord_speed = 0
        if pd.isna(velocity): velocity = 0
        if pd.isna(distance_m): distance_m = 0
        if pd.isna(course_bearing_diff): course_bearing_diff = 0
        if pd.isna(velocity_diff): velocity_diff = 0
        if pd.isna(hdop_diff): hdop_diff = 0
        if pd.isna(sat_discrepancy): sat_discrepancy = 0
        if pd.isna(sat_ratio): sat_ratio = 1
        if pd.isna(hdop): hdop = 0

        if speed_residual > 0.5 and coord_speed > 0.3:
            attack_types["gradual_drag"]["count"] += 1
            if "speed_residual" not in attack_types["gradual_drag"]["indicators"]:
                attack_types["gradual_drag"]["indicators"].append("speed_residual")
        if velocity < 0.15 and distance_m < 0.5:
            attack_types["freeze"]["count"] += 1
            if "low_velocity" not in attack_types["freeze"]["indicators"]:
                attack_types["freeze"]["indicators"].append("low_velocity")
        if course_bearing_diff > 20 and velocity_diff < 0.2:
            attack_types["replay"]["count"] += 1
            if "course_bearing_diff" not in attack_types["replay"]["indicators"]:
                attack_types["replay"]["indicators"].append("course_bearing_diff")
        if velocity > 0.5 and coord_speed < 0.3:
            attack_types["fake_walking"]["count"] += 1
            if "velocity_vs_coord" not in attack_types["fake_walking"]["indicators"]:
                attack_types["fake_walking"]["indicators"].append("velocity_vs_coord")
        if hdop_diff > 0.5 and sat_discrepancy > 2:
            attack_types["geofence_evasion"]["count"] += 1
            if "hdop_sat_diff" not in attack_types["geofence_evasion"]["indicators"]:
                attack_types["geofence_evasion"]["indicators"].append("hdop_sat_diff")
        if sat_ratio < 0.5 and hdop > 2:
            attack_types["signal_manipulation"]["count"] += 1
            if "sat_ratio_hdop" not in attack_types["signal_manipulation"]["indicators"]:
                attack_types["signal_manipulation"]["indicators"].append("sat_ratio_hdop")

    total = sum(v["count"] for v in attack_types.values())
    for key in attack_types:
        attack_types[key]["percentage"] = (attack_types[key]["count"] / total * 100) if total else 0
    return attack_types


def make_attack_report(attack_counts):
    lines = []
    lines.append("ATTACK TYPE DETECTION REPORT")
    lines.append("=" * 60)
    total = sum(v["count"] for v in attack_counts.values())
    lines.append(f"Total explained spoofed indicators: {total}")
    lines.append("")
    if total == 0:
        lines.append("No attack-pattern indicators were detected for spoofed predictions.")
        return "\n".join(lines)
    for name, data in attack_counts.items():
        if data["count"] > 0:
            label = name.replace("_", " ").title()
            indicators = ", ".join(data["indicators"]) if data["indicators"] else "-"
            lines.append(f"{label}: {data['count']} ({data['percentage']:.1f}%)")
            lines.append(f"  Indicators: {indicators}")
    return "\n".join(lines)


def load_model_bundle(path: str):
    """Load both the new bundle format and the older GUI format."""
    obj = joblib.load(path)

    if not isinstance(obj, dict):
        raise ValueError("This pkl is not a model bundle. Save it from this GUI or include preprocessing objects.")

    model = obj.get("model") or obj.get("ensemble") or obj.get("ensemble_model")
    imputer = obj.get("imputer")
    scaler = obj.get("scaler")
    normalizer = obj.get("normalizer")
    features = obj.get("feature_names") or obj.get("features")

    missing = []
    if model is None: missing.append("model/ensemble")
    if imputer is None: missing.append("imputer")
    if scaler is None: missing.append("scaler")
    if normalizer is None: missing.append("normalizer")
    if not features: missing.append("feature_names/features")
    if missing:
        raise ValueError(f"Model pkl is missing: {missing}")

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "normalizer": normalizer,
        "feature_names": list(features),
        "performance": obj.get("performance", {}),
        "created_at": obj.get("created_at", "unknown"),
        "source_training_file": obj.get("source_training_file", "unknown"),
        "auto_model_path": obj.get("auto_model_path", path),
        "latest_model_path": obj.get("latest_model_path", path),
    }


# ---------------------------------------------------------------------
# ML service
# ---------------------------------------------------------------------
class GPSEngine:
    def __init__(self, on_progress=None, is_stopped=None):
        self._progress_cb = on_progress
        self._stopped = is_stopped
        self.detector = None
        self.training_df = None
        self.model_bundle = None
        self.prediction_df = None
        self.prediction_feature_df = None
        self.attack_counts = None
        self.last_eval = None

    def progress(self, pct, msg=""):
        if self._progress_cb:
            self._progress_cb(pct, msg)

    def stopped(self):
        return bool(self._stopped and self._stopped())

    def train(self, training_path: str):
        self.progress(5, "Reading training dataset...")
        print("=" * 80)
        print("TRAIN MODE")
        print("=" * 80)
        print(f"Training dataset: {training_path}")

        raw_df = pd.read_csv(training_path)
        if "Data Type" in raw_df.columns and "label" not in raw_df.columns:
            raw_df["label"] = raw_df["Data Type"]

        validate_columns(raw_df, TRAIN_REQUIRED_COLS, "Training dataset")
        value_warnings = validate_gps_values(raw_df)
        for w in value_warnings:
            print(f"[WARNING] {w}")

        df = prepare_training_dataframe(raw_df)
        normal_count = int((df["label_numeric"] == 0).sum())
        spoof_count = int((df["label_numeric"] == 1).sum())
        print(f"[OK] Rows: {len(df):,}")
        print(f"[OK] Normal: {normal_count:,} | Spoofed: {spoof_count:,}")

        if self.stopped():
            return None

        self.progress(20, "Creating GPS features...")
        detector = HumanGPSDetector()
        df = detector.create_features(df)
        detector.analyze_data_quality(df)

        if self.stopped():
            return None

        self.progress(45, "Training ML models...")

        # Important: the original script also runs a time-based evaluation that
        # re-fits the preprocessing objects after the ensemble has already been
        # trained. For a deployable PKL, the saved imputer/scaler/normalizer must
        # stay matched with the trained ensemble, so we skip that extra re-fit here.
        def _skip_time_based_evaluation(_df):
            print("[INFO] Time-based evaluation skipped in GUI training to keep saved preprocessing matched with the trained model.")
        detector.evaluate_time_based_split = _skip_time_based_evaluation

        train_idx, test_idx, test_pred, test_probs, model_results = detector.train_and_evaluate(df)

        if self.stopped():
            return None

        y_test = df.iloc[test_idx]["label_numeric"].astype(int).values
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average="binary", zero_division=0,
        )

        detector.performance = {
            "train_accuracy": detector.performance.get("train_accuracy", 0),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        self.detector = detector
        self.training_df = df
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_model_path = os.path.join(AUTO_MODEL_DIR, f"gps_detector_model_{timestamp}.pkl")

        self.model_bundle = {
            "bundle_type": "gps_spoofing_detector_bundle",
            "created_at": created_at,
            "source_training_file": training_path,
            "model": detector.ensemble_model,
            "imputer": detector.imputer,
            "scaler": detector.scaler,
            "normalizer": detector.normalizer,
            "feature_names": detector.feature_names,
            "performance": detector.performance,
            "confusion_matrix": cm,
            "model_results": model_results,
            "required_input_columns": UNLABELED_REQUIRED_COLS,
            "auto_model_path": auto_model_path,
            "latest_model_path": LATEST_MODEL_PATH,
        }

        os.makedirs(AUTO_MODEL_DIR, exist_ok=True)
        joblib.dump(self.model_bundle, auto_model_path)
        joblib.dump(self.model_bundle, LATEST_MODEL_PATH)

        self.progress(100, "Training complete")
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Accuracy : {detector.performance['test_accuracy'] * 100:.2f}%")
        print(f"Precision: {detector.performance['precision'] * 100:.2f}%")
        print(f"Recall   : {detector.performance['recall'] * 100:.2f}%")
        print(f"F1 Score : {detector.performance['f1_score'] * 100:.2f}%")
        print(f"[OK] Auto-saved timestamped model: {auto_model_path}")
        print(f"[OK] Auto-updated latest model   : {LATEST_MODEL_PATH}")
        print("You can go to Predict Unlabeled now; the latest model is selected automatically.")
        return self.model_bundle

    def save_model(self, path: str):
        if not self.model_bundle:
            raise ValueError("No trained model to save.")
        joblib.dump(self.model_bundle, path)
        print(f"[OK] Model bundle saved: {path}")

    def load_model(self, path: str):
        self.model_bundle = load_model_bundle(path)
        print("=" * 80)
        print("MODEL LOADED")
        print("=" * 80)
        print(f"Model pkl: {path}")
        print(f"Created at: {self.model_bundle.get('created_at')}")
        print(f"Features: {len(self.model_bundle['feature_names'])}")
        return self.model_bundle

    def predict_unlabeled(self, model_path: str, unlabeled_path: str):
        self.progress(5, "Loading model...")
        bundle = self.load_model(model_path)

        self.progress(15, "Reading unlabeled dataset...")
        print("\n" + "=" * 80)
        print("PREDICT UNLABELED MODE")
        print("=" * 80)
        print(f"Unlabeled dataset: {unlabeled_path}")

        raw_df = pd.read_csv(unlabeled_path)
        validate_columns(raw_df, UNLABELED_REQUIRED_COLS, "Unlabeled dataset")
        value_warnings = validate_gps_values(raw_df)
        for w in value_warnings:
            print(f"[WARNING] {w}")

        if "label" in raw_df.columns:
            print("[INFO] This file has a label column, but prediction will ignore it.")

        self.progress(35, "Creating GPS features...")
        feature_df = build_features_unlabeled(raw_df)
        feature_names = bundle["feature_names"]
        missing_features = [f for f in feature_names if f not in feature_df.columns]
        if missing_features:
            raise ValueError(f"After feature engineering, these model features are missing: {missing_features}")

        if self.stopped():
            return None

        self.progress(60, "Running model prediction...")
        X = feature_df[feature_names].copy()
        X_imp = bundle["imputer"].transform(X)
        X_scaled = bundle["scaler"].transform(X_imp)
        X_norm = bundle["normalizer"].transform(X_scaled)

        pred = bundle["model"].predict(X_norm).astype(int)
        if hasattr(bundle["model"], "predict_proba"):
            prob = bundle["model"].predict_proba(X_norm)
            confidence = np.array([prob[i, p] * 100 for i, p in enumerate(pred)])
            spoof_probability = prob[:, 1] * 100 if prob.shape[1] > 1 else np.zeros(len(pred))
        else:
            confidence = np.full(len(pred), np.nan)
            spoof_probability = np.full(len(pred), np.nan)

        out = raw_df.copy().reset_index(drop=True)
        out["prediction_numeric"] = pred
        out["prediction"] = np.where(pred == 1, "spoofed", "normal")
        out["confidence"] = confidence
        out["spoof_probability"] = spoof_probability

        self.prediction_df = out
        self.prediction_feature_df = feature_df
        self.attack_counts = detect_attack_types(feature_df, pred)

        os.makedirs(AUTO_PRED_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_pred_path = os.path.join(AUTO_PRED_DIR, f"unlabeled_predictions_{ts}.csv")
        latest_pred_path = os.path.join(AUTO_PRED_DIR, "latest_unlabeled_predictions.csv")
        out.to_csv(auto_pred_path, index=False)
        out.to_csv(latest_pred_path, index=False)
        self.last_prediction_path = auto_pred_path
        self.latest_prediction_path = latest_pred_path

        normal_pred = int((pred == 0).sum())
        spoof_pred = int((pred == 1).sum())
        self.progress(100, "Prediction complete")

        print("\nPREDICTION COMPLETE")
        print(f"Rows: {len(out):,}")
        print(f"Predicted normal : {normal_pred:,} ({normal_pred / len(out) * 100:.2f}%)")
        print(f"Predicted spoofed: {spoof_pred:,} ({spoof_pred / len(out) * 100:.2f}%)")
        if not np.isnan(confidence).all():
            print(f"Average confidence: {np.nanmean(confidence):.2f}%")
        print(f"[OK] Auto-saved timestamped predictions: {auto_pred_path}")
        print(f"[OK] Auto-updated latest predictions   : {latest_pred_path}")
        print("\n" + make_attack_report(self.attack_counts))
        return out

    def save_predictions(self, path: str):
        if self.prediction_df is None:
            raise ValueError("No predictions to save.")
        self.prediction_df.to_csv(path, index=False)
        print(f"[OK] Predictions saved: {path}")

    def evaluate_predictions(self, predictions_path: str, true_labels_path: str):
        print("\n" + "=" * 80)
        print("EVALUATE PREDICTIONS MODE")
        print("=" * 80)
        print(f"Predictions file: {predictions_path}")
        print(f"True labels file: {true_labels_path}")

        pred_df = pd.read_csv(predictions_path)
        true_df = pd.read_csv(true_labels_path)

        if "prediction_numeric" in pred_df.columns:
            y_pred = pd.to_numeric(pred_df["prediction_numeric"], errors="coerce").astype("Int64")
        elif "prediction" in pred_df.columns:
            y_pred = normalize_label_series(pred_df["prediction"])
        else:
            raise ValueError("Predictions file must contain prediction_numeric or prediction column.")

        if "label_numeric" in true_df.columns:
            y_true = pd.to_numeric(true_df["label_numeric"], errors="coerce").astype("Int64")
        elif "label" in true_df.columns:
            y_true = normalize_label_series(true_df["label"])
        elif "Data Type" in true_df.columns:
            y_true = normalize_label_series(true_df["Data Type"])
        else:
            raise ValueError("True labels file must contain label, label_numeric, or Data Type column.")

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: true={len(y_true)}, predictions={len(y_pred)}")
        if y_true.isna().any() or y_pred.isna().any():
            raise ValueError("Some true labels or predictions could not be converted to 0/1.")

        y_true = y_true.astype(int).values
        y_pred = y_pred.astype(int).values

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report = classification_report(y_true, y_pred, target_names=["Normal", "Spoofed"], zero_division=0)

        self.last_eval = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "report": report,
        }

        print("\nEVALUATION COMPLETE")
        print(f"Accuracy : {acc * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall   : {recall * 100:.2f}%")
        print(f"F1 Score : {f1 * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        return self.last_eval


# ---------------------------------------------------------------------
# GUI application
# ---------------------------------------------------------------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GPS Spoofing Detector - Train / Predict / Evaluate")
        self.geometry("1380x860")
        self.minsize(1120, 700)
        self.configure(fg_color=COLORS["bg"])

        self._running = False
        self._stop_req = False
        self.engine = GPSEngine(on_progress=self._on_progress, is_stopped=lambda: self._stop_req)

        self.train_file = None
        self.model_file = None
        self.unlabeled_file = None
        self.predictions_file = None
        self.true_labels_file = None

        self._apply_tree_style()
        self._build_ui()
        sys.stdout = Redirector(self.console_box, self.dashboard_log)
        self._welcome()
        self._try_autoload_latest_model()

    def _apply_tree_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "GPS.Treeview",
            background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"],
            fieldbackground=COLORS["tbl_odd"], rowheight=38,
            font=("Segoe UI", 11), borderwidth=0, relief="flat",
        )
        style.configure(
            "GPS.Treeview.Heading",
            background=COLORS["tbl_head"], foreground=COLORS["text"],
            font=("Segoe UI", 12, "bold"), relief="flat", borderwidth=0,
        )
        style.map("GPS.Treeview", background=[("selected", COLORS["accent"])], foreground=[("selected", "#ffffff")])

    def _build_ui(self):
        self.header = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=64)
        self.header.pack(side="top", fill="x")
        self.header.pack_propagate(False)

        ctk.CTkLabel(
            self.header,
            text="GPS Spoofing Detector - Train / Predict / Evaluate",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left", padx=24, pady=12)

        self.header_status = ctk.CTkLabel(
            self.header, text="● Idle", font=ctk.CTkFont(size=13), text_color=COLORS["green"],
        )
        self.header_status.pack(side="right", padx=18)

        ctk.CTkFrame(self, fg_color=COLORS["border"], height=1).pack(side="top", fill="x")

        self.body = ctk.CTkFrame(self, fg_color="transparent")
        self.body.pack(side="top", fill="both", expand=True)

        self.sidebar = ctk.CTkFrame(self.body, fg_color=COLORS["surface"], corner_radius=0, width=NAV_W)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        ctk.CTkFrame(self.body, fg_color=COLORS["border"], width=1).pack(side="left", fill="y")

        self.content = ctk.CTkFrame(self.body, fg_color=COLORS["bg"])
        self.content.pack(side="left", fill="both", expand=True)

        self.footer = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=32)
        self.footer.pack(side="bottom", fill="x")
        self.footer.pack_propagate(False)
        ctk.CTkFrame(self, fg_color=COLORS["border"], height=1).pack(side="bottom", fill="x")
        self.status_label = ctk.CTkLabel(self.footer, text="Ready", font=ctk.CTkFont(size=12), text_color=COLORS["muted"])
        self.status_label.pack(side="left", padx=18)
        ctk.CTkLabel(
            self.footer, text="Powered by PROJECT_2.py", font=ctk.CTkFont(size=12), text_color=COLORS["dim"],
        ).pack(side="right", padx=18)

        self._build_sidebar()
        self._build_pages()
        self._nav("dashboard")

    def _build_sidebar(self):
        ctk.CTkLabel(
            self.sidebar, text="NAVIGATION", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["dim"],
        ).pack(anchor="w", padx=16, pady=(24, 8))

        self.nav_buttons = {}
        nav_items = [
            ("dashboard", "Dashboard", "⊞"),
            ("train", "Train Model", "◎"),
            ("predict", "Predict Unlabeled", "▶"),
            ("evaluate", "Evaluate", "✓"),
            ("console", "Console", "≡"),
            ("about", "About", "ℹ"),
        ]
        for key, text, icon in nav_items:
            btn = ctk.CTkButton(
                self.sidebar,
                text=f"  {icon}   {text}",
                command=lambda k=key: self._nav(k),
                fg_color="transparent", hover_color=COLORS["card"],
                text_color=COLORS["text"], anchor="w", height=42, corner_radius=8,
                font=ctk.CTkFont(size=14),
            )
            btn.pack(fill="x", padx=10, pady=3)
            self.nav_buttons[key] = btn

        ctk.CTkFrame(self.sidebar, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16, pady=18)

        ctk.CTkLabel(
            self.sidebar, text="SHORTCUTS", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["dim"],
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self.stop_btn = ctk.CTkButton(
            self.sidebar, text="■ Stop Current Job", command=self._stop,
            fg_color=COLORS["card"], hover_color=COLORS["border"],
            text_color=COLORS["red"], height=38, corner_radius=8,
            font=ctk.CTkFont(size=13), state="disabled",
        )
        self.stop_btn.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkButton(
            self.sidebar, text="⌫ Clear Console", command=self._clear_console,
            fg_color=COLORS["card"], hover_color=COLORS["border"],
            text_color=COLORS["text"], height=38, corner_radius=8,
            font=ctk.CTkFont(size=13),
        ).pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkFrame(self.sidebar, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16, pady=18)

        self.sidebar_info = ctk.CTkLabel(
            self.sidebar,
            text="Train → Save PKL\nPredict unlabeled → Save CSV\nEvaluate only if true labels exist",
            font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
            wraplength=NAV_W - 28, justify="left",
        )
        self.sidebar_info.pack(anchor="w", padx=16, pady=(0, 8))

    def _build_pages(self):
        self.pages = {}
        self.pages["dashboard"] = self._make_dashboard()
        self.pages["train"] = self._make_train_page()
        self.pages["predict"] = self._make_predict_page()
        self.pages["evaluate"] = self._make_evaluate_page()
        self.pages["console"] = self._make_console_page()
        self.pages["about"] = self._make_about_page()

    def _nav(self, key):
        for k, btn in self.nav_buttons.items():
            btn.configure(fg_color=COLORS["card"] if k == key else "transparent")
        for k, page in self.pages.items():
            if k == key:
                page.pack(fill="both", expand=True)
            else:
                page.pack_forget()

    def _make_dashboard(self):
        page = ctk.CTkFrame(self.content, fg_color=COLORS["bg"])

        progress_card = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        progress_card.pack(fill="x", padx=20, pady=(20, 12))

        self.progress_bar = ctk.CTkProgressBar(progress_card, height=8, fg_color=COLORS["card"], progress_color=COLORS["accent"])
        self.progress_bar.pack(fill="x", padx=18, pady=(16, 6))
        self.progress_bar.set(0)
        self.progress_label = ctk.CTkLabel(progress_card, text="Ready", font=ctk.CTkFont(size=12), text_color=COLORS["muted"])
        self.progress_label.pack(anchor="w", padx=18, pady=(0, 14))

        grid = tk.Frame(page, bg=COLORS["bg"])
        grid.pack(fill="x", padx=14, pady=(0, 12))
        for i in range(4):
            grid.columnconfigure(i, weight=1)

        self.card_accuracy = StatCard(grid, "ACCURACY", COLORS["accent"])
        self.card_precision = StatCard(grid, "PRECISION", COLORS["green"])
        self.card_recall = StatCard(grid, "RECALL", COLORS["yellow"])
        self.card_f1 = StatCard(grid, "F1 SCORE", COLORS["purple"])
        for i, card in enumerate([self.card_accuracy, self.card_precision, self.card_recall, self.card_f1]):
            card.grid(row=0, column=i, sticky="ew", padx=8, pady=8)

        log_wrap = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        log_wrap.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        ctk.CTkLabel(log_wrap, text="Live Console", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(12, 6))
        self.dashboard_log = ctk.CTkTextbox(log_wrap, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8, wrap="word")
        self.dashboard_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        return page

    def _make_train_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color=COLORS["bg"])
        

        card = self._card(page)
        self.train_file_label = self._file_row(card, "Training dataset", "No training file selected", self._browse_train_file)
        self.train_btn = self._action_button(card, "Train Model", self._start_training, COLORS["accent"])
        self.save_model_btn = self._action_button(card, "Save Extra Copy PKL", self._save_model, COLORS["green"], state="disabled")

        results = self._card(page)
        ctk.CTkLabel(results, text="Training Summary", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 8))
        self.train_summary_box = ctk.CTkTextbox(results, height=300, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8)
        self.train_summary_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.train_summary_box.insert("1.0", "No training yet.")

        self.train_results_card = self._card(page)
        self._init_embedded_results(self.train_results_card, "Training Results")
        return page

    def _make_predict_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color=COLORS["bg"])
        
        card = self._card(page)
        self.model_file_label = self._file_row(card, "Model PKL", "Auto: no trained model found yet", self._browse_model_file)
        self.unlabeled_file_label = self._file_row(card, "Unlabeled dataset", "No unlabeled file selected", self._browse_unlabeled_file)
        self.predict_btn = self._action_button(card, "Run Prediction", self._start_prediction, COLORS["accent"])
        self.save_predictions_btn = self._action_button(card, "Save Predictions CSV", self._save_predictions, COLORS["green"], state="disabled")

        summary = self._card(page)
        ctk.CTkLabel(summary, text="Prediction Summary", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 8))
        self.predict_summary_box = ctk.CTkTextbox(summary, height=230, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8)
        self.predict_summary_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.predict_summary_box.insert("1.0", "No predictions yet.")

        self.predict_results_card = self._card(page)
        self._init_embedded_results(self.predict_results_card, "Prediction Results")
        return page

    def _make_evaluate_page(self):
        page = ctk.CTkScrollableFrame(self.content, fg_color=COLORS["bg"])
        

        card = self._card(page)
        self.predictions_file_label = self._file_row(card, "Predictions CSV", "No predictions file selected", self._browse_predictions_file)
        self.true_labels_file_label = self._file_row(card, "True labels CSV", "No true labels file selected", self._browse_true_labels_file)
        self.evaluate_btn = self._action_button(card, "Evaluate Accuracy", self._start_evaluation, COLORS["accent"])

        out = self._card(page)
        ctk.CTkLabel(out, text="Evaluation Results", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 8))
        self.evaluate_summary_box = ctk.CTkTextbox(out, height=250, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8)
        self.evaluate_summary_box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.evaluate_summary_box.insert("1.0", "No evaluation yet.")

        self.evaluate_results_card = self._card(page)
        self._init_embedded_results(self.evaluate_results_card, "Evaluation Results")
        return page

    def _make_results_page(self):
        page = ctk.CTkFrame(self.content, fg_color=COLORS["bg"])
       

        self.results_scroll = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg"])
        self.results_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.results_scroll.columnconfigure(0, weight=1)

        self.results_empty = ctk.CTkLabel(
            self.results_scroll,
            text="No results yet.\nTrain a model or run evaluation first.",
            font=ctk.CTkFont(size=16),
            text_color=COLORS["muted"],
            justify="center",
        )
        self.results_empty.grid(row=0, column=0, sticky="ew", pady=80)
        return page

    def _make_console_page(self):
        page = ctk.CTkFrame(self.content, fg_color=COLORS["bg"])
        
        self.console_box = ctk.CTkTextbox(page, font=ctk.CTkFont(size=14, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=10, wrap="word")
        self.console_box.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        return page

    def _make_about_page(self):
        page = ctk.CTkFrame(self.content, fg_color=COLORS["bg"])
        box = self._card(page)
        ctk.CTkLabel(box, text="GPS Spoofing Detector", font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"]).pack(pady=(28, 6))
        txt = (
            "This GUI separates the workflow into three independent parts:\n\n"
            "1. Train Model: labeled dataset → trained pkl bundle.\n"
            "2. Predict Unlabeled: pkl + unlabeled GPS dataset → predictions CSV.\n"
            "3. Evaluate: predictions CSV + true labels CSV → accuracy/confusion matrix.\n\n"
            "The unlabeled file cannot produce real accuracy unless you provide a separate true-label file."
        )
        ctk.CTkLabel(box, text=txt, font=ctk.CTkFont(size=14), text_color=COLORS["muted"], justify="left").pack(anchor="w", padx=30, pady=(10, 28))
        return page

    def _page_title(self, parent, title, subtitle):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(20, 2))
        ctk.CTkLabel(parent, text=subtitle, font=ctk.CTkFont(size=13), text_color=COLORS["muted"]).pack(anchor="w", padx=20, pady=(0, 14))

    def _card(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        frame.pack(fill="x", padx=20, pady=(0, 16))
        return frame

    def _file_row(self, parent, title, default_text, command):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(14, 6))
        ctk.CTkLabel(row, text=title + ":", width=150, anchor="w", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["text"]).pack(side="left")
        lbl = ctk.CTkLabel(row, text=default_text, anchor="w", font=ctk.CTkFont(size=12), text_color=COLORS["muted"], wraplength=680)
        lbl.pack(side="left", fill="x", expand=True, padx=(8, 10))
        ctk.CTkButton(row, text="Browse", command=command, width=100, height=34, fg_color=COLORS["card"], hover_color=COLORS["border"], text_color=COLORS["text"]).pack(side="right")
        return lbl

    def _action_button(self, parent, text, command, color, state="normal"):
        btn = ctk.CTkButton(
            parent, text=text, command=command, fg_color=color, hover_color=color,
            text_color="white", height=40, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"), state=state,
        )
        btn.pack(fill="x", padx=18, pady=(8, 12))
        return btn

    # ------------------ browse actions ------------------
    def _browse_train_file(self):
        path = filedialog.askopenfilename(title="Select Training CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.train_file = path
            self.train_file_label.configure(text=os.path.basename(path), text_color=COLORS["text"])
            print(f"[OK] Training file selected: {path}")

    def _browse_model_file(self):
        path = filedialog.askopenfilename(title="Select Model PKL", filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if path:
            self.model_file = path
            self.model_file_label.configure(text=os.path.basename(path), text_color=COLORS["text"])
            print(f"[OK] Model file selected: {path}")

    def _browse_unlabeled_file(self):
        path = filedialog.askopenfilename(title="Select Unlabeled CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.unlabeled_file = path
            self.unlabeled_file_label.configure(text=os.path.basename(path), text_color=COLORS["text"])
            print(f"[OK] Unlabeled file selected: {path}")

    def _browse_predictions_file(self):
        path = filedialog.askopenfilename(title="Select Predictions CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.predictions_file = path
            self.predictions_file_label.configure(text=os.path.basename(path), text_color=COLORS["text"])
            print(f"[OK] Predictions file selected: {path}")

    def _browse_true_labels_file(self):
        path = filedialog.askopenfilename(title="Select True Labels CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.true_labels_file = path
            self.true_labels_file_label.configure(text=os.path.basename(path), text_color=COLORS["text"])
            print(f"[OK] True labels file selected: {path}")


    def _try_autoload_latest_model(self):
        """Use the latest auto-saved model without asking the user to browse/upload it."""
        if os.path.exists(LATEST_MODEL_PATH):
            try:
                self.model_file = LATEST_MODEL_PATH
                if hasattr(self, "model_file_label"):
                    self.model_file_label.configure(
                        text=f"Auto latest: {os.path.basename(LATEST_MODEL_PATH)}",
                        text_color=COLORS["green"],
                    )
                # Validate the bundle now, but do not force the user to browse it.
                self.engine.load_model(LATEST_MODEL_PATH)
                print(f"[OK] Auto-loaded latest trained model: {LATEST_MODEL_PATH}")
            except Exception as exc:
                print(f"[WARNING] Found latest model but could not load it: {exc}")

    def _capture_job_output(self, func):
        """Run a job and return both its result and the exact detailed console log."""
        buf = io.StringIO()
        current_stdout = sys.stdout
        tee = TeeCapture(current_stdout, buf)
        with contextlib.redirect_stdout(tee):
            result = func()
        return {"result": result, "log": buf.getvalue()}

    # ------------------ threaded jobs ------------------
    def _set_running(self, running: bool, msg=""):
        self._running = running
        self.stop_btn.configure(state="normal" if running else "disabled")
        state = "disabled" if running else "normal"
        self.train_btn.configure(state=state)
        self.predict_btn.configure(state=state)
        self.evaluate_btn.configure(state=state)
        self.header_status.configure(text="● Running" if running else "● Idle", text_color=COLORS["yellow"] if running else COLORS["green"])
        if msg:
            self._set_status(msg)

    def _run_thread(self, target, on_done=None):
        if self._running:
            return
        self._stop_req = False
        self.progress_bar.set(0)
        self._set_running(True, "Running...")

        def worker():
            result = None
            error = None
            try:
                result = target()
            except Exception as e:
                error = e
            finally:
                self.after(0, lambda: self._finish_thread(error, result, on_done))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_thread(self, error, result, on_done):
        self._set_running(False, "Ready")
        if error:
            print(f"\n[ERROR] {error}")
            messagebox.showerror("Error", str(error))
            return
        if self._stop_req:
            self._set_status("Stopped")
            print("[INFO] Job stopped.")
            return
        if on_done:
            on_done(result)

    def _start_training(self):
        if not self.train_file:
            messagebox.showwarning("Missing file", "Choose a training dataset first.")
            return
        self._clear_box(self.train_summary_box)
        self._run_thread(lambda: self._capture_job_output(lambda: self.engine.train(self.train_file)), self._training_done)

    def _training_done(self, payload):
        if not payload:
            return
        detailed_log = ""
        bundle = payload
        if isinstance(payload, dict) and "result" in payload and "log" in payload:
            detailed_log = payload.get("log", "")
            bundle = payload.get("result")
        if not bundle:
            return

        p = bundle["performance"]
        self.card_accuracy.set(f"{p['test_accuracy'] * 100:.1f}%")
        self.card_precision.set(f"{p['precision'] * 100:.1f}%")
        self.card_recall.set(f"{p['recall'] * 100:.1f}%")
        self.card_f1.set(f"{p['f1_score'] * 100:.1f}%")
        self.save_model_btn.configure(state="normal")

        self.model_file = bundle.get("latest_model_path") or bundle.get("auto_model_path")
        if self.model_file and hasattr(self, "model_file_label"):
            self.model_file_label.configure(
                text=f"Auto latest: {os.path.basename(self.model_file)}",
                text_color=COLORS["green"],
            )

        cm = bundle.get("confusion_matrix")
        model_results = bundle.get("model_results", [])
        rows = len(self.engine.training_df) if self.engine.training_df is not None else "unknown"
        normal_count = int((self.engine.training_df["label_numeric"] == 0).sum()) if self.engine.training_df is not None else "unknown"
        spoof_count = int((self.engine.training_df["label_numeric"] == 1).sum()) if self.engine.training_df is not None else "unknown"

        model_lines = []
        if model_results:
            model_lines.append("Individual Models:")
            for item in model_results:
                model_lines.append(
                    f"  - {item.get('Model', 'model')}: "
                    f"Train={item.get('Train_Acc', 0):.2f}% | Test={item.get('Test_Acc', 0):.2f}%"
                )

        final_summary = (
            "FINAL TRAINING SUMMARY\n" +
            "=" * 80 + "\n"
            f"Training file       : {bundle['source_training_file']}\n"
            f"Created at          : {bundle['created_at']}\n"
            f"Rows                : {rows}\n"
            f"Normal rows          : {normal_count}\n"
            f"Spoofed rows         : {spoof_count}\n"
            f"Features             : {len(bundle['feature_names'])}\n"
            f"Auto saved model     : {bundle.get('auto_model_path')}\n"
            f"Latest model used    : {bundle.get('latest_model_path')}\n\n"
            f"Accuracy             : {p['test_accuracy'] * 100:.2f}%\n"
            f"Precision            : {p['precision'] * 100:.2f}%\n"
            f"Recall               : {p['recall'] * 100:.2f}%\n"
            f"F1 Score             : {p['f1_score'] * 100:.2f}%\n\n"
            f"Confusion Matrix [Normal, Spoofed]:\n{cm}\n\n"
            + "\n".join(model_lines) + "\n\n"
            "FEATURES USED:\n"
            + "\n".join([f"  - {name}" for name in bundle["feature_names"]])
            + "\n\n"
            "DETAILED TRAINING LOG:\n" +
            "=" * 80 + "\n"
            + detailed_log
        )
        self._set_text(self.train_summary_box, final_summary)

        self._fill_embedded_results(
            self.train_results_card,
            "Training Results",
            metrics={
                "accuracy": p.get("test_accuracy", 0),
                "precision": p.get("precision", 0),
                "recall": p.get("recall", 0),
                "f1_score": p.get("f1_score", 0),
            },
            cm=cm,
            model_results=model_results,
            extra_text=(
                f"Training file: {bundle['source_training_file']}\n"
                f"Rows: {rows} | Normal: {normal_count} | Spoofed: {spoof_count}\n"
                f"Features used: {len(bundle['feature_names'])}\n"
                f"Latest model: {bundle.get('latest_model_path')}"
            ),
        )
        self._nav("train")

        self._set_status("Training complete - model auto-saved and selected")
        self.header_status.configure(text="● Complete", text_color=COLORS["green"])

    def _start_prediction(self):
        if not self.model_file and os.path.exists(LATEST_MODEL_PATH):
            self.model_file = LATEST_MODEL_PATH
            self.model_file_label.configure(text=f"Auto latest: {os.path.basename(LATEST_MODEL_PATH)}", text_color=COLORS["green"])
        if not self.model_file:
            messagebox.showwarning("Missing model", "Train a model first, or choose a model pkl manually.")
            return
        if not self.unlabeled_file:
            messagebox.showwarning("Missing file", "Choose an unlabeled dataset first.")
            return
        self._clear_box(self.predict_summary_box)
        self._run_thread(lambda: self._capture_job_output(lambda: self.engine.predict_unlabeled(self.model_file, self.unlabeled_file)), self._prediction_done)

    def _prediction_done(self, payload):
        if payload is None:
            return
        detailed_log = ""
        out = payload
        if isinstance(payload, dict) and "result" in payload and "log" in payload:
            detailed_log = payload.get("log", "")
            out = payload.get("result")
        if out is None:
            return

        self.save_predictions_btn.configure(state="normal")
        total = len(out)
        normal_pred = int((out["prediction_numeric"] == 0).sum())
        spoof_pred = int((out["prediction_numeric"] == 1).sum())
        avg_conf = pd.to_numeric(out["confidence"], errors="coerce").mean()
        report = make_attack_report(self.engine.attack_counts or {})
        auto_pred = getattr(self.engine, "last_prediction_path", "")
        latest_pred = getattr(self.engine, "latest_prediction_path", "")
        summary = (
            "PREDICTION COMPLETE\n" +
            "=" * 80 + "\n"
            f"Rows                 : {total}\n"
            f"Predicted normal     : {normal_pred} ({normal_pred / total * 100:.2f}%)\n"
            f"Predicted spoofed    : {spoof_pred} ({spoof_pred / total * 100:.2f}%)\n"
            f"Average confidence   : {avg_conf:.2f}%\n"
            f"Auto saved CSV       : {auto_pred}\n"
            f"Latest predictions   : {latest_pred}\n\n"
            f"{report}\n\n"
            "DETAILED PREDICTION LOG:\n" +
            "=" * 80 + "\n"
            f"{detailed_log}"
        )
        self._set_text(self.predict_summary_box, summary)
        # Prediction has no true labels, so Accuracy/F1/Confusion Matrix cannot be calculated here.
        # Show the prediction distribution inside the Predict page.
        self._fill_embedded_results(
            self.predict_results_card,
            "Prediction Results",
            prediction_stats={
                "TOTAL ROWS": f"{total:,}",
                "NORMAL": f"{normal_pred:,}",
                "SPOOFED": f"{spoof_pred:,}",
                "AVG CONF": f"{avg_conf:.2f}%",
            },
            extra_text=(
                "This dataset has no true labels, so Accuracy, Precision, Recall, F1, and Confusion Matrix are not available.\n"
                "Use the Evaluate page only if you have the true labels for the same rows and same order.\n\n"
                f"Predicted normal: {normal_pred} ({normal_pred / total * 100:.2f}%)\n"
                f"Predicted spoofed: {spoof_pred} ({spoof_pred / total * 100:.2f}%)\n"
                f"Average confidence: {avg_conf:.2f}%\n"
                f"Auto saved CSV: {auto_pred}\n"
                f"Latest predictions: {latest_pred}\n\n"
                f"{report}"
            ),
        )
        self._nav("predict")
        self._set_status("Prediction complete - CSV auto-saved")
        self.header_status.configure(text="● Complete", text_color=COLORS["green"])

    def _start_evaluation(self):
        if not self.predictions_file:
            messagebox.showwarning("Missing file", "Choose predictions CSV first.")
            return
        if not self.true_labels_file:
            messagebox.showwarning("Missing file", "Choose true labels CSV first.")
            return
        self._clear_box(self.evaluate_summary_box)
        self._run_thread(lambda: self.engine.evaluate_predictions(self.predictions_file, self.true_labels_file), self._evaluation_done)

    def _evaluation_done(self, result):
        if not result:
            return
        summary = (
            "EVALUATION COMPLETE\n" +
            "=" * 50 + "\n"
            f"Accuracy : {result['accuracy'] * 100:.2f}%\n"
            f"Precision: {result['precision'] * 100:.2f}%\n"
            f"Recall   : {result['recall'] * 100:.2f}%\n"
            f"F1 Score : {result['f1_score'] * 100:.2f}%\n\n"
            "Confusion Matrix [Normal/Spoofed]:\n"
            f"{result['confusion_matrix']}\n\n"
            "Classification Report:\n"
            f"{result['report']}\n"
        )
        self._set_text(self.evaluate_summary_box, summary)
        self._fill_embedded_results(
            self.evaluate_results_card,
            "Evaluation Results",
            metrics={
                "accuracy": result.get("accuracy", 0),
                "precision": result.get("precision", 0),
                "recall": result.get("recall", 0),
                "f1_score": result.get("f1_score", 0),
            },
            cm=result.get("confusion_matrix"),
            report=result.get("report"),
        )
        self._nav("evaluate")
        self.card_accuracy.set(f"{result['accuracy'] * 100:.1f}%")
        self.card_precision.set(f"{result['precision'] * 100:.1f}%")
        self.card_recall.set(f"{result['recall'] * 100:.1f}%")
        self.card_f1.set(f"{result['f1_score'] * 100:.1f}%")
        self._set_status("Evaluation complete")
        self.header_status.configure(text="● Complete", text_color=COLORS["green"])

    # ------------------ save actions ------------------
    def _save_model(self):
        if not self.engine.model_bundle:
            messagebox.showwarning("No model", "Train a model first.")
            return
        default = f"gps_detector_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        path = filedialog.asksaveasfilename(title="Save Model PKL", defaultextension=".pkl", initialfile=default, filetypes=[("Pickle files", "*.pkl")])
        if not path:
            return
        try:
            self.engine.save_model(path)
            messagebox.showinfo("Saved", f"Model saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def _save_predictions(self):
        if self.engine.prediction_df is None:
            messagebox.showwarning("No predictions", "Run prediction first.")
            return
        default = f"unlabeled_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(title="Save Predictions CSV", defaultextension=".csv", initialfile=default, filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.engine.save_predictions(path)
            messagebox.showinfo("Saved", f"Predictions saved:\n{path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ------------------ misc ------------------

    def _init_embedded_results(self, parent, title):
        """Prepare an in-page results card with a placeholder."""
        for widget in parent.winfo_children():
            widget.destroy()
        ctk.CTkLabel(
            parent,
            text=title,
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=COLORS["text"],
        ).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            parent,
            text="No results yet.",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["muted"],
        ).pack(anchor="w", padx=18, pady=(0, 18))

    def _fill_embedded_results(self, parent, title, metrics=None, cm=None, model_results=None, report=None, extra_text=None, prediction_stats=None):
        """Render results inside the same page that produced them."""
        for widget in parent.winfo_children():
            widget.destroy()

        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(16, 8))
        ctk.CTkLabel(
            header,
            text=title,
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=COLORS["text"],
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            text=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["muted"],
        ).pack(side="right")

        if metrics:
            metrics_frame = ctk.CTkFrame(parent, fg_color="transparent")
            metrics_frame.pack(fill="x", padx=10, pady=(0, 14))
            for i in range(4):
                metrics_frame.columnconfigure(i, weight=1)

            def _fmt_metric(key):
                value = metrics.get(key, None)
                if value is None:
                    return "N/A"
                try:
                    return f"{float(value) * 100:.2f}%"
                except Exception:
                    return "N/A"

            self._metric_tile(metrics_frame, "ACCURACY", _fmt_metric("accuracy"), COLORS["accent"], 0, 0)
            self._metric_tile(metrics_frame, "PRECISION", _fmt_metric("precision"), COLORS["green"], 0, 1)
            self._metric_tile(metrics_frame, "RECALL", _fmt_metric("recall"), COLORS["yellow"], 0, 2)
            self._metric_tile(metrics_frame, "F1 SCORE", _fmt_metric("f1_score"), COLORS["purple"], 0, 3)

        if prediction_stats:
            stats_frame = ctk.CTkFrame(parent, fg_color="transparent")
            stats_frame.pack(fill="x", padx=10, pady=(0, 14))
            for i in range(4):
                stats_frame.columnconfigure(i, weight=1)
            colors = [COLORS["accent"], COLORS["green"], COLORS["red"], COLORS["purple"]]
            for i, (name, value) in enumerate(prediction_stats.items()):
                self._metric_tile(stats_frame, name, value, colors[i % len(colors)], 0, i)

        if cm is not None:
            cm_card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            cm_card.pack(fill="x", padx=18, pady=(0, 14))
            ctk.CTkLabel(
                cm_card,
                text="Confusion Matrix",
                font=ctk.CTkFont(size=16, weight="bold"),
                text_color=COLORS["text"],
            ).pack(anchor="w", padx=18, pady=(14, 4))
            ctk.CTkLabel(
                cm_card,
                text="Rows = Actual label, Columns = Predicted label",
                font=ctk.CTkFont(size=12),
                text_color=COLORS["muted"],
            ).pack(anchor="w", padx=18, pady=(0, 8))
            self._draw_cm_in_frame(cm_card, cm)

        if model_results:
            model_card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            model_card.pack(fill="x", padx=18, pady=(0, 14))
            ctk.CTkLabel(model_card, text="Individual Model Results", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(14, 10))
            cols = ("Model", "Train Accuracy", "Test Accuracy")
            tree = ttk.Treeview(model_card, columns=cols, show="headings", style="GPS.Treeview", height=min(5, len(model_results)))
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=180)
            tree.column("Model", anchor="w", width=260)
            for i, item in enumerate(model_results):
                tag = "even" if i % 2 == 0 else "odd"
                tree.insert("", "end", tags=(tag,), values=(
                    str(item.get("Model", "model")).replace("_", " ").title(),
                    f"{item.get('Train_Acc', 0):.2f}%",
                    f"{item.get('Test_Acc', 0):.2f}%",
                ))
            tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
            tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])
            tree.pack(fill="x", expand=True, padx=18, pady=(0, 18))

        if report or extra_text:
            details_card = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            details_card.pack(fill="both", expand=True, padx=18, pady=(0, 18))
            ctk.CTkLabel(details_card, text="Detailed Report", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(14, 8))
            box = ctk.CTkTextbox(details_card, height=210, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8)
            box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
            text = ""
            if extra_text:
                text += str(extra_text).strip() + "\n\n"
            if report:
                text += str(report).strip()
            box.insert("1.0", text.strip())

    def _metric_tile(self, parent, title, value, color, row, col):
        tile = ctk.CTkFrame(parent, fg_color=COLORS["card"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        tile.grid(row=row, column=col, sticky="ew", padx=8, pady=8)
        ctk.CTkFrame(tile, fg_color=color, height=4, corner_radius=3).pack(fill="x")
        ctk.CTkLabel(tile, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["muted"]).pack(anchor="w", padx=14, pady=(10, 0))
        ctk.CTkLabel(tile, text=value, font=ctk.CTkFont(size=28, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=14, pady=(2, 14))
        return tile

    def _fill_results_page(self, title, metrics, cm=None, model_results=None, report=None, extra_text=None):
        if not hasattr(self, "results_scroll"):
            return
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        header = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        header.columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=20, weight="bold"), text_color=COLORS["text"]).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(header, text=f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", font=ctk.CTkFont(size=12), text_color=COLORS["muted"]).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 14))

        metrics_frame = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        metrics_frame.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        for i in range(4):
            metrics_frame.columnconfigure(i, weight=1)

        def _fmt_metric(key):
            value = metrics.get(key, None)
            if value is None:
                return "N/A"
            try:
                return f"{float(value) * 100:.2f}%"
            except Exception:
                return "N/A"

        self._metric_tile(metrics_frame, "ACCURACY", _fmt_metric("accuracy"), COLORS["accent"], 0, 0)
        self._metric_tile(metrics_frame, "PRECISION", _fmt_metric("precision"), COLORS["green"], 0, 1)
        self._metric_tile(metrics_frame, "RECALL", _fmt_metric("recall"), COLORS["yellow"], 0, 2)
        self._metric_tile(metrics_frame, "F1 SCORE", _fmt_metric("f1_score"), COLORS["purple"], 0, 3)

        row_index = 2
        if cm is not None:
            cm_card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            cm_card.grid(row=row_index, column=0, sticky="ew", pady=(0, 16))
            ctk.CTkLabel(cm_card, text="Confusion Matrix", font=ctk.CTkFont(size=17, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 4))
            ctk.CTkLabel(cm_card, text="Rows = Actual label, Columns = Predicted label", font=ctk.CTkFont(size=12), text_color=COLORS["muted"]).pack(anchor="w", padx=18, pady=(0, 8))
            self._draw_cm_in_frame(cm_card, cm)
            row_index += 1

        if model_results:
            model_card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            model_card.grid(row=row_index, column=0, sticky="ew", pady=(0, 16))
            ctk.CTkLabel(model_card, text="Individual Model Results", font=ctk.CTkFont(size=17, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))
            cols = ("Model", "Train Accuracy", "Test Accuracy")
            tree = ttk.Treeview(model_card, columns=cols, show="headings", style="GPS.Treeview", height=min(5, len(model_results)))
            for col in cols:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=180)
            tree.column("Model", anchor="w", width=260)
            for i, item in enumerate(model_results):
                tag = "even" if i % 2 == 0 else "odd"
                tree.insert("", "end", tags=(tag,), values=(
                    str(item.get("Model", "model")).replace("_", " ").title(),
                    f"{item.get('Train_Acc', 0):.2f}%",
                    f"{item.get('Test_Acc', 0):.2f}%",
                ))
            tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
            tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])
            tree.pack(fill="x", expand=True, padx=18, pady=(0, 18))
            row_index += 1

        if report or extra_text:
            details_card = ctk.CTkFrame(self.results_scroll, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
            details_card.grid(row=row_index, column=0, sticky="ew", pady=(0, 16))
            ctk.CTkLabel(details_card, text="Detailed Report", font=ctk.CTkFont(size=17, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 8))
            box = ctk.CTkTextbox(details_card, height=260, font=ctk.CTkFont(size=13, family="Consolas"), fg_color=COLORS["bg"], text_color=COLORS["text"], border_width=1, border_color=COLORS["border"], corner_radius=8)
            box.pack(fill="both", expand=True, padx=18, pady=(0, 18))
            text = ""
            if extra_text:
                text += str(extra_text).strip() + "\n\n"
            if report:
                text += str(report).strip()
            box.insert("1.0", text.strip())
            row_index += 1

    def _draw_cm_in_frame(self, parent, cm):
        cm = np.asarray(cm, dtype=int)
        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        cmap = LinearSegmentedColormap.from_list(
            "gps_confusion_blue",
            ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"],
        )
        fig, ax = plt.subplots(figsize=(6.4, 4.9), dpi=115)
        fig.patch.set_facecolor(COLORS["surface"])
        ax.set_facecolor(COLORS["surface"])
        im = ax.imshow(cm, cmap=cmap)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted\nNormal", "Predicted\nSpoofed"], fontsize=11, color=COLORS["text"])
        ax.set_yticklabels(["Actual\nNormal", "Actual\nSpoofed"], fontsize=11, color=COLORS["text"])
        ax.set_xlabel("Predicted", fontsize=12, fontweight="bold", color=COLORS["muted"], labelpad=12)
        ax.set_ylabel("Actual", fontsize=12, fontweight="bold", color=COLORS["muted"], labelpad=12)

        max_val = cm.max() if cm.size and cm.max() > 0 else 1
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                txt_color = "white" if value > max_val * 0.55 else COLORS["text"]
                ax.text(
                    j, i,
                    f"{labels[i, j]}\n{value:,}",
                    ha="center", va="center",
                    color=txt_color,
                    fontsize=20,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9, colors=COLORS["muted"])
        cbar.outline.set_edgecolor(COLORS["border"])
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor="center", padx=18, pady=(0, 18))
        plt.close(fig)

    def _draw_cm(self, cm):
        # Optional pop-up version, using the same clear colors as the Results page.
        win = ctk.CTkToplevel(self)
        win.title("Confusion Matrix")
        win.geometry("680x560")
        win.configure(fg_color=COLORS["bg"])
        card = ctk.CTkFrame(win, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color=COLORS["border"])
        card.pack(fill="both", expand=True, padx=18, pady=18)
        ctk.CTkLabel(card, text="Confusion Matrix", font=ctk.CTkFont(size=18, weight="bold"), text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(card, text="Rows = Actual label, Columns = Predicted label", font=ctk.CTkFont(size=12), text_color=COLORS["muted"]).pack(anchor="w", padx=18, pady=(0, 8))
        self._draw_cm_in_frame(card, cm)

    def _stop(self):
        if self._running:
            self._stop_req = True
            self._set_status("Stopping...")
            print("[INFO] Stop requested...")

    def _on_progress(self, pct, msg=""):
        self.after(0, lambda: self.progress_bar.set(pct / 100))
        if msg:
            self.after(0, lambda: self.progress_label.configure(text=msg))
            self.after(0, lambda: self._set_status(msg))

    def _set_status(self, text):
        self.status_label.configure(text=text)

    def _clear_console(self):
        self._clear_box(self.console_box)
        self._clear_box(self.dashboard_log)

    def _clear_box(self, box):
        try:
            box.configure(state="normal")
            box.delete("1.0", "end")
        except Exception:
            pass

    def _set_text(self, box, text):
        self._clear_box(box)
        box.insert("1.0", text)

    def _welcome(self):
        print("GPS Spoofing Detector GUI")
        print(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print("Workflow:")
        print("1) Train Model: labeled CSV -> PKL")
        print("2) Predict Unlabeled: PKL + unlabeled CSV -> predictions CSV")
        print("3) Evaluate: predictions CSV + true labels CSV -> accuracy")
        print("=" * 60)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUTO_MODEL_DIR, exist_ok=True)
    os.makedirs(AUTO_PRED_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    app = App()
    app.mainloop()
