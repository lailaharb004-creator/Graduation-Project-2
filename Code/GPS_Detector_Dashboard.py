# gps_detector_pro_light.py - الإصدار النهائي
# يستورد كل شيء من PROJECT_2.py

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, f1_score)
import warnings, os, joblib, threading, sys
from datetime import datetime

# ============================================================
# IMPORT FROM PROJECT_2.py
# ============================================================
try:
    import PROJECT_2 as gps_engine_module
    from PROJECT_2 import (
        haversine_m,
        bearing_deg,
        circular_diff_deg,
        gps_time_to_seconds,
        add_time_delta_preserve_order,
        HumanGPSDetector,
    )
except ImportError as e:
    print(f"ERROR importing from PROJECT_2.py: {e}")
    print("Make sure PROJECT_2.py is in the same folder.")
    sys.exit(1)

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

warnings.filterwarnings('ignore')

# ─── Light Theme Colors ──────────────────────────────────────────────────────
ctk.set_default_color_theme("blue")
ctk.set_appearance_mode("light")

COLORS = {
    "bg":       "#ffffff",
    "surface":  "#ffffff",
    "card":     "#f8fafc",
    "border":   "#e2e8f0",
    "accent":   "#2563eb",
    "green":    "#16a34a",
    "yellow":   "#d97706",
    "red":      "#dc2626",
    "purple":   "#7c3aed",
    "text":     "#0f172a",
    "text_dark": "#1e293b",
    "muted":    "#475569",
    "dim":      "#94a3b8",
    "tbl_odd":  "#ffffff",
    "tbl_even": "#f1f5f9",
    "tbl_fg":   "#0f172a",
    "tbl_head": "#f1f5f9",
}

NAV_W = 224

# ─── Attack Type Detection ──────────────────────────────────────────────────

def detect_attack_types(df, predictions, feature_df):
    """Detect different types of GPS spoofing attacks."""
    attack_types = {
        "freeze": {"count": 0, "indicators": []},
        "replay": {"count": 0, "indicators": []},
        "geofence_evasion": {"count": 0, "indicators": []},
    }
    
    spoof_idx = np.where(predictions == 1)[0]
    if len(spoof_idx) == 0:
        return attack_types
    
    for idx in spoof_idx:
        if idx >= len(feature_df):
            continue
        row = feature_df.iloc[idx]
        
        velocity = row.get("velocity", 1) if not pd.isna(row.get("velocity", 1)) else 1
        distance_m = row.get("distance_m", 1) if not pd.isna(row.get("distance_m", 1)) else 1
        course_bearing_diff = row.get("course_bearing_diff", 0) if not pd.isna(row.get("course_bearing_diff", 0)) else 0
        velocity_diff = row.get("velocity_diff", 0) if not pd.isna(row.get("velocity_diff", 0)) else 0
        hdop_diff = row.get("hdop_diff", 0) if not pd.isna(row.get("hdop_diff", 0)) else 0
        sat_discrepancy = row.get("sat_discrepancy", 0) if not pd.isna(row.get("sat_discrepancy", 0)) else 0
        
        if velocity < 0.15 and distance_m < 0.5:
            attack_types["freeze"]["count"] += 1
            if "low_velocity" not in attack_types["freeze"]["indicators"]:
                attack_types["freeze"]["indicators"].append("low_velocity")
        
        if course_bearing_diff > 20 and velocity_diff < 0.2:
            attack_types["replay"]["count"] += 1
            if "course_bearing_diff" not in attack_types["replay"]["indicators"]:
                attack_types["replay"]["indicators"].append("course_bearing_diff")
        
        if hdop_diff > 0.5 and sat_discrepancy > 2:
            attack_types["geofence_evasion"]["count"] += 1
            if "hdop_sat_diff" not in attack_types["geofence_evasion"]["indicators"]:
                attack_types["geofence_evasion"]["indicators"].append("hdop_sat_diff")
    
    return attack_types

# ─── Text Redirector ──────────────────────────────────────────────────────────

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

# ─── ML Engine ──────────────────────────────────────────────────────────────

class GPSDetectorWrapper:
    """
    Thin adapter between the GUI and the real HumanGPSDetector engine
    (PROJECT_2.py). It calls the engine's actual pipeline
    methods directly — no re-implementation of loading/feature logic —
    and just collects the extra bits the GUI needs to display
    (per-model scores, confusion matrix, sample-size table, attack types).
    """
    def __init__(self, on_progress=None, on_status=None, is_stopped=None):
        self._prog = on_progress
        self._stat = on_status
        self._stopped = is_stopped

        # The real engine from PROJECT_2.py
        self.detector = HumanGPSDetector()
        self.ensemble = None
        self.scaler = None
        self.normalizer = None
        self.imputer = None
        self.features = []
        self.model_scores = {}
        self.perf = dict(test_accuracy=0, precision=0, recall=0, f1_score=0)
        self.cm = None
        self.y_test = None
        self.y_pred = None
        self.test_idx = None
        self.test_probs = None
        self.X_test_transformed = None
        self.attack_counts = None
        self.df = None

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}]  {msg}")

    def _progress(self, pct, msg=""):
        if self._prog:
            self._prog(pct, msg)

    def _check(self):
        return self._stopped and self._stopped()

    def load(self, path: str):
        """Delegates straight to the engine's own load_data (handles
        column validation, label mapping, and time-delta ordering)."""
        self._progress(5, "Reading CSV …")
        self._log("━" * 52)
        self._log("GPS Spoofing Detector  ·  Professional Edition")
        self._log("━" * 52)

        try:
            df = self.detector.load_data(path)
        except Exception as e:
            self._log(f"ERROR  Cannot read file: {e}")
            return None

        self.df = df
        return df

    def train(self, df) -> bool:
        if self._check():
            return False

        try:
            self._progress(20, "Engineering GPS features …")
            df = self.detector.create_features(df)
            self.features = self.detector.feature_names

            if self._check():
                return False
            self._progress(35, "Analyzing data quality …")
            self.detector.analyze_data_quality(df)

            if self._check():
                return False
            self._progress(45, "Generating diagnostic plots …")
            self.detector.plot_stream_behavior(df)

            if self._check():
                return False
            self._progress(55, "Training ensemble models …")
            train_idx, test_idx, test_ens, test_probs, model_results = \
                self.detector.train_and_evaluate(df)

            self.ensemble = self.detector.ensemble_model
            self.scaler = self.detector.scaler
            self.normalizer = self.detector.normalizer
            self.imputer = self.detector.imputer
            self.perf = self.detector.performance

            y_all = df['label_numeric'].to_numpy()
            self.y_test = y_all[test_idx]
            self.test_idx = test_idx
            self.y_pred = test_ens
            self.test_probs = test_probs
            self.cm = confusion_matrix(self.y_test, self.y_pred)

            if self._check():
                return False
            self._progress(75, "Scoring individual models …")
            X_test = self.detector.transform_preprocess(df[self.features].iloc[test_idx])
            self.X_test_transformed = X_test  # stored for weighted confidence
            self.model_scores = {}
            for name, model in self.detector.models.items():
                pred = model.predict(X_test)
                p, r, f1, _ = precision_recall_fscore_support(
                    self.y_test, pred, average='binary', zero_division=0)
                self.model_scores[name.replace('_', ' ').title()] = {
                    'accuracy': accuracy_score(self.y_test, pred),
                    'precision': p, 'recall': r, 'f1': f1,
                }
            self.model_scores['Ensemble (Voting)'] = {
                'accuracy': self.perf['test_accuracy'],
                'precision': self.perf['precision'],
                'recall': self.perf['recall'],
                'f1': self.perf['f1_score'],
            }

            if self._check():
                return False
            self._progress(88, "Predicting full dataset & saving outputs …")
            full_out = self.detector.predict_all_and_save(df)

            self._progress(95, "Analyzing attack patterns …")
            full_pred = full_out['prediction_numeric'].to_numpy()
            self.attack_counts = detect_attack_types(full_out, full_pred, full_out)
            total_spoofed = sum([v['count'] for v in self.attack_counts.values()])
            if total_spoofed > 0:
                for attack_type, data in self.attack_counts.items():
                    data['percentage'] = (data['count'] / total_spoofed) * 100
            else:
                for attack_type in self.attack_counts:
                    self.attack_counts[attack_type]['percentage'] = 0

            self._log("")
            self._log("─── Ensemble Results ───────────────────────")
            self._log(f"  Accuracy   {self.perf['test_accuracy']*100:.2f}%")
            self._log(f"  Precision  {self.perf['precision']*100:.2f}%")
            self._log(f"  Recall     {self.perf['recall']*100:.2f}%")
            self._log(f"  F1 Score   {self.perf['f1_score']*100:.2f}%")
            self._log("")
            self._log("─── Attack Type Detection ─────────────────")
            for at, data in self.attack_counts.items():
                if data['count'] > 0:
                    self._log(f"  {at.replace('_', ' ').title()}: {data['count']} ({data['percentage']:.1f}%)")
                    self._log(f"    Indicators: {', '.join(data['indicators'])}")
            self._log("─" * 44)

            return True

        except Exception as e:
            self._log(f"ERROR  Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_blind_test(self, unlabeled_path: str, true_label_path: str = None):
        """
        Predicts on a CSV with no 'label' column using whichever model is
        currently active in this wrapper — trained this session OR loaded
        via Load PKL. Self-contained: builds features and predicts directly
        with self.ensemble/scaler/normalizer/imputer/features, instead of
        relying on the engine's run_blind_prediction() (which always reads
        a fixed model file from disk regardless of what's loaded here).
        """
        if self.ensemble is None or self.scaler is None or self.normalizer is None \
                or self.imputer is None or not self.features:
            raise RuntimeError("No active model. Run Training or Load PKL first.")

        self._log("─── Blind Test ─────────────────────────────")

        udf = pd.read_csv(unlabeled_path)
        udf = gps_engine_module.build_features_unlabeled(udf)

        X = udf[self.features].copy()
        X_norm = self.normalizer.transform(self.scaler.transform(self.imputer.transform(X)))

        pred = self.ensemble.predict(X_norm)
        prob = self.ensemble.predict_proba(X_norm)

        udf['prediction_numeric'] = pred
        udf['prediction'] = np.where(pred == 1, 'spoofed', 'normal')
        udf['confidence'] = [prob[i, p] * 100 for i, p in enumerate(pred)]

        out_dir = gps_engine_module.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        blind_path = os.path.join(out_dir, f"unlabeled_predictions_{ts}.csv")
        udf.to_csv(blind_path, index=False)

        result = {'udf': udf, 'blind_path': blind_path, 'acc': None, 'cm': None,
                  'perf': None, 'model_scores': None}

        if true_label_path:
            true_df = pd.read_csv(true_label_path)

            # البحث عن عمود الـ label بغض النظر عن الحالة
            label_col = None
            for col in true_df.columns:
                if col.strip().lower() == 'label':
                    label_col = col
                    break

            if label_col is None:
                self._log(f"WARNING  True-label CSV has no 'label' column "
                          f"(found: {list(true_df.columns)}) — skipping scoring.")
            else:
                true_df['label_numeric'] = true_df[label_col].astype(int)
                true_df = true_df.reset_index(drop=True)

                if len(true_df) != len(udf):
                    self._log(f"WARNING  Row count mismatch (true={len(true_df)}, "
                              f"predicted={len(udf)}) — skipping scoring.")
                else:
                    y_true = true_df['label_numeric'].reset_index(drop=True)
                    y_pred = udf['prediction_numeric'].reset_index(drop=True).astype(int)
                    result['acc'] = accuracy_score(y_true, y_pred)
                    result['cm'] = confusion_matrix(y_true, y_pred)
                    self._log(f"  Blind-test accuracy: {result['acc']*100:.2f}%")

                    # Ensemble Performance + Model Comparison, but on THIS
                    # blind dataset (not the original training test split).
                    ens_p, ens_r, ens_f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0)
                    result['perf'] = {
                        'test_accuracy': result['acc'],
                        'precision': ens_p, 'recall': ens_r, 'f1_score': ens_f1,
                    }

                    model_scores = {}
                    for name, model in self.detector.models.items():
                        try:
                            mp = model.predict(X_norm)
                        except Exception:
                            continue  # not fitted (e.g. a PKL saved before this feature existed)
                        p, r, f1, _ = precision_recall_fscore_support(
                            y_true, mp, average='binary', zero_division=0)
                        model_scores[name.replace('_', ' ').title()] = {
                            'accuracy': accuracy_score(y_true, mp),
                            'precision': p, 'recall': r, 'f1': f1,
                        }
                    model_scores['Ensemble (Voting)'] = {
                        'accuracy': result['acc'], 'precision': ens_p,
                        'recall': ens_r, 'f1': ens_f1,
                    }
                    result['model_scores'] = model_scores

        self._log("─" * 44)
        return result

    # Predictions flagged "spoofed" with confidence below this are treated
    # as possible zero-day / novel-pattern detections: the model is fairly
    # sure something is wrong, but not confident it matches a pattern it
    # was actually trained on.
    ZERO_DAY_CONFIDENCE_THRESHOLD = 0.90

    def generate_sample_table(self) -> list:
        if self.y_test is None or self.y_pred is None:
            return []

        # Force integer dtype so == 1 comparison never fails on float arrays
        yt = np.asarray(self.y_test, dtype=int)
        yp = np.asarray(self.y_pred, dtype=int)
        total = len(yt)

        # ── Weighted Confidence ─────────────────────────────────────────────
        # Instead of simple-average voting probabilities, we weight each
        # model's per-class probability by its F1 Score on the test set.
        # Models that performed better get more influence on the final
        # confidence score, making Zero-Day detection more meaningful.
        #
        #   weighted_conf = Σ(model_prob × model_f1) / Σ(model_f1)
        #
        # Falls back to the plain voting probability if individual model
        # scores or probabilities are unavailable (e.g. after PKL load).
        confidence = None
        if self.test_probs is not None and len(self.test_probs) == total:
            try:
                # Collect per-model probabilities and their F1 weights
                # model_scores keys use Title-case of the detector key
                _model_key_map = {
                    'random_forest':  'Random Forest',
                    'neural_network': 'Neural Network',
                    'extra_trees':    'Extra Trees',
                }
                weighted_probs = None
                total_weight   = 0.0

                if not hasattr(self, 'X_test_transformed') or self.X_test_transformed is None:
                    raise ValueError("X_test_transformed not available")

                X_test_tr = self.X_test_transformed

                for det_key, score_key in _model_key_map.items():
                    if (det_key in self.detector.models and
                            score_key in self.model_scores):
                        model    = self.detector.models[det_key]
                        w        = float(self.model_scores[score_key]['f1'])
                        m_probs  = model.predict_proba(X_test_tr)   # (n, 2)
                        if weighted_probs is None:
                            weighted_probs = m_probs * w
                        else:
                            weighted_probs += m_probs * w
                        total_weight += w

                if weighted_probs is not None and total_weight > 0:
                    norm_probs = weighted_probs / total_weight       # (n, 2)
                    confidence = norm_probs[np.arange(total), yp]
                    print(f"[WeightedConf] using F1-weighted confidence "
                          f"(weights sum={total_weight:.3f})")
                else:
                    raise ValueError("no weighted probs computed")

            except Exception as _e:
                # Fallback: plain voting probabilities
                print(f"[WeightedConf] fallback to voting probs — {_e}")
                confidence = self.test_probs[np.arange(total), yp]

        table_rows = []
        sizes = list(range(25, total + 1, 25))
        if total not in sizes:
            sizes.append(total)

        for n in sizes:
            if n > total:
                break
            yt_n = yt[:n]
            yp_n = yp[:n]

            acc = accuracy_score(yt_n, yp_n)
            f1 = f1_score(yt_n, yp_n, average='binary', zero_division=0)

            cm_n = confusion_matrix(yt_n, yp_n, labels=[0, 1])
            if cm_n.shape == (2, 2):
                tn, fp, fn, tp = cm_n.ravel()
            else:
                tn = fp = fn = tp = 0

            if confidence is not None:
                conf_n = confidence[:n]
                spoofed_mask = (yp_n == 1)
                low_conf_mask = (conf_n < self.ZERO_DAY_CONFIDENCE_THRESHOLD)
                zero_day = int(np.sum(spoofed_mask & low_conf_mask))
                if n == sizes[-1]:
                    print(f"[ZeroDay] spoofed={int(spoofed_mask.sum())} "
                          f"low_conf={int(low_conf_mask.sum())} "
                          f"zero_day={zero_day} "
                          f"conf=[{conf_n.min():.3f},{conf_n.max():.3f}] "
                          f"threshold={self.ZERO_DAY_CONFIDENCE_THRESHOLD}")
            else:
                zero_day = 0

            table_rows.append({
                'Samples': n,
                'Accuracy': round(acc, 4),
                'F1_Score': round(f1, 4),
                'Normal_Correct': int(tn),
                'Known_Correct': int(tp),
                'ZeroDay_Predicted': zero_day,
                'FP': int(fp),
                'FN': int(fn),
            })

        return table_rows

    def get_attack_report(self):
        if not self.attack_counts:
            return "No attack data available."
        
        report = "╔══════════════════════════════════════════════════════════════╗\n"
        report += "║                    ATTACK TYPE DETECTION REPORT            ║\n"
        report += "╚══════════════════════════════════════════════════════════════╝\n\n"
        
        total = sum([v['count'] for v in self.attack_counts.values()])
        if total == 0:
            report += "✅ No spoofing attacks detected in the dataset.\n"
            return report
        
        report += f"📊 Total Spoofed Points Detected: {total}\n\n"
        report += "┌─────────────────────────┬──────────┬────────────┬────────────────────────────┐\n"
        report += "│ Attack Type             │ Count    │ Percentage │ Key Indicators              │\n"
        report += "├─────────────────────────┼──────────┼────────────┼────────────────────────────┤\n"
        
        for at, data in self.attack_counts.items():
            if data['count'] > 0:
                name = at.replace('_', ' ').title()
                count = data['count']
                pct = data.get('percentage', 0)
                indicators = ', '.join(data['indicators'][:3])
                if len(data['indicators']) > 3:
                    indicators += f", +{len(data['indicators'])-3} more"
                report += f"│ {name:<23} │ {count:>6} │ {pct:>8.1f}% │ {indicators:<26} │\n"
        
        report += "└─────────────────────────┴──────────┴────────────┴────────────────────────────┘\n\n"
        
        report += "🔍 Detection Indicators Legend:\n"
        report += "  • speed_residual     : Difference between reported and calculated speed\n"
        report += "  • low_velocity       : GPS reports near-zero speed for extended periods\n"
        report += "  • course_bearing_diff: Discrepancy between course and movement bearing\n"
        report += "  • velocity_vs_coord  : Reported speed doesn't match coordinate movement\n"
        report += "  • hdop_sat_diff      : Unusual changes in HDOP and satellite count\n"
        report += "  • sat_ratio_hdop     : Low satellite lock ratio with high HDOP\n"
        
        return report

    def run(self, path: str) -> bool:
        df = self.load(path)
        if df is None:
            return False
        ok = self.train(df)
        if ok and not self._check():
            self._progress(100, "Complete")
        return ok

# ─── StatCard Widget ──────────────────────────────────────────────────────────

class StatCard(ctk.CTkFrame):
    def __init__(self, master, title: str, color: str, **kw):
        super().__init__(master, fg_color=COLORS["card"], corner_radius=12,
                         border_width=1, border_color=COLORS["border"], **kw)
        ctk.CTkFrame(self, fg_color=color, height=3, corner_radius=3).pack(fill="x")
        self._title_lbl = ctk.CTkLabel(
            self, text=title, font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLORS["muted"])
        self._title_lbl.pack(anchor="w", padx=14, pady=(10, 0))
        self._val = ctk.CTkLabel(self, text="—",
                                 font=ctk.CTkFont(size=30, weight="bold"),
                                 text_color=COLORS["text"])
        self._val.pack(anchor="w", padx=14, pady=(2, 12))

    def set(self, v: str):
        self._val.configure(text=v)

# ─── Main Application ──────────────────────────────────────────────────────────

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GPS Spoofing Detector  ·  Light Edition")
        self.geometry("1380x860")
        self.minsize(1140, 700)
        self.configure(fg_color=COLORS["bg"])

        self._file = None
        self._unlabeled_file = None
        self._truelabel_file = None
        self._running = False
        self._blind_running = False
        self._stop_req = False
        self._detector = None

        self._style = ttk.Style()
        self._apply_treeview_style()

        self._build()
        sys.stdout = Redirector(self._console_box, self._dash_log)
        self._welcome()

    def _apply_treeview_style(self):
        s = self._style
        s.theme_use("clam")
        s.configure("GPS.Treeview",
                    background=COLORS["tbl_odd"],
                    foreground=COLORS["tbl_fg"],
                    fieldbackground=COLORS["tbl_odd"],
                    rowheight=42,
                    font=("Segoe UI", 12),
                    borderwidth=0,
                    relief="flat")
        s.configure("GPS.Treeview.Heading",
                    background=COLORS["tbl_head"],
                    foreground=COLORS["text"],
                    font=("Segoe UI", 13, "bold"),
                    relief="flat",
                    borderwidth=0)
        s.map("GPS.Treeview",
              background=[("selected", COLORS["accent"])],
              foreground=[("selected", "#ffffff")])

    def _welcome(self):
        now = datetime.now().strftime("%Y-%m-%d  %H:%M")
        print("GPS Spoofing Detector  —  Light Edition")
        print(f"Session: {now}")
        print("─" * 50)
        print("Powered by PROJECT_2.py")
        print("Features: 31 advanced human-movement GPS features")
        print("Attack Detection: 6 types")
        print()
        print("1. Open the Training tab, click Browse and choose your CSV file.")
        print("2. Click  Run Analysis  to start.")
        print()

    def _build(self):
        self._hdr = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=64)
        self._hdr.pack(side="top", fill="x")
        self._hdr.pack_propagate(False)

        ctk.CTkLabel(self._hdr, text="GPS Spoofing Detector - Light Edition",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=24, pady=12)

        self._hdr_status = ctk.CTkLabel(self._hdr, text="● Idle",
                                        font=ctk.CTkFont(size=13),
                                        text_color=COLORS["green"])
        self._hdr_status.pack(side="right", padx=16)

        self._hdr_sep = ctk.CTkFrame(self, fg_color=COLORS["border"], height=1)
        self._hdr_sep.pack(side="top", fill="x")

        self._bar = ctk.CTkFrame(self, fg_color=COLORS["surface"], corner_radius=0, height=32)
        self._bar.pack(side="bottom", fill="x")
        self._bar.pack_propagate(False)

        self._bar_sep = ctk.CTkFrame(self, fg_color=COLORS["border"], height=1)
        self._bar_sep.pack(side="bottom", fill="x")

        self._bar_lbl = ctk.CTkLabel(self._bar, text="Ready",
                                     font=ctk.CTkFont(size=12),
                                     text_color=COLORS["muted"])
        self._bar_lbl.pack(side="left", padx=18)

        ctk.CTkLabel(self._bar, text="Powered by PROJECT_2.py",
                     font=ctk.CTkFont(size=12),
                     text_color=COLORS["dim"]).pack(side="right", padx=18)

        self._body = ctk.CTkFrame(self, fg_color="transparent")
        self._body.pack(side="top", fill="both", expand=True)

        self._build_sidebar(self._body)
        self._sep_line = ctk.CTkFrame(self._body, fg_color=COLORS["border"], width=1)
        self._sep_line.pack(side="left", fill="y")

        self._build_content(self._body)

    def _build_sidebar(self, body):
        self._sb = ctk.CTkFrame(body, fg_color=COLORS["surface"], corner_radius=0, width=NAV_W)
        self._sb.pack(side="left", fill="y")
        self._sb.pack_propagate(False)

        ctk.CTkLabel(self._sb, text="NAVIGATION",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=16, pady=(24, 8))

        self._nav_btns = {}
        self._pages = {}

        nav_items = [
            ("training", "Training", "⚙"),
            ("blind_test", "Blind Test", "◐"),
            ("results", "Results", "◈"),
            ("attack_report", "Attack Report", "⚠"),
            ("console", "Console", "≡"),
            ("about", "About", "ℹ"),
        ]

        for key, label, icon in nav_items:
            btn = ctk.CTkButton(self._sb,
                                text=f"  {icon}   {label}",
                                command=lambda k=key: self._nav(k),
                                fg_color="transparent",
                                hover_color=COLORS["card"],
                                text_color=COLORS["text"],
                                anchor="w",
                                height=42,
                                corner_radius=8,
                                font=ctk.CTkFont(size=14))
            btn.pack(fill="x", padx=10, pady=3)
            self._nav_btns[key] = btn

    def _build_content(self, body):
        ct = ctk.CTkFrame(body, fg_color="transparent")
        ct.pack(side="left", fill="both", expand=True)

        self._pages["training"] = self._make_training(ct)
        self._pages["blind_test"] = self._make_blind_test(ct)
        self._pages["results"] = self._make_results(ct)
        self._pages["attack_report"] = self._make_attack_report(ct)
        self._pages["console"] = self._make_console(ct)
        self._pages["about"] = self._make_about(ct)

        self._nav("training")

    def _make_attack_report(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Attack Type Detection Report",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 8))

        self._attack_scroll = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg"])
        self._attack_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self._attack_scroll.columnconfigure(0, weight=1)

        ctk.CTkLabel(self._attack_scroll, text="No attack data yet.\nRun an analysis first.",
                     font=ctk.CTkFont(size=14), text_color=COLORS["muted"],
                     justify="center").grid(row=0, column=0, pady=60)

        return page

    def _make_blind_test(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Blind Test — Unseen Data Evaluation",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 4))

        ctk.CTkLabel(page,
                     text="Predicts on a CSV with no label column, using the model trained "
                          "in 'Run Analysis'. Optionally score it against a true-label CSV.",
                     font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                     wraplength=900, justify="left").pack(anchor="w", padx=20, pady=(0, 12))

        files_card = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=12,
                                  border_width=1, border_color=COLORS["border"])
        files_card.pack(fill="x", padx=20, pady=(0, 12))

        row1 = ctk.CTkFrame(files_card, fg_color="transparent")
        row1.pack(fill="x", padx=18, pady=(16, 8))
        ctk.CTkLabel(row1, text="Prediction Dataset (required)", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"], width=220, anchor="w").pack(side="left")
        self._unlabeled_lbl = ctk.CTkLabel(row1, text="No file selected",
                                           font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                                           anchor="w")
        self._unlabeled_lbl.pack(side="left", fill="x", expand=True, padx=(8, 8))
        ctk.CTkButton(row1, text="Browse …", command=self._browse_unlabeled,
                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                      text_color=COLORS["text"], width=100, height=32,
                      corner_radius=6, font=ctk.CTkFont(size=12)).pack(side="right")

        row2 = ctk.CTkFrame(files_card, fg_color="transparent")
        row2.pack(fill="x", padx=18, pady=(0, 16))
        ctk.CTkLabel(row2, text="Evaluation Dataset (optional)", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"], width=220, anchor="w").pack(side="left")
        self._truelabel_lbl = ctk.CTkLabel(row2, text="Not set — predictions only, no scoring",
                                           font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                                           anchor="w")
        self._truelabel_lbl.pack(side="left", fill="x", expand=True, padx=(8, 8))
        ctk.CTkButton(row2, text="Browse …", command=self._browse_truelabel,
                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                      text_color=COLORS["text"], width=100, height=32,
                      corner_radius=6, font=ctk.CTkFont(size=12)).pack(side="right")

        self._blind_btn = ctk.CTkButton(files_card, text="▶  Run Blind Test",
                                        command=self._run_blind_test,
                                        fg_color=COLORS["purple"], hover_color="#6d28d9",
                                        text_color="white", height=38, corner_radius=8,
                                        font=ctk.CTkFont(size=13, weight="bold"),
                                        state="disabled")
        self._blind_btn.pack(fill="x", padx=18, pady=(0, 16))

        self._blind_scroll = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg"])
        self._blind_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self._blind_scroll.columnconfigure(0, weight=1)

        summary_card = ctk.CTkFrame(self._blind_scroll, fg_color=COLORS["card"], corner_radius=12,
                                    border_width=1, border_color=COLORS["border"])
        summary_card.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(summary_card, text="Summary", font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(14, 8))

        self._blind_summary_box = ctk.CTkTextbox(summary_card, font=ctk.CTkFont(size=13, family="Consolas"),
                                                  fg_color=COLORS["bg"], text_color=COLORS["text"],
                                                  border_width=1, border_color=COLORS["border"],
                                                  corner_radius=8, height=160, wrap="word")
        self._blind_summary_box.pack(fill="x", padx=16, pady=(0, 16))
        self._blind_summary_box.insert("1.0", "Run an analysis first (tab: Dashboard), then "
                                              "pick an unlabeled CSV above and click Run Blind Test.")
        self._blind_summary_box.configure(state="disabled")

        self._blind_chart_frame = ctk.CTkFrame(self._blind_scroll, fg_color="transparent")
        self._blind_chart_frame.grid(row=1, column=0, sticky="ew")

        return page

    def _show_page(self, key: str):
        for k, frm in self._pages.items():
            if k == key:
                frm.pack(fill="both", expand=True)
            else:
                frm.pack_forget()

    def _nav(self, key: str):
        for k, btn in self._nav_btns.items():
            if k == key:
                btn.configure(fg_color=COLORS["card"], text_color=COLORS["text"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["text"])
        self._show_page(key)

    def _make_training(self, parent):
        """Dedicated training page — dataset selection + live progress, metrics & log."""
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Training Dataset", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 4))

        ctk.CTkLabel(page,
                     text="Select a labeled CSV file or load a pre-trained PKL below, then "
                          "use the action buttons to run, stop, save or clear the console.",
                     font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                     wraplength=900, justify="left").pack(anchor="w", padx=20, pady=(0, 8))

        # Scrollable body — keeps Training Log always visible
        page_body = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg"])
        page_body.pack(fill="both", expand=True)

        file_card = ctk.CTkFrame(page_body, fg_color=COLORS["surface"], corner_radius=12,
                                 border_width=1, border_color=COLORS["border"])
        file_card.pack(fill="x", padx=20, pady=(0, 16))

        row = ctk.CTkFrame(file_card, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(18, 8))

        ctk.CTkLabel(row, text="Dataset (CSV)", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"], width=160, anchor="w").pack(side="left")

        self._file_lbl = ctk.CTkLabel(row, text="No file loaded",
                                      font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                                      anchor="w")
        self._file_lbl.pack(side="left", fill="x", expand=True, padx=(8, 8))

        self._browse_btn = ctk.CTkButton(row, text="Browse …", command=self._browse,
                                         fg_color=COLORS["card"], hover_color=COLORS["border"],
                                         text_color=COLORS["text"], width=110, height=34,
                                         corner_radius=8, font=ctk.CTkFont(size=13))
        self._browse_btn.pack(side="right")

        # ── OR load a pre-trained PKL ──────────────────────────────
        divider_row = ctk.CTkFrame(file_card, fg_color="transparent")
        divider_row.pack(fill="x", padx=18, pady=(0, 4))
        ctk.CTkFrame(divider_row, fg_color=COLORS["border"], height=1).pack(fill="x")

        pkl_row = ctk.CTkFrame(file_card, fg_color="transparent")
        pkl_row.pack(fill="x", padx=18, pady=(4, 18))

        ctk.CTkLabel(pkl_row, text="Pre-trained Model (PKL)", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=COLORS["text"], width=160, anchor="w").pack(side="left")

        self._pkl_lbl = ctk.CTkLabel(pkl_row, text="No PKL loaded — train from CSV instead",
                                     font=ctk.CTkFont(size=12), text_color=COLORS["muted"],
                                     anchor="w")
        self._pkl_lbl.pack(side="left", fill="x", expand=True, padx=(8, 8))

        self._pkl_btn = ctk.CTkButton(pkl_row, text="Load PKL …", command=self._load_pkl,
                                      fg_color=COLORS["card"], hover_color=COLORS["border"],
                                      text_color=COLORS["purple"], width=110, height=34,
                                      corner_radius=8, font=ctk.CTkFont(size=13),
                                      border_width=1, border_color=COLORS["purple"])
        self._pkl_btn.pack(side="right")

        # ── Actions (moved here from the sidebar) ───────────────────
        act_card = ctk.CTkFrame(page_body, fg_color=COLORS["surface"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        act_card.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(act_card, text="Actions", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(14, 8))

        self._run_btn = ctk.CTkButton(act_card, text="▶  Run Analysis", command=self._run,
                                      fg_color=COLORS["accent"], hover_color="#1d4ed8",
                                      text_color="white", height=40, corner_radius=8,
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self._run_btn.pack(fill="x", padx=18, pady=(0, 8))

        act_row = ctk.CTkFrame(act_card, fg_color="transparent")
        act_row.pack(fill="x", padx=18, pady=(0, 18))

        self._stop_btn = ctk.CTkButton(act_row, text="■  Stop", command=self._stop,
                                       fg_color=COLORS["card"], hover_color=COLORS["border"],
                                       text_color=COLORS["red"], height=36, corner_radius=8,
                                       font=ctk.CTkFont(size=13), state="disabled")
        self._stop_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._save_btn = ctk.CTkButton(act_row, text="↓  Save Model", command=self._save,
                                       fg_color=COLORS["card"], hover_color=COLORS["border"],
                                       text_color=COLORS["yellow"], height=36, corner_radius=8,
                                       font=ctk.CTkFont(size=13), state="disabled")
        self._save_btn.pack(side="left", fill="x", expand=True, padx=6)

        self._clear_btn = ctk.CTkButton(act_row, text="⌫  Clear Console", command=self._clear,
                                        fg_color=COLORS["card"], hover_color=COLORS["border"],
                                        text_color=COLORS["text"], height=36, corner_radius=8,
                                        font=ctk.CTkFont(size=13))
        self._clear_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        prog_card = ctk.CTkFrame(page_body, fg_color=COLORS["surface"], corner_radius=12,
                                 border_width=1, border_color=COLORS["border"])
        prog_card.pack(fill="x", padx=20, pady=(0, 16))

        self._prog_bar = ctk.CTkProgressBar(prog_card, height=8, fg_color=COLORS["card"],
                                            progress_color=COLORS["accent"], corner_radius=3)
        self._prog_bar.pack(fill="x", padx=18, pady=(16, 6))
        self._prog_bar.set(0)

        self._prog_lbl = ctk.CTkLabel(prog_card, text="Ready", font=ctk.CTkFont(size=12),
                                      text_color=COLORS["muted"])
        self._prog_lbl.pack(anchor="w", padx=18, pady=(0, 14))

        grid = tk.Frame(page_body, bg=COLORS["bg"])
        grid.pack(fill="x", padx=14, pady=(0, 12))
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        specs = [
            ("ACCURACY", COLORS["accent"]),
            ("PRECISION", COLORS["green"]),
            ("RECALL", COLORS["yellow"]),
            ("F1 SCORE", COLORS["purple"]),
        ]
        self._cards = {}
        for i, (label, color) in enumerate(specs):
            card = StatCard(grid, label, color)
            card.grid(row=i//2, column=i%2, sticky="ew", padx=8, pady=8)
            self._cards[label] = card

        log_wrap = ctk.CTkFrame(page_body, fg_color=COLORS["surface"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        log_wrap.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(log_wrap, text="Training Log", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(12, 6))

        self._dash_log = ctk.CTkTextbox(log_wrap, font=ctk.CTkFont(size=13, family="Consolas"),
                                        fg_color=COLORS["bg"], text_color=COLORS["text"],
                                        border_width=1, border_color=COLORS["border"],
                                        corner_radius=8, wrap="word", height=340)
        self._dash_log.pack(fill="x", padx=12, pady=(0, 12))

        return page

    def _make_console(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Console Output", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 8))

        self._console_box = ctk.CTkTextbox(page, font=ctk.CTkFont(size=14, family="Consolas"),
                                           fg_color=COLORS["bg"], text_color=COLORS["text"],
                                           border_width=1, border_color=COLORS["border"],
                                           corner_radius=10, wrap="word")
        self._console_box.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        return page

    def _make_results(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Analysis Results", font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 8))

        self._results_scroll = ctk.CTkScrollableFrame(page, fg_color=COLORS["bg"])
        self._results_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self._results_scroll.columnconfigure(0, weight=1)

        ctk.CTkLabel(self._results_scroll, text="No results yet.\nRun an analysis first.",
                     font=ctk.CTkFont(size=14), text_color=COLORS["muted"],
                     justify="center").grid(row=0, column=0, pady=60)

        return page

    def _make_about(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        box = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=14,
                           border_width=1, border_color=COLORS["border"])
        box.pack(expand=True, pady=70, padx=90)

        ctk.CTkLabel(box, text="GPS Spoofing Detector", font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=COLORS["text"]).pack(pady=(32, 4))
        ctk.CTkLabel(box, text="Light Edition — Powered by PROJECT_2.py", 
                     font=ctk.CTkFont(size=14),
                     text_color=COLORS["accent"]).pack()

        ctk.CTkFrame(box, fg_color=COLORS["border"], height=1).pack(fill="x", padx=28, pady=20)

        items = [
            ("Engine", "HumanGPSDetector from PROJECT_2.py"),
            ("Ensemble", "Random Forest + Neural Network + Extra Trees"),
            ("Voting", "Soft probability voting"),
            ("Validation", "5-fold Stratified K-Fold cross-validation"),
            ("Features", "31 advanced human-movement GPS features"),
            ("Attack Types", "3 types: Freeze Attack, Replay Attack, Geofence Evasion"),
        ]
        for k, v in items:
            row = ctk.CTkFrame(box, fg_color="transparent")
            row.pack(fill="x", padx=28, pady=6)
            ctk.CTkLabel(row, text=f"{k}:", width=130, font=ctk.CTkFont(size=13),
                         text_color=COLORS["muted"], anchor="e").pack(side="left")
            ctk.CTkLabel(row, text=v, font=ctk.CTkFont(size=13),
                         text_color=COLORS["text"]).pack(side="left", padx=12)

        ctk.CTkFrame(box, fg_color="transparent", height=24).pack()
        return page

    # ── File ──────────────────────────────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Open GPS Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self._file = path
        name = os.path.basename(path)
        self._file_lbl.configure(text=name, text_color=COLORS["text"])
        self._set_status(f"Loaded: {name}", COLORS["green"])
        print(f"[OK]  Dataset: {path}\n")

    def _browse_unlabeled(self):
        path = filedialog.askopenfilename(
            title="Open Unlabeled GPS CSV (no 'label' column)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self._unlabeled_file = path
        self._unlabeled_lbl.configure(text=os.path.basename(path), text_color=COLORS["text"])
        self._update_blind_btn_state()

    def _browse_truelabel(self):
        path = filedialog.askopenfilename(
            title="Open True-Label GPS CSV (has 'label' column)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self._truelabel_file = path
        self._truelabel_lbl.configure(text=os.path.basename(path), text_color=COLORS["text"])

    def _load_pkl(self):
        path = filedialog.askopenfilename(
            title="Load Pre-trained Model (PKL)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            data = joblib.load(path)
            # بناء GPSDetectorWrapper وحشو القيم من الـ PKL
            self._detector = GPSDetectorWrapper(
                on_progress=self._on_prog,
                on_status=self._on_stat,
                is_stopped=lambda: self._stop_req)
            self._detector.ensemble    = data.get('ensemble')
            self._detector.scaler      = data.get('scaler')
            self._detector.normalizer  = data.get('normalizer')
            self._detector.imputer     = data.get('imputer')
            self._detector.features    = data.get('features', [])
            self._detector.perf        = data.get('performance',
                                            dict(test_accuracy=0, precision=0, recall=0, f1_score=0))
            self._detector.attack_counts = data.get('attack_counts')
            if data.get('models'):
                self._detector.detector.models = data['models']

            # ── نتائج التدريب ─────────────────────────────────────
            self._detector.y_test             = data.get('y_test')
            self._detector.y_pred             = data.get('y_pred')
            self._detector.test_probs         = data.get('test_probs')
            self._detector.model_scores       = data.get('model_scores', {})
            self._detector.cm                 = data.get('cm')
            self._detector.X_test_transformed = data.get('X_test_transformed')
            self._detector.test_idx           = data.get('test_idx')
            if data.get('dataset_info'):
                self._detector.dataset_info   = data['dataset_info']

            # تحديث الـ engine الداخلي بحيث run_blind_test يشتغل
            self._detector.detector.ensemble_model = self._detector.ensemble
            self._detector.detector.scaler         = self._detector.scaler
            self._detector.detector.normalizer      = self._detector.normalizer
            self._detector.detector.imputer         = self._detector.imputer
            self._detector.detector.feature_names   = self._detector.features

            name = os.path.basename(path)
            self._pkl_lbl.configure(text=name, text_color=COLORS["purple"])
            self._set_status(f"PKL loaded: {name}", COLORS["purple"])
            print(f"[OK]  Pre-trained model loaded: {path}\n")

            # تفعيل Blind Test مباشرة
            self._update_blind_btn_state()
            self._save_btn.configure(state="normal")

            # لو الـ PKL فيه نتائج محفوظة → عرضها مباشرة
            if (self._detector.y_test is not None and
                    self._detector.y_pred is not None):
                print("\n[PKL] Results restored from saved model.\n")
                # تحديث الـ stat cards
                p = self._detector.perf
                self._cards["ACCURACY"].set(f"{p['test_accuracy']*100:.1f}%")
                self._cards["PRECISION"].set(f"{p['precision']*100:.1f}%")
                self._cards["RECALL"].set(f"{p['recall']*100:.1f}%")
                self._cards["F1 SCORE"].set(f"{p['f1_score']*100:.1f}%")
                self._fill_results()
            else:
                # PKL قديم بدون نتائج → لازم CSV
                print("\n[PKL] Model loaded (no saved results).\n"
                      "      ► Browse a labeled CSV then click Run Analysis to see Results.\n"
                      "      ► Or go to Blind Test for unlabeled data.\n")

        except Exception as exc:
            messagebox.showerror("Load PKL Failed", str(exc))

    def _update_blind_btn_state(self):
        ready = bool(self._unlabeled_file) and self._detector is not None \
                and getattr(self._detector, "ensemble", None) is not None
        self._blind_btn.configure(state="normal" if ready else "disabled")

    # ── Run ───────────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        if not self._file:
            messagebox.showwarning("No Dataset", "Please browse and select a CSV file first.")
            return
        self._running = True
        self._stop_req = False
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._browse_btn.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        self._prog_bar.set(0)
        self._set_status("Running analysis …", COLORS["yellow"])
        self._hdr_status.configure(text="● Running", text_color=COLORS["yellow"])
        self._clear()
        threading.Thread(target=self._thread, daemon=True).start()

    def _thread(self):
        try:
            self._detector = GPSDetectorWrapper(
                on_progress=self._on_prog,
                on_status=self._on_stat,
                is_stopped=lambda: self._stop_req)
            ok = self._detector.run(self._file)
            if ok and not self._stop_req:
                self.after(0, self._done)
            elif self._stop_req:
                self.after(0, lambda: self._set_status("Stopped", COLORS["red"]))
        except Exception as exc:
            import traceback
            error_msg = f"Analysis failed:\n{exc}\n\n{traceback.format_exc()}"
            self.after(0, lambda e=error_msg: messagebox.showerror("Error", e))
        finally:
            self._running = False
            self.after(0, self._restore)

    def _done(self):
        p = self._detector.perf
        self._cards["ACCURACY"].set(f"{p['test_accuracy']*100:.1f}%")
        self._cards["PRECISION"].set(f"{p['precision']*100:.1f}%")
        self._cards["RECALL"].set(f"{p['recall']*100:.1f}%")
        self._cards["F1 SCORE"].set(f"{p['f1_score']*100:.1f}%")
        self._prog_bar.configure(progress_color=COLORS["green"])
        self._set_status("Analysis complete", COLORS["green"])
        self._hdr_status.configure(text="● Complete", text_color=COLORS["green"])
        self._save_btn.configure(state="normal")
        self._fill_results()
        self._fill_attack_report()
        self._update_blind_btn_state()

    def _fill_attack_report(self):
        if not self._detector or not self._detector.attack_counts:
            return

        for w in self._attack_scroll.winfo_children():
            w.destroy()

        counts = self._detector.attack_counts
        total = sum(v['count'] for v in counts.values())

        summary_wrap = ctk.CTkFrame(self._attack_scroll, fg_color=COLORS["card"], corner_radius=12,
                                    border_width=1, border_color=COLORS["border"])
        summary_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(summary_wrap, text=f"📊  Total Spoofed Points Detected: {total}",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=16)

        if total == 0:
            ctk.CTkLabel(self._attack_scroll, text="✅  No spoofing attacks detected in the dataset.",
                         font=ctk.CTkFont(size=14),
                         text_color=COLORS["green"]).grid(row=1, column=0, pady=20)
            return

        tbl_wrap = ctk.CTkFrame(self._attack_scroll, fg_color=COLORS["card"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        tbl_wrap.grid(row=1, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(tbl_wrap, text="Detected Attack Types", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

        rows = [(at, data) for at, data in counts.items() if data['count'] > 0]
        rows.sort(key=lambda x: x[1]['count'], reverse=True)

        cols = ("Attack Type", "Count", "Percentage", "Key Indicators")
        tree = ttk.Treeview(tbl_wrap, columns=cols, show="headings", style="GPS.Treeview",
                            height=max(1, min(8, len(rows))))

        widths = (180, 90, 110, 360)
        for col, w in zip(cols, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w,
                       anchor="w" if col in ("Attack Type", "Key Indicators") else "center")

        for i, (at, data) in enumerate(rows):
            tag = "even" if i % 2 == 0 else "odd"
            name = at.replace('_', ' ').title()
            pct = data.get('percentage', 0)
            indicators = ', '.join(data['indicators'])
            tree.insert("", "end", tags=(tag,), values=(name, data['count'], f"{pct:.1f}%", indicators))

        tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
        tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])

        vsb = ttk.Scrollbar(tbl_wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="x", expand=True, padx=(18, 0), pady=(0, 18))
        vsb.pack(side="left", fill="y", pady=(0, 18))

        legend_wrap = ctk.CTkFrame(self._attack_scroll, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
        legend_wrap.grid(row=2, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(legend_wrap, text="🔍  Detection Indicators Legend",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 8))

        legend_rows = [
            ("low_velocity", "GPS reports near-zero speed for extended periods (freeze attack)."),
            ("course_bearing_diff", "Discrepancy between reported course and bearing from movement (replay)."),
            ("hdop_sat_diff", "Unusual changes in HDOP and satellite count simultaneously (geofence evasion)."),
        ]
        for i, (key, desc) in enumerate(legend_rows):
            bg = COLORS["tbl_odd"] if i % 2 == 0 else COLORS["tbl_even"]
            row = ctk.CTkFrame(legend_wrap, fg_color=bg, corner_radius=6)
            row.pack(fill="x", padx=18, pady=3)
            ctk.CTkLabel(row, text=key, font=ctk.CTkFont(size=12, weight="bold", family="Consolas"),
                         text_color=COLORS["accent"], width=180,
                         anchor="w").pack(side="left", padx=(10, 6), pady=8)
            ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=12), text_color=COLORS["tbl_fg"],
                         anchor="w", justify="left",
                         wraplength=600).pack(side="left", fill="x", expand=True, pady=8)

        ctk.CTkFrame(legend_wrap, fg_color="transparent", height=8).pack()

    # ── Blind Test ───────────────────────────────────────────────────────────
    def _run_blind_test(self):
        if self._blind_running:
            return
        if not self._detector or not getattr(self._detector, "ensemble", None):
            messagebox.showwarning("No Trained Model",
                                   "Run an analysis first (Dashboard → Run Analysis) "
                                   "before running the blind test.")
            return
        if not self._unlabeled_file:
            messagebox.showwarning("No File", "Please select an unlabeled CSV file first.")
            return

        self._blind_running = True
        self._blind_btn.configure(state="disabled", text="Running …")
        self._set_status("Running blind test …", COLORS["yellow"])
        threading.Thread(target=self._blind_thread, daemon=True).start()

    def _blind_thread(self):
        try:
            result = self._detector.run_blind_test(self._unlabeled_file, self._truelabel_file)
            self.after(0, lambda r=result: self._fill_blind_test(r))
            self.after(0, lambda: self._set_status("Blind test complete", COLORS["green"]))
        except Exception as exc:
            import traceback
            error_msg = f"Blind test failed:\n{exc}\n\n{traceback.format_exc()}"
            self.after(0, lambda e=error_msg: messagebox.showerror("Error", e))
            self.after(0, lambda: self._set_status("Blind test failed", COLORS["red"]))
        finally:
            self._blind_running = False
            self.after(0, lambda: self._blind_btn.configure(state="normal", text="▶  Run Blind Test"))

    def _fill_blind_test(self, result):
        udf = result["udf"]
        counts = udf["prediction"].value_counts()

        lines = []
        lines.append("BLIND TEST RESULTS")
        lines.append("=" * 50)
        lines.append(f"Total rows predicted: {len(udf)}")
        for label, cnt in counts.items():
            lines.append(f"  {label.title()}: {cnt}")
        lines.append("")
        lines.append(f"Predictions saved to:\n  {result['blind_path']}")

        if result["acc"] is not None:
            lines.append("")
            lines.append(f"Accuracy vs. true labels: {result['acc']*100:.2f}%")
        else:
            lines.append("")
            lines.append("(No true-label file provided, or row counts didn't match "
                         "— accuracy not computed.)")

        self._blind_summary_box.configure(state="normal")
        self._blind_summary_box.delete("1.0", "end")
        self._blind_summary_box.insert("1.0", "\n".join(lines))
        self._blind_summary_box.configure(state="disabled")

        for w in self._blind_chart_frame.winfo_children():
            w.destroy()

        if result.get("perf") is not None:
            metrics_wrap = ctk.CTkFrame(self._blind_chart_frame, fg_color=COLORS["card"], corner_radius=12,
                                        border_width=1, border_color=COLORS["border"])
            metrics_wrap.pack(fill="x", pady=(0, 16))
            metrics_wrap.columnconfigure(1, weight=1)

            ctk.CTkLabel(metrics_wrap, text="Ensemble Performance (on this Blind Test)",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                          sticky="w", padx=18, pady=(16, 8))

            p = result["perf"]
            metrics_rows = [
                ("Accuracy", f"{p['test_accuracy']*100:.2f}%", COLORS["accent"]),
                ("Precision", f"{p['precision']*100:.2f}%", COLORS["green"]),
                ("Recall", f"{p['recall']*100:.2f}%", COLORS["yellow"]),
                ("F1 Score", f"{p['f1_score']*100:.2f}%", COLORS["purple"]),
            ]
            for i, (metric, value, color) in enumerate(metrics_rows):
                bg = COLORS["tbl_odd"] if i % 2 == 0 else COLORS["tbl_even"]
                rf = ctk.CTkFrame(metrics_wrap, fg_color=bg, corner_radius=0, height=44)
                rf.grid(row=i+1, column=0, columnspan=2, sticky="ew")
                rf.columnconfigure(1, weight=1)
                rf.grid_propagate(False)
                ctk.CTkLabel(rf, text=metric, font=ctk.CTkFont(size=13),
                             text_color=COLORS["muted"]).grid(row=0, column=0, sticky="w", padx=18)
                ctk.CTkLabel(rf, text=value, font=ctk.CTkFont(size=15, weight="bold"),
                             text_color=color).grid(row=0, column=1, sticky="e", padx=18)
            ctk.CTkFrame(metrics_wrap, fg_color="transparent", height=8).grid(row=5, column=0)

        if result.get("model_scores"):
            tbl_wrap = ctk.CTkFrame(self._blind_chart_frame, fg_color=COLORS["card"], corner_radius=12,
                                    border_width=1, border_color=COLORS["border"])
            tbl_wrap.pack(fill="x", pady=(0, 16))

            ctk.CTkLabel(tbl_wrap, text="Model Comparison (on this Blind Test)",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

            cols = ("Model", "Accuracy", "Precision", "Recall", "F1 Score")
            tree = ttk.Treeview(tbl_wrap, columns=cols, show="headings", style="GPS.Treeview",
                                height=min(5, len(result["model_scores"])))

            widths = [240, 120, 120, 110, 120]
            for col, w in zip(cols, widths):
                tree.heading(col, text=col)
                tree.column(col, width=w, anchor="center" if col != "Model" else "w")
            tree.column("Model", anchor="w")

            for i, (name, sc) in enumerate(result["model_scores"].items()):
                tag = "even" if i % 2 == 0 else "odd"
                tree.insert("", "end", tags=(tag,), values=(
                    name, f"{sc['accuracy']*100:.2f}%", f"{sc['precision']*100:.2f}%",
                    f"{sc['recall']*100:.2f}%", f"{sc['f1']*100:.2f}%",
                ))

            tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
            tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])

            vsb = ttk.Scrollbar(tbl_wrap, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side="left", fill="x", expand=True, padx=(18, 0), pady=(0, 18))
            vsb.pack(side="left", fill="y", pady=(0, 18))

        if result["cm"] is not None:
            cm_wrap = ctk.CTkFrame(self._blind_chart_frame, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            cm_wrap.pack(fill="x", pady=(0, 16))

            ctk.CTkLabel(cm_wrap, text="Confusion Matrix (Blind Test)",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor(COLORS["card"])
            ax.set_facecolor(COLORS["card"])

            cm_data = result["cm"]
            threshold = cm_data.max() / 2.0

            sns.heatmap(cm_data, annot=False, fmt='d', cmap='Blues', ax=ax,
                        linewidths=0.5, linecolor=COLORS["border"], cbar_kws={"shrink": 0.8})

            for (row_i, col_j), val in np.ndenumerate(cm_data):
                txt_color = "white" if val > threshold else COLORS["text"]
                ax.text(col_j + 0.5, row_i + 0.5, str(val), ha='center', va='center',
                        fontsize=18, fontweight='bold', color=txt_color)

            ax.set_xlabel("Predicted", color=COLORS["muted"], fontsize=13)
            ax.set_ylabel("Actual", color=COLORS["muted"], fontsize=13)
            ax.tick_params(colors=COLORS["muted"], labelsize=12)

            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_tick_params(color=COLORS["muted"], labelsize=11)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS["muted"])

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=cm_wrap)
            canvas.draw()
            canvas.get_tk_widget().pack(padx=18, pady=(0, 18))
            plt.close(fig)

    def _restore(self):
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._browse_btn.configure(state="normal")
        self._hdr_status.configure(text="● Idle", text_color=COLORS["muted"])

    def _stop(self):
        if self._running:
            self._stop_req = True
            self._set_status("Stopping …", COLORS["red"])
            print("[INFO]  Stop requested …")

    def _save(self):
        if not self._detector or not self._detector.ensemble:
            return
        path = filedialog.asksaveasfilename(
            title="Save Trained Model",
            defaultextension=".pkl",
            filetypes=[("Pickle", "*.pkl")])
        if not path:
            return
        try:
            joblib.dump({
                'ensemble':          self._detector.ensemble,
                'scaler':            self._detector.scaler,
                'normalizer':        self._detector.normalizer,
                'imputer':           self._detector.imputer,
                'features':          self._detector.features,
                'performance':       self._detector.perf,
                'attack_counts':     self._detector.attack_counts,
                'models':            self._detector.detector.models,
                # ── نتائج التدريب ─────────────────────────────────
                'y_test':            self._detector.y_test,
                'y_pred':            self._detector.y_pred,
                'test_probs':        self._detector.test_probs,
                'model_scores':      self._detector.model_scores,
                'cm':                self._detector.cm,
                'X_test_transformed': self._detector.X_test_transformed,
                'dataset_info':      getattr(self._detector, 'dataset_info', None),
                'test_idx':          getattr(self._detector, 'test_idx', None),
            }, path)
            print(f"[OK]  Model saved: {path}")
            messagebox.showinfo("Saved", f"Model saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _clear(self):
        for box in (self._console_box, self._dash_log):
            try:
                box.delete("1.0", "end")
            except Exception:
                pass

    def _open_sample_window(self):
        if not hasattr(self, "_sample_rows") or not self._sample_rows:
            return

        win = ctk.CTkToplevel(self)
        win.title("Sample Performance — Full View")
        win.geometry("1100x600")
        win.configure(fg_color=COLORS["bg"])
        win.grab_set()
        win.lift()
        win.focus_force()

        hdr = ctk.CTkFrame(win, fg_color=COLORS["surface"], corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="Model Performance on Increasing Test Sample Sizes",
                     font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left", padx=24, pady=14)

        ctk.CTkButton(hdr, text="✕  Close", command=win.destroy,
                      fg_color=COLORS["red"], hover_color="#b91c1c",
                      text_color="white", height=34, width=90,
                      corner_radius=6, font=ctk.CTkFont(size=13)).pack(side="right", padx=18)

        body = ctk.CTkFrame(win, fg_color=COLORS["card"], corner_radius=12,
                            border_width=1, border_color=COLORS["border"])
        body.pack(fill="both", expand=True, padx=20, pady=18)

        sp_cols = ("Samples", "Accuracy", "F1_Score", "Normal_Correct", "Known_Correct", "ZeroDay_Predicted", "FP", "FN")
        sp_widths = (100, 110, 110, 150, 150, 170, 80, 80)

        tree = ttk.Treeview(body, columns=sp_cols, show="headings", style="GPS.Treeview",
                            height=len(self._sample_rows))

        for col, w in zip(sp_cols, sp_widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center", stretch=True)

        for i, row in enumerate(self._sample_rows):
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end", tags=(tag,), values=(
                row['Samples'], f"{row['Accuracy']:.4f}", f"{row['F1_Score']:.4f}",
                row['Normal_Correct'], row['Known_Correct'], row['ZeroDay_Predicted'],
                row['FP'], row['FN'],
            ))

        tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
        tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])

        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(body, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew", padx=(16, 0), pady=(16, 0))
        vsb.grid(row=0, column=1, sticky="ns", pady=(16, 0))
        hsb.grid(row=1, column=0, sticky="ew", padx=(16, 0))

        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)

    def _on_prog(self, pct: int, msg: str = ""):
        self.after(0, lambda: self._prog_bar.set(pct / 100))
        if msg:
            self.after(0, lambda m=msg: self._prog_lbl.configure(text=m))
            self.after(0, lambda m=msg: self._bar_lbl.configure(text=m))

    def _on_stat(self, msg: str):
        self.after(0, lambda m=msg: self._set_status(m, COLORS["green"]))

    def _set_status(self, text: str, color: str = ""):
        self._bar_lbl.configure(text=text, text_color=color if color else COLORS["muted"])

    def _fill_results(self):
        for w in self._results_scroll.winfo_children():
            w.destroy()

        self._sample_rows = self._detector.generate_sample_table()

        # ── Dataset Summary ──────────────────────────────────────────
        if self._detector.df is not None:
            ds_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            ds_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 16))
            ds_wrap.columnconfigure(1, weight=1)

            ctk.CTkLabel(ds_wrap, text="Dataset Summary", font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2,
                                                          sticky="w", padx=18, pady=(16, 8))

            # استخدام الـ test set فقط (النتائج الفعلية للتنبؤ)
            yt = np.asarray(self._detector.y_test, dtype=int)
            yp = np.asarray(self._detector.y_pred, dtype=int)
            total = len(yt)
            normal_n  = int((yt == 0).sum())
            spoof_n   = int((yt == 1).sum())

            ds_rows = [
                ("Total Rows (Test Set)", f"{total:,}"),
                ("Normal",  f"{normal_n:,}  ({normal_n/total*100:.1f}%)"),
                ("Spoofed", f"{spoof_n:,}  ({spoof_n/total*100:.1f}%)"),
            ]
            df = self._detector.df
            if 'session_id' in df.columns and self._detector.test_idx is not None:
                test_sessions = df.iloc[self._detector.test_idx]['session_id'].nunique()
                ds_rows.append(("Sessions (Test)", f"{test_sessions:,}"))
            ds_rows.append(("Features Used", f"{len(self._detector.features)}"))

            # ZeroDay count from full sample table
            if self._sample_rows:
                last = self._sample_rows[-1]
                zd = last.get('ZeroDay_Predicted', 0)
                ds_rows.append(("Zero-Day Detections",
                                f"{zd}  (predicted spoofed, confidence < "
                                f"{int(self._detector.ZERO_DAY_CONFIDENCE_THRESHOLD*100)}%)"))

            for i, (label, value) in enumerate(ds_rows):
                bg = COLORS["tbl_odd"] if i % 2 == 0 else COLORS["tbl_even"]
                rf = ctk.CTkFrame(ds_wrap, fg_color=bg, corner_radius=0, height=40)
                rf.grid(row=i+1, column=0, columnspan=2, sticky="ew")
                rf.columnconfigure(1, weight=1)
                rf.grid_propagate(False)
                ctk.CTkLabel(rf, text=label, font=ctk.CTkFont(size=13),
                             text_color=COLORS["muted"]).grid(row=0, column=0, sticky="w", padx=18)
                ctk.CTkLabel(rf, text=value, font=ctk.CTkFont(size=14, weight="bold"),
                             text_color=COLORS["text"]).grid(row=0, column=1, sticky="e", padx=18)
            ctk.CTkFrame(ds_wrap, fg_color="transparent", height=8).grid(row=len(ds_rows)+1, column=0)



        # ── Misclassified Samples (the actual rows the model got wrong) ─
        y_test = self._detector.y_test
        y_pred = self._detector.y_pred

        err_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        err_wrap.grid(row=1, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(err_wrap, text="Misclassified Samples", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 4))
        ctk.CTkLabel(err_wrap,
                     text="The actual test rows the model got wrong — missed attacks are the "
                          "ones most worth investigating.",
                     font=ctk.CTkFont(size=11), text_color=COLORS["dim"]).pack(anchor="w", padx=18, pady=(0, 8))

        mis_mask = y_test != y_pred
        n_mis = int(mis_mask.sum())

        if n_mis == 0 or self._detector.df is None or self._detector.test_idx is None:
            ctk.CTkLabel(err_wrap, text="✅  No misclassified samples in the test set.",
                         font=ctk.CTkFont(size=13),
                         text_color=COLORS["green"]).pack(anchor="w", padx=18, pady=(0, 18))
        else:
            err_idx = self._detector.test_idx[mis_mask]
            err_df = self._detector.df.iloc[err_idx]
            err_yt = y_test[mis_mask]
            err_yp = y_pred[mis_mask]

            if self._detector.test_probs is not None:
                err_conf = self._detector.test_probs[mis_mask][np.arange(n_mis), err_yp] * 100
            else:
                err_conf = [None] * n_mis

            cols = ("Type", "Session ID", "GPS Time", "Velocity", "Course", "HDOP", "Confidence")
            tree = ttk.Treeview(err_wrap, columns=cols, show="headings", style="GPS.Treeview",
                                height=min(8, n_mis))
            widths = (120, 130, 100, 90, 90, 80, 100)
            for col, w in zip(cols, widths):
                tree.heading(col, text=col)
                tree.column(col, width=w, anchor="w" if col in ("Type", "Session ID") else "center")

            for i in range(n_mis):
                row = err_df.iloc[i]
                missed = err_yt[i] == 1  # actually spoofed, model said normal
                err_type = "Missed Attack" if missed else "False Alarm"
                tag = "missed" if missed else ("even" if i % 2 == 0 else "odd")
                conf_txt = f"{err_conf[i]:.1f}%" if err_conf[i] is not None else "—"
                tree.insert("", "end", tags=(tag,), values=(
                    err_type,
                    row.get('session_id', '—'),
                    row.get('gps_time', '—'),
                    row.get('velocity', '—'),
                    row.get('course', '—'),
                    row.get('hdop', '—'),
                    conf_txt,
                ))

            tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
            tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])
            tree.tag_configure("missed", background="#fde8e8", foreground=COLORS["tbl_fg"])

            vsb = ttk.Scrollbar(err_wrap, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=vsb.set)
            tree.pack(side="left", fill="x", expand=True, padx=(18, 0), pady=(0, 18))
            vsb.pack(side="left", fill="y", pady=(0, 18))

        if hasattr(self._detector.detector.models['random_forest'], 'feature_importances_'):
            fi_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            fi_wrap.grid(row=3, column=0, sticky="ew", pady=(0, 16))

            ctk.CTkLabel(fi_wrap, text="Top 15 Feature Importance",
                         font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

            rf = self._detector.detector.models['random_forest']
            importances = rf.feature_importances_
            features = self._detector.features

            indices = np.argsort(importances)[::-1][:15]
            top_features = [features[i] for i in indices]
            top_importances = [importances[i] for i in indices]

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor(COLORS["card"])
            ax.set_facecolor(COLORS["card"])

            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_features)))[::-1]
            ax.barh(top_features, top_importances, color=colors)
            ax.set_xlabel("Importance", color=COLORS["muted"], fontsize=12)
            ax.tick_params(colors=COLORS["muted"], labelsize=11)
            ax.set_xlim(0, max(top_importances) * 1.1)
            ax.invert_yaxis()

            for i, v in enumerate(top_importances):
                ax.text(v + 0.002, i, f"{v:.3f}", va='center', color=COLORS["muted"], fontsize=10)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=fi_wrap)
            canvas.draw()
            canvas.get_tk_widget().pack(padx=18, pady=(0, 18))
            plt.close(fig)


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    app = App()
    app.mainloop()