# GUI 1 — one-click analyzer dashboard
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
        "gradual_drag": {"count": 0, "indicators": []},
        "freeze": {"count": 0, "indicators": []},
        "replay": {"count": 0, "indicators": []},
        "fake_walking": {"count": 0, "indicators": []},
        "geofence_evasion": {"count": 0, "indicators": []},
        "signal_manipulation": {"count": 0, "indicators": []},
    }
    
    spoof_idx = np.where(predictions == 1)[0]
    if len(spoof_idx) == 0:
        return attack_types
    
    for idx in spoof_idx:
        if idx >= len(feature_df):
            continue
        row = feature_df.iloc[idx]
        
        speed_residual = row.get("speed_residual", 0) if not pd.isna(row.get("speed_residual", 0)) else 0
        coord_speed = row.get("coord_speed", 0) if not pd.isna(row.get("coord_speed", 0)) else 0
        velocity = row.get("velocity", 1) if not pd.isna(row.get("velocity", 1)) else 1
        distance_m = row.get("distance_m", 1) if not pd.isna(row.get("distance_m", 1)) else 1
        course_bearing_diff = row.get("course_bearing_diff", 0) if not pd.isna(row.get("course_bearing_diff", 0)) else 0
        velocity_diff = row.get("velocity_diff", 0) if not pd.isna(row.get("velocity_diff", 0)) else 0
        hdop_diff = row.get("hdop_diff", 0) if not pd.isna(row.get("hdop_diff", 0)) else 0
        sat_discrepancy = row.get("sat_discrepancy", 0) if not pd.isna(row.get("sat_discrepancy", 0)) else 0
        sat_ratio = row.get("sat_ratio", 1) if not pd.isna(row.get("sat_ratio", 1)) else 1
        hdop = row.get("hdop", 0) if not pd.isna(row.get("hdop", 0)) else 0
        
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
            self.y_pred = test_ens
            self.cm = confusion_matrix(self.y_test, self.y_pred)

            if self._check():
                return False
            self._progress(75, "Scoring individual models …")
            X_test = self.detector.transform_preprocess(df[self.features].iloc[test_idx])
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
        Runs the engine's blind-prediction workflow: loads the model that
        was just trained/saved (during Run Analysis) and predicts on a
        CSV that has no 'label' column. If a true-label CSV is also
        given, scores the blind predictions against it.
        Requires Run Analysis to have completed successfully at least
        once in this session (so a trained model exists on disk).
        """
        self._log("─── Blind Test ─────────────────────────────")

        gps_engine_module.UNLABELED_FILE = unlabeled_path
        udf, blind_path = gps_engine_module.run_blind_prediction()

        result = {'udf': udf, 'blind_path': blind_path, 'acc': None, 'cm': None}

        if true_label_path:
            true_df = pd.read_csv(true_label_path)
            true_df['label_numeric'] = true_df['label'].astype(int)
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

        self._log("─" * 44)
        return result

    def generate_sample_table(self) -> list:
        if self.y_test is None or self.y_pred is None:
            return []
        
        yt = self.y_test
        yp = self.y_pred
        total = len(yt)

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

            table_rows.append({
                'Samples': n,
                'Accuracy': round(acc, 4),
                'F1_Score': round(f1, 4),
                'Normal_Correct': int(tn),
                'Known_Correct': int(tp),
                'ZeroDay_Predicted': 0,
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
        print("1. Click  Browse  and choose your CSV file.")
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
            ("dashboard", "Dashboard", "⊞"),
            ("console", "Console", "≡"),
            ("results", "Results", "◈"),
            ("attack_report", "Attack Report", "⚠"),
            ("blind_test", "Blind Test", "◐"),
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

        ctk.CTkFrame(self._sb, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16, pady=18)

        ctk.CTkLabel(self._sb, text="DATASET",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=16, pady=(0, 8))

        self._file_lbl = ctk.CTkLabel(self._sb, text="No file loaded",
                                      font=ctk.CTkFont(size=12),
                                      text_color=COLORS["muted"],
                                      wraplength=NAV_W - 32, justify="left")
        self._file_lbl.pack(anchor="w", padx=16, pady=(0, 10))

        self._browse_btn = ctk.CTkButton(self._sb, text="Browse …", command=self._browse,
                                         fg_color=COLORS["card"], hover_color=COLORS["border"],
                                         text_color=COLORS["text"], height=38, corner_radius=8,
                                         font=ctk.CTkFont(size=13))
        self._browse_btn.pack(fill="x", padx=12, pady=(0, 6))

        ctk.CTkFrame(self._sb, fg_color=COLORS["border"], height=1).pack(fill="x", padx=16, pady=18)

        ctk.CTkLabel(self._sb, text="ACTIONS",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=COLORS["dim"]).pack(anchor="w", padx=16, pady=(0, 8))

        self._run_btn = ctk.CTkButton(self._sb, text="▶  Run Analysis", command=self._run,
                                      fg_color=COLORS["accent"], hover_color="#1d4ed8",
                                      text_color="white", height=40, corner_radius=8,
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self._run_btn.pack(fill="x", padx=12, pady=(0, 6))

        self._stop_btn = ctk.CTkButton(self._sb, text="■  Stop", command=self._stop,
                                       fg_color=COLORS["card"], hover_color=COLORS["border"],
                                       text_color=COLORS["red"], height=38, corner_radius=8,
                                       font=ctk.CTkFont(size=13), state="disabled")
        self._stop_btn.pack(fill="x", padx=12, pady=(0, 6))

        self._save_btn = ctk.CTkButton(self._sb, text="↓  Save Model", command=self._save,
                                       fg_color=COLORS["card"], hover_color=COLORS["border"],
                                       text_color=COLORS["yellow"], height=38, corner_radius=8,
                                       font=ctk.CTkFont(size=13), state="disabled")
        self._save_btn.pack(fill="x", padx=12, pady=(0, 6))

        self._clear_btn = ctk.CTkButton(self._sb, text="⌫  Clear Console", command=self._clear,
                                        fg_color=COLORS["card"], hover_color=COLORS["border"],
                                        text_color=COLORS["text"], height=38, corner_radius=8,
                                        font=ctk.CTkFont(size=13))
        self._clear_btn.pack(fill="x", padx=12, pady=(0, 8))

    def _build_content(self, body):
        ct = ctk.CTkFrame(body, fg_color="transparent")
        ct.pack(side="left", fill="both", expand=True)

        self._pages["dashboard"] = self._make_dashboard(ct)
        self._pages["console"] = self._make_console(ct)
        self._pages["results"] = self._make_results(ct)
        self._pages["attack_report"] = self._make_attack_report(ct)
        self._pages["blind_test"] = self._make_blind_test(ct)
        self._pages["about"] = self._make_about(ct)

        self._nav("dashboard")

    def _make_attack_report(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        ctk.CTkLabel(page, text="Attack Type Detection Report",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(16, 8))

        self._attack_report_box = ctk.CTkTextbox(page, font=ctk.CTkFont(size=13, family="Consolas"),
                                                  fg_color=COLORS["bg"], text_color=COLORS["text"],
                                                  border_width=1, border_color=COLORS["border"],
                                                  corner_radius=10, wrap="word")
        self._attack_report_box.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        ctk.CTkLabel(page, text="Detection Indicators Legend:",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=20, pady=(0, 8))

        legend_text = """
        ╔══════════════════════════════════════════════════════════════════════╗
        ║  INDICATOR                MEANING                                    ║
        ╠══════════════════════════════════════════════════════════════════════╣
        ║  speed_residual           : Difference between reported and          ║
        ║                            calculated speed from coordinates         ║
        ║  low_velocity             : GPS reports near-zero speed for          ║
        ║                            extended periods (freeze attack)          ║
        ║  course_bearing_diff      : Discrepancy between reported course      ║
        ║                            and bearing from movement                 ║
        ║  velocity_vs_coord        : Reported speed doesn't match             ║
        ║                            coordinate movement rate                  ║
        ║  hdop_sat_diff            : Unusual changes in HDOP and              ║
        ║                            satellite count simultaneously            ║
        ║  sat_ratio_hdop           : Low satellite lock ratio with            ║
        ║                            unusually high HDOP                       ║
        ╚══════════════════════════════════════════════════════════════════════╝
        """
        legend_box = ctk.CTkTextbox(page, font=ctk.CTkFont(size=12, family="Consolas"),
                                     fg_color=COLORS["card"], text_color=COLORS["text"],
                                     border_width=1, border_color=COLORS["border"],
                                     corner_radius=10, height=200, wrap="word")
        legend_box.pack(fill="x", padx=20, pady=(0, 16))
        legend_box.insert("1.0", legend_text)
        legend_box.configure(state="disabled")

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
        ctk.CTkLabel(row1, text="Unlabeled CSV (required)", font=ctk.CTkFont(size=13, weight="bold"),
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
        ctk.CTkLabel(row2, text="True-label CSV (optional)", font=ctk.CTkFont(size=13, weight="bold"),
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

    def _make_dashboard(self, parent):
        page = ctk.CTkFrame(parent, fg_color=COLORS["bg"])

        prog_card = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=12,
                                 border_width=1, border_color=COLORS["border"])
        prog_card.pack(fill="x", padx=20, pady=(20, 12))

        self._prog_bar = ctk.CTkProgressBar(prog_card, height=8, fg_color=COLORS["card"],
                                            progress_color=COLORS["accent"], corner_radius=3)
        self._prog_bar.pack(fill="x", padx=18, pady=(16, 6))
        self._prog_bar.set(0)

        self._prog_lbl = ctk.CTkLabel(prog_card, text="Ready", font=ctk.CTkFont(size=12),
                                      text_color=COLORS["muted"])
        self._prog_lbl.pack(anchor="w", padx=18, pady=(0, 14))

        grid = tk.Frame(page, bg=COLORS["bg"])
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

        log_wrap = ctk.CTkFrame(page, fg_color=COLORS["surface"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        log_wrap.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        ctk.CTkLabel(log_wrap, text="Console", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=16, pady=(12, 6))

        self._dash_log = ctk.CTkTextbox(log_wrap, font=ctk.CTkFont(size=13, family="Consolas"),
                                        fg_color=COLORS["bg"], text_color=COLORS["text"],
                                        border_width=1, border_color=COLORS["border"],
                                        corner_radius=8, wrap="word")
        self._dash_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

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
            ("Attack Types", "6 types: Gradual Drag, Freeze, Replay, Fake Walking, Geofence Evasion, Signal Manipulation"),
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
        if not self._detector:
            return
        
        self._attack_report_box.delete("1.0", "end")
        report = self._detector.get_attack_report()
        self._attack_report_box.insert("1.0", report)
        self._attack_report_box.configure(state="disabled")

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

        if result["cm"] is not None:
            cm_wrap = ctk.CTkFrame(self._blind_chart_frame, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            cm_wrap.pack(fill="x")

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
                'ensemble': self._detector.ensemble,
                'scaler': self._detector.scaler,
                'normalizer': self._detector.normalizer,
                'imputer': self._detector.imputer,
                'features': self._detector.features,
                'performance': self._detector.perf,
                'attack_counts': self._detector.attack_counts,
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

        p = self._detector.perf

        self._sample_rows = self._detector.generate_sample_table()

        sp_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                               border_width=1, border_color=COLORS["border"])
        sp_wrap.grid(row=0, column=0, sticky="ew", pady=(0, 16))

        sp_hdr = ctk.CTkFrame(sp_wrap, fg_color="transparent")
        sp_hdr.pack(fill="x", padx=18, pady=(16, 6))

        ctk.CTkLabel(sp_hdr, text="Model Performance on Increasing Test Sample Sizes",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(side="left")

        ctk.CTkButton(sp_hdr, text="⤢  Expand", command=self._open_sample_window,
                      fg_color="transparent", hover_color=COLORS["border"],
                      text_color=COLORS["accent"], border_width=1, border_color=COLORS["accent"],
                      height=32, width=100, corner_radius=6, font=ctk.CTkFont(size=13)).pack(side="right")

        sp_cols = ("Samples", "Accuracy", "F1_Score", "Normal_Correct", "Known_Correct", "ZeroDay_Predicted", "FP", "FN")
        sp_widths = (90, 100, 100, 140, 140, 160, 70, 70)

        sp_tree = ttk.Treeview(sp_wrap, columns=sp_cols, show="headings", style="GPS.Treeview",
                               height=min(12, len(self._sample_rows)))

        for col, w in zip(sp_cols, sp_widths):
            sp_tree.heading(col, text=col)
            sp_tree.column(col, width=w, anchor="center")

        for i, row in enumerate(self._sample_rows):
            tag = "even" if i % 2 == 0 else "odd"
            sp_tree.insert("", "end", tags=(tag,), values=(
                row['Samples'], f"{row['Accuracy']:.4f}", f"{row['F1_Score']:.4f}",
                row['Normal_Correct'], row['Known_Correct'], row['ZeroDay_Predicted'],
                row['FP'], row['FN'],
            ))

        sp_tree.tag_configure("odd", background=COLORS["tbl_odd"], foreground=COLORS["tbl_fg"])
        sp_tree.tag_configure("even", background=COLORS["tbl_even"], foreground=COLORS["tbl_fg"])
        sp_tree.bind("<Double-1>", lambda e: self._open_sample_window())

        sp_vsb = ttk.Scrollbar(sp_wrap, orient="vertical", command=sp_tree.yview)
        sp_tree.configure(yscrollcommand=sp_vsb.set)
        sp_tree.pack(side="left", fill="x", expand=True, padx=(18, 0), pady=(0, 18))
        sp_vsb.pack(side="left", fill="y", pady=(0, 18))

        tbl_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                border_width=1, border_color=COLORS["border"])
        tbl_wrap.grid(row=1, column=0, sticky="ew", pady=(0, 16))

        ctk.CTkLabel(tbl_wrap, text="Model Comparison", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

        cols = ("Model", "Accuracy", "Precision", "Recall", "F1 Score")
        tree = ttk.Treeview(tbl_wrap, columns=cols, show="headings", style="GPS.Treeview", height=5)

        widths = [240, 120, 120, 110, 120]
        for col, w in zip(cols, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center" if col != "Model" else "w")
        tree.column("Model", anchor="w")

        for i, (name, sc) in enumerate(self._detector.model_scores.items()):
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

        metrics_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                    border_width=1, border_color=COLORS["border"])
        metrics_wrap.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        metrics_wrap.columnconfigure(1, weight=1)

        ctk.CTkLabel(metrics_wrap, text="Ensemble Performance", font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=COLORS["text"]).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

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

        if self._detector.cm is not None:
            cm_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            cm_wrap.grid(row=3, column=0, sticky="ew", pady=(0, 16))

            ctk.CTkLabel(cm_wrap, text="Confusion Matrix", font=ctk.CTkFont(size=16, weight="bold"),
                         text_color=COLORS["text"]).pack(anchor="w", padx=18, pady=(16, 10))

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor(COLORS["card"])
            ax.set_facecolor(COLORS["card"])

            cm_data = self._detector.cm
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

        if hasattr(self._detector.detector.models['random_forest'], 'feature_importances_'):
            fi_wrap = ctk.CTkFrame(self._results_scroll, fg_color=COLORS["card"], corner_radius=12,
                                   border_width=1, border_color=COLORS["border"])
            fi_wrap.grid(row=4, column=0, sticky="ew", pady=(0, 16))

            ctk.CTkLabel(fi_wrap, text="Top 15 Feature Importance (Random Forest)",
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