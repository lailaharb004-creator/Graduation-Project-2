# Data generator
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

# CSV files live in the sibling "DataSets" folder, next to this "Code" folder.
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE), "DataSets")

# Clean (all-normal) input recording to inject spoofing into.
# Use the 40K recording for the training set, or switch to the 7K one below:
#   INPUT_FILE = os.path.join(DATA_DIR, "GPS_Data_All_Normal_7K.csv")
INPUT_FILE = os.path.join(DATA_DIR, "GPS_Data_All_Normal_40K.csv")

# Output 1: use this for model training/testing
TRAINING_OUTPUT_FILE = os.path.join(DATA_DIR, "GPS_Data_Mixed_40K.csv")

# Output 2: use this for report/evaluation only
METADATA_OUTPUT_FILE = os.path.join(DATA_DIR, "GPS_Data_Mixed_40K_with_metadata.csv")

ATTACK_LOG_FILE = os.path.join(DATA_DIR, "attack_windows_report.csv")
PLOTS_DIR = "spoofing_plots"

RANDOM_SEED = 42

# About 28% of rows become spoofed across the attack windows.
TARGET_SPOOF_RATIO = 0.28
NUM_ATTACK_ROUNDS = 10

# Keep beginning and ending records normal.
START_END_BUFFER = 240

# Human walking/running limits.
# IMPORTANT: velocity in your dataset is m/s, not km/h.
MAX_NORMAL_HUMAN_SPEED_MPS = 1.8
MAX_TRANSITION_SPEED_MPS = 2.2

# Smoother attack settings. These make the synthetic attacks less obvious by
# reducing coordinate offsets, spreading takeover/release over more rows, and
# reducing course/signal jumps.
GRADUAL_DRAG_OFFSET_RANGE_M = (12.0, 45.0)
FREEZE_OFFSET_RANGE_M = (3.0, 8.0)

# geofence_evasion has two movement styles, chosen randomly per attack
# instance. Both styles are kept inside the allowed zone (unlike the old
# standalone fake_walking attack, which never checked the zone at all).
#   "loiter"  -> slow pacing/fidgeting in place (low movement)
#   "walking" -> normal walking pace (looks like ordinary activity)
GEOFENCE_LOITER_STEP_MEAN_M = 0.45
GEOFENCE_LOITER_STEP_STD_M = 0.12
GEOFENCE_WALK_STEP_MEAN_M = 0.75
GEOFENCE_WALK_STEP_STD_M = 0.12
GEOFENCE_WALKING_STYLE_PROB = 0.5  # chance of "walking" vs "loiter" per attack



# Extra controls for the four detection features you want to make more logical:
# distance_m, coord_speed, speed_residual, and course_change.
MAX_REPORTED_STEP_SPEED_STABLE_MPS = 1.55
MAX_REPORTED_STEP_SPEED_TRANSITION_MPS = 2.15
MAX_COURSE_TURN_STABLE_DEG = 14.0
MAX_COURSE_TURN_TRANSITION_DEG = 22.0
TARGET_SPEED_RESIDUAL_RANGE_MPS = (0.10, 0.24)
COURSE_BEARING_NOISE_STD_DEG = 3.2


# Geofence settings.
# If ALLOWED_CENTER_LAT/LON are None, the script uses the first GPS point as the allowed-zone center.
ALLOWED_CENTER_LAT = None
ALLOWED_CENTER_LON = None
ALLOWED_RADIUS_M = 100.0

# Pool of attack types. Each session gets its own shuffled order built from
# this pool (see build_session_attack_plan), so attack-type order varies
# across sessions instead of always following the same fixed sequence.
BASE_ATTACK_TYPES = [
    "gradual_drag",
    "freeze",
    "replay",
    "geofence_evasion",
    
]

# Required columns in your current dataset.
REQUIRED_COLUMNS = [
    "session_id",
    "gps_date",
    "gps_time",
    "latitude",
    "longitude",
    "velocity",
    "course",
    "satellites_in_view",
    "satellites_used",
    "hdop",
    "label",
]

# Optional column. If present in future datasets, it will be handled.
OPTIONAL_SAT_LOCKS_COL = "satellite_locks"

EARTH_RADIUS_M = 6371000.0


# ============================================================
# GPS HELPERS
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points. Supports scalars."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return EARTH_RADIUS_M * 2.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1 = math.radians(float(lat1))
    lat2 = math.radians(float(lat2))
    dlon = math.radians(float(lon2) - float(lon1))

    x = math.sin(dlon) * math.cos(lat2)
    y = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    )

    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def destination_point(lat, lon, distance_m, bearing_degrees):
    """Move from lat/lon by distance_m at bearing_degrees."""
    bearing = math.radians(float(bearing_degrees))
    lat1 = math.radians(float(lat))
    lon1 = math.radians(float(lon))
    ang_dist = float(distance_m) / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(ang_dist)
        + math.cos(lat1) * math.sin(ang_dist) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(ang_dist) * math.cos(lat1),
        math.cos(ang_dist) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def smoothstep(x):
    """Smooth transition 0 -> 1."""
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def transition_value(raw_t, mode, rng):
    """
    Not every attack uses perfect smoothstep.
    This avoids making spoofed data look artificially smooth every time.
    """
    raw_t = float(np.clip(raw_t, 0, 1))

    if mode == "linear":
        t = raw_t

    elif mode == "smooth":
        t = smoothstep(raw_t)

    elif mode == "noisy_smooth":
        t = smoothstep(raw_t)
        # Less noise than the original version, so takeover/release do not
        # create large direction jumps.
        t += rng.normal(0, 0.012)
        t = float(np.clip(t, 0, 1))

    elif mode == "delayed":
        # Starts slower, then catches up.
        delayed_t = max(0.0, (raw_t - 0.18) / 0.82)
        t = smoothstep(delayed_t)

    else:
        t = smoothstep(raw_t)

    return float(np.clip(t, 0, 1))


def random_transition_mode(rng):
    # Prefer smooth transitions. Linear and delayed transitions can create
    # sharper coordinate-speed/course-change patterns, so they are rare here.
    modes = ["linear", "smooth", "noisy_smooth", "delayed"]
    probabilities = [0.00, 0.72, 0.28, 0.00]
    return str(rng.choice(modes, p=probabilities))


# ============================================================
# DATA PREPARATION
# ============================================================

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + str(missing) + "\n"
            "Available columns: " + str(list(df.columns))
        )


def parse_datetime(df):
    return pd.to_datetime(
        df["gps_date"].astype(str) + " " + df["gps_time"].astype(str),
        errors="coerce",
    )


def _gps_time_to_seconds(value):
    """
    Robust parser for your exported gps_time values.

    Your CSV shows values like 12:56.0, 58:39.0, 01:49.0.
    That is usually MM:SS.s from the GPS/NMEA time after spreadsheet export,
    not HH:MM. So we convert it to seconds inside the current hour.
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip()

    try:
        parts = s.split(":")
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            sec = float(parts[2])
            return h * 3600.0 + m * 60.0 + sec

        if len(parts) == 2:
            m = float(parts[0])
            sec = float(parts[1])
            return m * 60.0 + sec

        return float(s)
    except Exception:
        return np.nan


def compute_time_deltas(df):
    df = df.copy()

    # Keep a readable datetime only for metadata. It may be NaT because gps_time
    # is not a complete HH:MM:SS value in this exported file.
    df["_datetime"] = parse_datetime(df)

    seconds = df["gps_time"].apply(_gps_time_to_seconds)
    delta = seconds.diff()

    # If MM:SS wraps from 59:59 to 00:00, add one hour.
    delta = np.where(delta < 0, delta + 3600.0, delta)

    df["time_delta_sec"] = pd.Series(delta, index=df.index)
    df["time_delta_sec"] = pd.to_numeric(df["time_delta_sec"], errors="coerce")

    # First row of each session or unparsable rows default to 1 second.
    df["time_delta_sec"] = df["time_delta_sec"].fillna(1.0)
    df.loc[df["time_delta_sec"] <= 0, "time_delta_sec"] = 1.0

    # Your collector stores about one row per second. This keeps rare jumps safe.
    df["time_delta_sec"] = df["time_delta_sec"].clip(lower=0.5, upper=5.0)
    return df


def fill_missing_course(df):
    """
    Fill missing course using movement bearing.
    If almost stationary, course becomes 0.0 instead of being carried forever.
    """
    df = df.copy()
    course = pd.to_numeric(df["course"], errors="coerce")

    for i in range(len(df)):
        if pd.notna(course.iloc[i]):
            continue

        if i == 0:
            course.iloc[i] = 0.0
            continue

        dist = haversine_m(
            df.loc[i - 1, "latitude"],
            df.loc[i - 1, "longitude"],
            df.loc[i, "latitude"],
            df.loc[i, "longitude"],
        )

        if dist > 0.6:
            course.iloc[i] = bearing_deg(
                df.loc[i - 1, "latitude"],
                df.loc[i - 1, "longitude"],
                df.loc[i, "latitude"],
                df.loc[i, "longitude"],
            )
        else:
            course.iloc[i] = 0.0

    df["course"] = course.fillna(0.0)
    return df


def prepare_input_data(input_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Put the CSV file in the same folder as this script, or change INPUT_FILE."
        )

    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()
    validate_columns(df)

    # Preserve order. Do NOT shuffle.
    df = df.copy().reset_index(drop=True)

    numeric_cols = [
        "latitude", "longitude", "velocity", "course",
        "satellites_in_view", "satellites_used", "hdop"
    ]

    if OPTIONAL_SAT_LOCKS_COL in df.columns:
        numeric_cols.append(OPTIONAL_SAT_LOCKS_COL)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    df = compute_time_deltas(df)
    df = fill_missing_course(df)

    # Original/true coordinates kept only for metadata evaluation.
    df["true_latitude"] = df["latitude"]
    df["true_longitude"] = df["longitude"]

    # Original labels reset to normal before generating new spoof windows.
    df["label"] = 0

    # Metadata columns. These are NOT for training.
    df["is_generated_spoof"] = 0
    df["attack_round_id"] = 0
    df["attack_type"] = "normal"
    df["attack_phase"] = "normal"
    df["transition_mode"] = "none"
    df["geofence_style"] = "none"
    df["replay_source_start_index"] = np.nan

    return df


# ============================================================
# ATTACK WINDOW SELECTION
# ============================================================

def make_attack_durations(total_records, target_ratio, num_rounds, rng, min_duration=80):
    target_total = int(round(total_records * target_ratio))
    base = target_total // num_rounds

    durations = []
    for _ in range(num_rounds):
        jitter = int(rng.integers(-45, 46))
        durations.append(max(min_duration, int(base + jitter)))

    # Adjust durations to hit the exact target total.
    # Guard against infinite loop: if all durations are at minimum (140)
    # and we need to reduce further, just accept the closest we can get.
    diff = target_total - sum(durations)
    i = 0
    max_iter = (abs(diff) + 1) * num_rounds * 3
    iterations = 0
    while diff != 0 and iterations < max_iter:
        step = 1 if diff > 0 else -1
        idx = i % num_rounds
        if durations[idx] + step >= min_duration:
            durations[idx] += step
            diff -= step
        i += 1
        iterations += 1

    return durations


def choose_attack_windows(n, durations):
    """
    Choose non-overlapping windows in chronological order.
    No record shuffling.
    """
    total_attack = sum(durations)
    available_normal = n - total_attack - (2 * START_END_BUFFER)

    if available_normal <= (len(durations) + 1) * 20:
        raise ValueError("Dataset is too small for these attack settings.")

    gap = available_normal // (len(durations) + 1)
    windows = []
    cursor = START_END_BUFFER + gap

    for duration in durations:
        start = int(cursor)
        end = int(start + duration - 1)
        windows.append((start, end, duration))
        cursor = end + 1 + gap

    return windows


def split_phases(start, end):
    """
    Split each attack into smoother phases.

    Compared with the previous version, takeover/release are longer. This
    spreads the location change over more records, lowering coord_speed and
    course_change so the attack is less obvious.
    """
    length = end - start + 1

    capture_len = max(10, int(length * 0.06))
    takeover_len = max(35, int(length * 0.35))
    release_len = max(35, int(length * 0.35))
    stable_len = length - capture_len - takeover_len - release_len

    min_stable_len = min(45, max(15, int(length * 0.18)))

    # If the window is short, borrow rows from takeover/release while keeping
    # both transitions long enough to stay smooth.
    while stable_len < min_stable_len and (takeover_len > 25 or release_len > 25):
        if takeover_len >= release_len and takeover_len > 25:
            takeover_len -= 1
        elif release_len > 25:
            release_len -= 1
        else:
            break
        stable_len = length - capture_len - takeover_len - release_len

    if stable_len < 1:
        stable_len = 1
        overflow = capture_len + takeover_len + release_len + stable_len - length
        release_len = max(10, release_len - overflow)

    capture = (start, start + capture_len - 1)
    takeover = (capture[1] + 1, capture[1] + takeover_len)
    stable = (takeover[1] + 1, takeover[1] + stable_len)
    release = (stable[1] + 1, end)

    return {
        "capture": capture,
        "takeover": takeover,
        "stable": stable,
        "release": release,
    }

def assign_phase_metadata(df, phases):
    for phase_name, (p_start, p_end) in phases.items():
        df.loc[p_start:p_end, "attack_phase"] = phase_name


def build_session_attack_plan(num_rounds, rng):
    """
    Build a per-session sequence of attack types by shuffling full cycles of
    BASE_ATTACK_TYPES. This keeps the type counts balanced (each type
    appears roughly num_rounds / len(BASE_ATTACK_TYPES) times) while varying
    the order across sessions, instead of every session following the same
    fixed sequence. Still reproducible since rng is seeded.
    """
    plan = []
    while len(plan) < num_rounds:
        cycle = BASE_ATTACK_TYPES.copy()
        rng.shuffle(cycle)
        plan.extend(cycle)
    return plan[:num_rounds]


# ============================================================
# GEOFENCE HELPERS
# ============================================================

def get_allowed_center(df):
    if ALLOWED_CENTER_LAT is not None and ALLOWED_CENTER_LON is not None:
        return float(ALLOWED_CENTER_LAT), float(ALLOWED_CENTER_LON)

    # For one-session dataset, use the first point as the allowed area center.
    return float(df.loc[0, "latitude"]), float(df.loc[0, "longitude"])


def inside_allowed_zone(lat, lon, center_lat, center_lon, radius_m):
    d = haversine_m(lat, lon, center_lat, center_lon)
    return bool(d <= radius_m)


def random_point_inside_allowed_zone(center_lat, center_lon, radius_m, rng, margin=0.75):
    # sqrt makes points more evenly distributed over the circle area.
    r = radius_m * margin * math.sqrt(float(rng.random()))
    bearing = float(rng.uniform(0, 360))
    return destination_point(center_lat, center_lon, r, bearing)


def add_zone_metadata(df, center_lat, center_lon):
    true_dist = haversine_m(
        df["true_latitude"].to_numpy(),
        df["true_longitude"].to_numpy(),
        center_lat,
        center_lon,
    )

    reported_dist = haversine_m(
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy(),
        center_lat,
        center_lon,
    )

    df["true_distance_from_allowed_center_m"] = np.round(true_dist, 2)
    df["reported_distance_from_allowed_center_m"] = np.round(reported_dist, 2)

    df["true_zone_status"] = np.where(true_dist <= ALLOWED_RADIUS_M, "allowed", "forbidden")
    df["reported_zone_status"] = np.where(reported_dist <= ALLOWED_RADIUS_M, "allowed", "forbidden")

    df["geofence_evasion_case"] = np.where(
        (df["true_zone_status"] == "forbidden")
        & (df["reported_zone_status"] == "allowed")
        & (df["label"] == 1),
        1,
        0,
    )

    return df


# ============================================================
# SIGNAL QUALITY MODIFICATION
# ============================================================

def modify_signal_quality(df, idxs, phase, rng):
    """
    Slightly modifies hdop/satellites.
    The goal is realistic inconsistency, not obvious impossible values.
    """
    # Restored to the original/pre-smoother signal behavior.
    # Only the motion-related behavior is made subtler in this file.
    if phase == "capture":
        drop_used_range = (0, 1)
        hdop_mult_range = (1.02, 1.18)
        view_drop_prob = 0.20

    elif phase == "takeover":
        drop_used_range = (1, 3)
        hdop_mult_range = (1.15, 1.85)
        view_drop_prob = 0.45

    elif phase == "stable":
        drop_used_range = (0, 2)
        hdop_mult_range = (1.05, 1.45)
        view_drop_prob = 0.25

    else:  # release
        drop_used_range = (1, 3)
        hdop_mult_range = (1.15, 1.75)
        view_drop_prob = 0.40

    for idx in idxs:
        original_view = int(df.at[idx, "satellites_in_view"])
        original_used = int(df.at[idx, "satellites_used"])
        original_hdop = float(df.at[idx, "hdop"])

        view = original_view
        if rng.random() < view_drop_prob:
            view -= int(rng.integers(0, 3))
        view = int(np.clip(view, 6, 14))

        used_drop = int(rng.integers(drop_used_range[0], drop_used_range[1] + 1))
        used = original_used - used_drop
        used = int(np.clip(used, 4, min(view, 10)))

        # Sometimes keep signal almost normal.
        if phase == "stable" and rng.random() < 0.35:
            used = int(np.clip(original_used, 4, min(view, 10)))

        hdop = original_hdop * rng.uniform(*hdop_mult_range) + rng.uniform(0.00, 0.15)
        hdop = float(np.clip(hdop, 0.75, 4.00))

        df.at[idx, "satellites_in_view"] = view
        df.at[idx, "satellites_used"] = used
        df.at[idx, "hdop"] = round(hdop, 2)

        if OPTIONAL_SAT_LOCKS_COL in df.columns:
            original_locks = int(df.at[idx, OPTIONAL_SAT_LOCKS_COL])
            locks = int(np.clip(original_locks - used_drop, 3, used))
            df.at[idx, OPTIONAL_SAT_LOCKS_COL] = locks


# ============================================================
# WRITE / METADATA / RECALCULATE
# ============================================================

def write_fake_points(df, fake_points):
    for idx, (lat, lon) in fake_points.items():
        df.at[idx, "latitude"] = float(lat)
        df.at[idx, "longitude"] = float(lon)


def set_attack_metadata(df, start, end, round_id, attack_type, transition_mode):
    df.loc[start:end, "label"] = 1
    df.loc[start:end, "is_generated_spoof"] = 1
    df.loc[start:end, "attack_round_id"] = int(round_id)
    df.loc[start:end, "attack_type"] = str(attack_type)
    df.loc[start:end, "transition_mode"] = str(transition_mode)



def _limited_angle(prev_deg, target_deg, max_turn_deg):
    """Move from prev_deg toward target_deg using the shortest direction, capped by max_turn_deg."""
    if pd.isna(prev_deg):
        return float(target_deg) % 360.0
    diff = (float(target_deg) - float(prev_deg) + 180.0) % 360.0 - 180.0
    diff = float(np.clip(diff, -max_turn_deg, max_turn_deg))
    return (float(prev_deg) + diff) % 360.0


def limit_reported_motion_for_window(df, start, end, rng):
    """
    Make the reported coordinate path less obvious by limiting per-row movement.

    This directly targets the detector features:
      - distance_m
      - coord_speed
      - speed_residual (because velocity is later recalculated from the limited path)
      - course_change (because large coordinate jumps create large bearing changes)
    """
    for idx in range(start + 1, end + 1):
        phase = df.at[idx, "attack_phase"]
        dt = df.at[idx, "time_delta_sec"] if "time_delta_sec" in df.columns else 1.0
        if pd.isna(dt) or dt <= 0:
            dt = 1.0

        max_speed = (
            MAX_REPORTED_STEP_SPEED_TRANSITION_MPS
            if phase in ["takeover", "release"]
            else MAX_REPORTED_STEP_SPEED_STABLE_MPS
        )
        max_step_m = max_speed * float(dt) * float(rng.uniform(0.92, 1.00))

        prev_lat = df.at[idx - 1, "latitude"]
        prev_lon = df.at[idx - 1, "longitude"]
        cur_lat = df.at[idx, "latitude"]
        cur_lon = df.at[idx, "longitude"]

        dist = float(haversine_m(prev_lat, prev_lon, cur_lat, cur_lon))
        if dist > max_step_m:
            brng = bearing_deg(prev_lat, prev_lon, cur_lat, cur_lon)
            new_lat, new_lon = destination_point(prev_lat, prev_lon, max_step_m, brng)
            df.at[idx, "latitude"] = new_lat
            df.at[idx, "longitude"] = new_lon

def recalculate_motion_for_window(df, start, end, rng):
    """
    Recalculate velocity/course after modifying coordinates.
    Keeps reported velocity in m/s.
    """
    for idx in range(start, end + 1):
        if idx == 0:
            continue

        lat1 = df.at[idx - 1, "latitude"]
        lon1 = df.at[idx - 1, "longitude"]
        lat2 = df.at[idx, "latitude"]
        lon2 = df.at[idx, "longitude"]

        dist = float(haversine_m(lat1, lon1, lat2, lon2))

        dt = df.at[idx, "time_delta_sec"] if "time_delta_sec" in df.columns else 1.0
        if pd.isna(dt) or dt <= 0:
            dt = 1.0

        speed_mps = dist / dt

        phase = df.at[idx, "attack_phase"]
        transition_cap = MAX_TRANSITION_SPEED_MPS * rng.uniform(0.98, 1.02)
        max_speed = transition_cap if phase in ["takeover", "release"] else MAX_NORMAL_HUMAN_SPEED_MPS

        # Keep reported velocity realistic, but not perfectly equal to coord_speed.
        # Real GPS velocity normally has small measurement noise, so speed_residual
        # should not become almost zero for spoofed rows.
        residual = float(rng.uniform(*TARGET_SPEED_RESIDUAL_RANGE_MPS))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        candidate_speed = speed_mps + sign * residual

        # If the chosen direction makes velocity impossible, put the noise on
        # the other side. This keeps residual present without creating negatives.
        if candidate_speed < 0.03:
            candidate_speed = speed_mps + residual
        if candidate_speed > max_speed:
            candidate_speed = max(0.03, speed_mps - residual)

        reported_speed = float(np.clip(candidate_speed + rng.normal(0, 0.012), 0.03, max_speed))

        prev_course = df.at[idx - 1, "course"] if pd.notna(df.at[idx - 1, "course"]) else 0.0
        max_turn = (
            MAX_COURSE_TURN_TRANSITION_DEG
            if phase in ["takeover", "release"]
            else MAX_COURSE_TURN_STABLE_DEG
        )

        if dist > 0.7:
            # Make course agree with the actual coordinate bearing, with normal
            # GPS heading noise. This reduces course_bearing_diff. The larger
            # turn cap still prevents impossible instant direction flips.
            target_crs = bearing_deg(lat1, lon1, lat2, lon2)
            target_crs = (target_crs + rng.normal(0, COURSE_BEARING_NOISE_STD_DEG)) % 360
            crs = _limited_angle(prev_course, target_crs, max_turn)

            # If the cap still leaves course too far from the coordinate bearing,
            # allow one extra partial correction. This fixes high course_bearing_diff
            # without fully removing natural course smoothing.
            diff_to_target = (target_crs - crs + 180.0) % 360.0 - 180.0
            if abs(diff_to_target) > 10.0:
                crs = (crs + np.clip(diff_to_target, -8.0, 8.0)) % 360.0
        else:
            # When almost stationary, course is not reliable. Keep it calm.
            if reported_speed < 0.10:
                crs = 0.0
            else:
                target_crs = (prev_course + rng.normal(0, 2.0)) % 360
                crs = _limited_angle(prev_course, target_crs, max_turn)

        df.at[idx, "velocity"] = round(reported_speed, 3)
        df.at[idx, "course"] = round(crs, 2)


# ============================================================
# ATTACK GENERATORS
# ============================================================

def apply_gradual_drag_attack(df, start, end, phases, round_id, rng):
    """
    The reported location is gradually dragged away from the real path,
    then returns to the real path. Useful for geofence-related spoofing behavior.
    """
    final_offset_m = float(rng.uniform(*GRADUAL_DRAG_OFFSET_RANGE_M))
    offset_bearing = float(rng.uniform(0, 360))
    transition_mode = random_transition_mode(rng)

    fake_points = {}

    for idx in range(start, end + 1):
        real_lat = df.at[idx, "true_latitude"]
        real_lon = df.at[idx, "true_longitude"]

        if phases["capture"][0] <= idx <= phases["capture"][1]:
            offset = 0.0

        elif phases["takeover"][0] <= idx <= phases["takeover"][1]:
            denom = max(1, phases["takeover"][1] - phases["takeover"][0])
            t = transition_value((idx - phases["takeover"][0]) / denom, transition_mode, rng)
            offset = final_offset_m * t

        elif phases["stable"][0] <= idx <= phases["stable"][1]:
            offset = final_offset_m + rng.normal(0, 1.0)

        else:
            denom = max(1, phases["release"][1] - phases["release"][0])
            t = transition_value((idx - phases["release"][0]) / denom, transition_mode, rng)
            offset = final_offset_m * (1 - t)

        # Small realistic jitter added to avoid perfect geometry.
        jitter_m = float(abs(rng.normal(0, 0.25)))
        lat2, lon2 = destination_point(real_lat, real_lon, max(0.0, offset), offset_bearing)
        lat2, lon2 = destination_point(lat2, lon2, jitter_m, rng.uniform(0, 360))
        fake_points[idx] = (lat2, lon2)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "gradual_drag", transition_mode)


def apply_freeze_attack(df, start, end, phases, round_id, rng):
    """
    The reported location remains almost fixed near a safe point,
    with very small GPS jitter.
    """
    transition_mode = random_transition_mode(rng)

    lat0 = df.at[start, "true_latitude"]
    lon0 = df.at[start, "true_longitude"]

    offset_m = float(rng.uniform(*FREEZE_OFFSET_RANGE_M))
    fake_lat, fake_lon = destination_point(lat0, lon0, offset_m, rng.uniform(0, 360))

    fake_points = {}

    for idx in range(start, end + 1):
        if phases["capture"][0] <= idx <= phases["capture"][1]:
            fake_points[idx] = (df.at[idx, "true_latitude"], df.at[idx, "true_longitude"])

        elif phases["takeover"][0] <= idx <= phases["takeover"][1]:
            denom = max(1, phases["takeover"][1] - phases["takeover"][0])
            t = transition_value((idx - phases["takeover"][0]) / denom, transition_mode, rng)
            real_lat = df.at[idx, "true_latitude"]
            real_lon = df.at[idx, "true_longitude"]
            jlat, jlon = destination_point(fake_lat, fake_lon, rng.uniform(0, 0.8), rng.uniform(0, 360))
            fake_points[idx] = ((1 - t) * real_lat + t * jlat, (1 - t) * real_lon + t * jlon)

        elif phases["stable"][0] <= idx <= phases["stable"][1]:
            jitter_m = float(rng.uniform(0.2, 1.2))
            fake_points[idx] = destination_point(fake_lat, fake_lon, jitter_m, rng.uniform(0, 360))

        else:
            denom = max(1, phases["release"][1] - phases["release"][0])
            t = transition_value((idx - phases["release"][0]) / denom, transition_mode, rng)
            real_lat = df.at[idx, "true_latitude"]
            real_lon = df.at[idx, "true_longitude"]
            jlat, jlon = destination_point(fake_lat, fake_lon, rng.uniform(0, 0.8), rng.uniform(0, 360))
            fake_points[idx] = ((1 - t) * jlat + t * real_lat, (1 - t) * jlon + t * real_lon)

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "freeze", transition_mode)


def find_replay_source(df, start, stable_len):
    """
    Find an earlier normal segment.
    For one-session data, source must be before the attack and not already spoofed.
    """
    min_source_start = 30
    max_source_start = start - stable_len - 90

    if max_source_start <= min_source_start:
        return None

    current_lat = df.at[start, "true_latitude"]
    current_lon = df.at[start, "true_longitude"]

    best_candidate = None
    best_dist = float("inf")

    for cand in range(min_source_start, max_source_start, 8):
        cand_end = cand + stable_len - 1
        if cand_end >= start:
            continue

        # Do not replay from already spoofed windows.
        if df.loc[cand:cand_end, "label"].sum() != 0:
            continue

        d = haversine_m(current_lat, current_lon, df.at[cand, "true_latitude"], df.at[cand, "true_longitude"])

        if d < best_dist:
            best_dist = float(d)
            best_candidate = cand

    return best_candidate


def apply_replay_attack(df, start, end, phases, round_id, rng):
    """
    Replays an older legitimate trajectory at current time.
    """
    transition_mode = random_transition_mode(rng)

    stable_start, stable_end = phases["stable"]
    stable_len = stable_end - stable_start + 1

    source_start = find_replay_source(df, start, stable_len)

    if source_start is None:
        # If no source is available, fallback to gradual drag.
        apply_gradual_drag_attack(df, start, end, phases, round_id, rng)
        df.loc[start:end, "attack_type"] = "replay_fallback_gradual_drag"
        return

    source_idxs = list(range(source_start, source_start + stable_len))
    replay_lats = df.loc[source_idxs, "true_latitude"].to_numpy()
    replay_lons = df.loc[source_idxs, "true_longitude"].to_numpy()

    fake_points = {}

    # Capture: still true position.
    for idx in range(phases["capture"][0], phases["capture"][1] + 1):
        fake_points[idx] = (df.at[idx, "true_latitude"], df.at[idx, "true_longitude"])

    # Takeover: move from current true point to replayed old point.
    target_lat, target_lon = replay_lats[0], replay_lons[0]
    for idx in range(phases["takeover"][0], phases["takeover"][1] + 1):
        denom = max(1, phases["takeover"][1] - phases["takeover"][0])
        t = transition_value((idx - phases["takeover"][0]) / denom, transition_mode, rng)

        real_lat = df.at[idx, "true_latitude"]
        real_lon = df.at[idx, "true_longitude"]

        fake_points[idx] = (
            (1 - t) * real_lat + t * target_lat,
            (1 - t) * real_lon + t * target_lon,
        )

    # Stable: old legitimate route is replayed.
    for k, idx in enumerate(range(stable_start, stable_end + 1)):
        fake_points[idx] = (replay_lats[k], replay_lons[k])

    # Release: return from last replay point to current true path.
    last_lat, last_lon = replay_lats[-1], replay_lons[-1]
    for idx in range(phases["release"][0], phases["release"][1] + 1):
        denom = max(1, phases["release"][1] - phases["release"][0])
        t = transition_value((idx - phases["release"][0]) / denom, transition_mode, rng)

        real_lat = df.at[idx, "true_latitude"]
        real_lon = df.at[idx, "true_longitude"]

        fake_points[idx] = (
            (1 - t) * last_lat + t * real_lat,
            (1 - t) * last_lon + t * real_lon,
        )

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "replay", transition_mode)
    df.loc[start:end, "replay_source_start_index"] = int(source_start)


def apply_geofence_evasion_attack(df, start, end, phases, round_id, rng, center_lat, center_lon):
    """
    Bracelet/geofence scenario:
    True path may be outside or moving, but reported GPS is kept inside the
    allowed zone.

    Two movement styles are chosen randomly per attack instance, and BOTH are
    kept inside the allowed zone via inside_allowed_zone() / pull-back:
      - "loiter":  slow pacing/fidgeting in place (low movement)
      - "walking": normal walking pace, looks like ordinary activity

    "walking" is the harder-to-detect, more dangerous case for this threat
    model (someone outside the allowed zone, reporting GPS as if they were
    inside walking around normally), so it's included here instead of as a
    separate zone-unaware attack.
    """
    transition_mode = random_transition_mode(rng)
    style = "walking" if rng.random() < GEOFENCE_WALKING_STYLE_PROB else "loiter"

    if style == "walking":
        step_mean, step_std = GEOFENCE_WALK_STEP_MEAN_M, GEOFENCE_WALK_STEP_STD_M
        step_clip = (0.25, 1.35)
        bearing_noise_deg = 4.5
    else:
        step_mean, step_std = GEOFENCE_LOITER_STEP_MEAN_M, GEOFENCE_LOITER_STEP_STD_M
        step_clip = (0.10, 1.00)
        bearing_noise_deg = 7.0

    # Pick a safe reported point inside the allowed zone, but near the
    # direction of the true path. This avoids an unnecessarily long jump to
    # the opposite side of the geofence.
    bearing_to_true = bearing_deg(
        center_lat, center_lon,
        df.at[start, "true_latitude"], df.at[start, "true_longitude"]
    )
    safe_distance_m = ALLOWED_RADIUS_M * float(rng.uniform(0.40, 0.70))
    safe_lat, safe_lon = destination_point(
        center_lat, center_lon, safe_distance_m,
        (bearing_to_true + rng.normal(0, 10.0)) % 360
    )

    fake_points = {}

    stable_start, stable_end = phases["stable"]

    # Generate fake movement inside allowed zone.
    current_lat, current_lon = safe_lat, safe_lon
    current_bearing = float(rng.uniform(0, 360))
    stable_points = []

    for _ in range(stable_start, stable_end + 1):
        step_m = float(np.clip(rng.normal(step_mean, step_std), *step_clip))
        current_bearing = (current_bearing + rng.normal(0, bearing_noise_deg)) % 360

        next_lat, next_lon = destination_point(current_lat, current_lon, step_m, current_bearing)

        # If the point leaves the allowed zone, pull it back inside.
        if not inside_allowed_zone(next_lat, next_lon, center_lat, center_lon, ALLOWED_RADIUS_M * 0.90):
            next_lat, next_lon = random_point_inside_allowed_zone(
                center_lat, center_lon, ALLOWED_RADIUS_M, rng, margin=0.70
            )

        current_lat, current_lon = next_lat, next_lon
        stable_points.append((current_lat, current_lon))

    # Capture: true position at first.
    for idx in range(phases["capture"][0], phases["capture"][1] + 1):
        fake_points[idx] = (df.at[idx, "true_latitude"], df.at[idx, "true_longitude"])

    # Takeover: reported position moves into allowed safe path.
    target_lat, target_lon = stable_points[0]
    for idx in range(phases["takeover"][0], phases["takeover"][1] + 1):
        denom = max(1, phases["takeover"][1] - phases["takeover"][0])
        t = transition_value((idx - phases["takeover"][0]) / denom, transition_mode, rng)

        real_lat = df.at[idx, "true_latitude"]
        real_lon = df.at[idx, "true_longitude"]

        fake_points[idx] = (
            (1 - t) * real_lat + t * target_lat,
            (1 - t) * real_lon + t * target_lon,
        )

    # Stable: stay/move inside allowed zone.
    for k, idx in enumerate(range(stable_start, stable_end + 1)):
        fake_points[idx] = stable_points[k]

    # Release: return to true path.
    last_lat, last_lon = stable_points[-1]
    for idx in range(phases["release"][0], phases["release"][1] + 1):
        denom = max(1, phases["release"][1] - phases["release"][0])
        t = transition_value((idx - phases["release"][0]) / denom, transition_mode, rng)

        real_lat = df.at[idx, "true_latitude"]
        real_lon = df.at[idx, "true_longitude"]

        fake_points[idx] = (
            (1 - t) * last_lat + t * real_lat,
            (1 - t) * last_lon + t * real_lon,
        )

    write_fake_points(df, fake_points)
    set_attack_metadata(df, start, end, round_id, "geofence_evasion", transition_mode)
    df.loc[start:end, "geofence_style"] = style




# ============================================================
# PLOTS / REPORT
# ============================================================

def create_plots(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    x = np.arange(len(df))

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["label"].to_numpy())
    plt.title("Label Timeline: Normal vs Spoofed")
    plt.xlabel("Record index")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "label_timeline.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["velocity"].to_numpy())
    plt.title("Velocity Over Time")
    plt.xlabel("Record index")
    plt.ylabel("Velocity (m/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "velocity_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(x, df["hdop"].to_numpy())
    plt.title("HDOP Over Time")
    plt.xlabel("Record index")
    plt.ylabel("HDOP")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hdop_over_time.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 7))
    normal = df[df["label"] == 0]
    spoofed = df[df["label"] == 1]
    plt.scatter(normal["longitude"], normal["latitude"], s=3, alpha=0.45, label="normal")
    plt.scatter(spoofed["longitude"], spoofed["latitude"], s=3, alpha=0.70, label="spoofed")
    plt.title("Reported Trajectory: Normal and Spoofed")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "reported_trajectory_normal_spoofed.png"), dpi=200)
    plt.close()

    # True vs reported, useful for metadata review.
    plt.figure(figsize=(7, 7))
    plt.scatter(df["true_longitude"], df["true_latitude"], s=2, alpha=0.30, label="true/original")
    plt.scatter(df["longitude"], df["latitude"], s=2, alpha=0.30, label="reported/output")
    plt.title("True Original Path vs Reported Path")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "true_vs_reported_path.png"), dpi=200)
    plt.close()


def print_summary(df, attack_log):
    total = len(df)
    spoofed = int((df["label"] == 1).sum())
    normal = total - spoofed

    print("\n" + "=" * 72)
    print("BALANCED-MOTION GPS SPOOFING SIMULATION GENERATED")
    print("=" * 72)
    print(f"Total records:   {total}")
    print(f"Normal records:  {normal} ({normal / total * 100:.2f}%)")
    print(f"Spoofed records: {spoofed} ({spoofed / total * 100:.2f}%)")

    print("\nPer-session summary:")
    for sid, grp in df.groupby("session_id"):
        s = int((grp["label"] == 1).sum())
        print(f"  {sid}: {len(grp)} rows, spoofed={s} ({s/len(grp)*100:.1f}%)")

    print("\nAttack windows:")
    for item in attack_log:
        sid_str = f"[{item.get('session_id', '?')}] " if "session_id" in item else ""
        print(
            f"  {sid_str}Round {item['attack_round_id']:>2}: "
            f"{item['attack_type']:<18} "
            f"start={item['start_index']:<5} "
            f"end={item['end_index']:<5} "
            f"duration={item['duration_records']:<4} "
            f"transition={item['transition_mode']}"
        )

    print("\nFeature ranges after generation:")
    for col in ["velocity", "satellites_in_view", "satellites_used", "hdop"]:
        print(
            f"  {col:<20} min={df[col].min():.3f}  "
            f"mean={df[col].mean():.3f}  max={df[col].max():.3f}"
        )

    if "attack_type" in df.columns:
        print("\nAttack type counts:")
        print(df[df["label"] == 1]["attack_type"].value_counts().to_string())
    print("=" * 72)


# ============================================================
# MAIN PIPELINE
# ============================================================

def _process_one_session(session_df, rng, center_lat, center_lon):
    """
    Run the full attack pipeline on a single session's data.
    session_df must already be reset_index(drop=True).
    Returns (processed_df, attack_log_list).
    """
    df = prepare_input_data_from_df(session_df)
    n = len(df)

    # Sessions are processed independently. Short sessions cannot safely fit
    # NUM_ATTACK_ROUNDS windows, so the round count is chosen dynamically below.
    min_required = 2 * START_END_BUFFER + 100
    if n < min_required:
        print(f"  [SKIP] Session too short ({n} rows, need at least {min_required}). Keeping as normal.")
        df["label"] = 0
        df["is_generated_spoof"] = 0
        df["attack_round_id"] = 0
        df["attack_type"] = "normal"
        df["attack_phase"] = "normal"
        df["transition_mode"] = "none"
        df["geofence_style"] = "none"
        return df, []

    target_attack_rows = int(round(n * TARGET_SPOOF_RATIO))

    # Aim for realistic windows of about 160+ records, but allow fewer rounds
    # for shorter sessions so the spoof ratio stays close to TARGET_SPOOF_RATIO.
    num_rounds = min(NUM_ATTACK_ROUNDS, max(1, target_attack_rows // 160))

    durations = None
    windows = None

    while num_rounds >= 1:
        durations_try = make_attack_durations(
            n,
            TARGET_SPOOF_RATIO,
            num_rounds,
            rng,
            min_duration=80,
        )
        try:
            windows_try = choose_attack_windows(n, durations_try)
            durations = durations_try
            windows = windows_try
            break
        except ValueError:
            num_rounds -= 1

    if windows is None:
        print(f"  [SKIP] Session too short for attack windows after buffers. Keeping as normal.")
        df["label"] = 0
        df["is_generated_spoof"] = 0
        df["attack_round_id"] = 0
        df["attack_type"] = "normal"
        df["attack_phase"] = "normal"
        df["transition_mode"] = "none"
        df["geofence_style"] = "none"
        return df, []

    print(f"  Attack rounds in this session: {num_rounds}")

    # Each session gets its own shuffled order of attack types, instead of
    # always following the same fixed sequence (still reproducible since rng
    # is seeded).
    session_plan = build_session_attack_plan(num_rounds, rng)
    print(f"  Attack order for this session: {session_plan}")

    attack_log = []

    for round_i, (start, end, duration) in enumerate(windows, start=1):
        attack_type = session_plan[round_i - 1]

        phases = split_phases(start, end)
        assign_phase_metadata(df, phases)

        if attack_type == "gradual_drag":
            apply_gradual_drag_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "freeze":
            apply_freeze_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "replay":
            apply_replay_attack(df, start, end, phases, round_i, rng)
        elif attack_type == "geofence_evasion":
            apply_geofence_evasion_attack(df, start, end, phases, round_i, rng, center_lat, center_lon)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # First make the coordinate path human-like, then recalculate motion.
        # This is the key step that makes distance_m, coord_speed,
        # speed_residual, and course_change more logical.
        limit_reported_motion_for_window(df, start, end, rng)

        for phase_name, (p_start, p_end) in phases.items():
            modify_signal_quality(df, range(p_start, p_end + 1), phase_name, rng)

        recalculate_motion_for_window(df, start, end, rng)

        attack_log.append({
            "session_id": df.loc[0, "session_id"],
            "attack_round_id": round_i,
            "attack_type": str(df.loc[start, "attack_type"]),
            "start_index": start,
            "end_index": end,
            "duration_records": duration,
            "capture_start": phases["capture"][0],
            "capture_end": phases["capture"][1],
            "takeover_start": phases["takeover"][0],
            "takeover_end": phases["takeover"][1],
            "stable_start": phases["stable"][0],
            "stable_end": phases["stable"][1],
            "release_start": phases["release"][0],
            "release_end": phases["release"][1],
            "transition_mode": str(df.loc[start, "transition_mode"]),
        })

    # Cleanup
    df["satellites_in_view"] = df["satellites_in_view"].round().astype(int)
    df["satellites_used"] = df["satellites_used"].round().astype(int)
    df["satellites_used"] = np.minimum(df["satellites_used"], df["satellites_in_view"])
    df["hdop"] = df["hdop"].round(2)
    df["velocity"] = df["velocity"].round(3)
    df["course"] = df["course"].round(2)

    if OPTIONAL_SAT_LOCKS_COL in df.columns:
        df[OPTIONAL_SAT_LOCKS_COL] = df[OPTIONAL_SAT_LOCKS_COL].round().astype(int)

    s_center_lat, s_center_lon = get_allowed_center(df)
    df = add_zone_metadata(df, s_center_lat, s_center_lon)

    return df, attack_log


def prepare_input_data_from_df(session_df):
    """Same as prepare_input_data but accepts a DataFrame directly (already one session)."""
    df = session_df.copy().reset_index(drop=True)
    df.columns = df.columns.str.strip()

    numeric_cols = [
        "latitude", "longitude", "velocity", "course",
        "satellites_in_view", "satellites_used", "hdop"
    ]
    if OPTIONAL_SAT_LOCKS_COL in df.columns:
        numeric_cols.append(OPTIONAL_SAT_LOCKS_COL)

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df = compute_time_deltas(df)
    df = fill_missing_course(df)

    df["true_latitude"] = df["latitude"]
    df["true_longitude"] = df["longitude"]
    df["label"] = 0
    df["is_generated_spoof"] = 0
    df["attack_round_id"] = 0
    df["attack_type"] = "normal"
    df["attack_phase"] = "normal"
    df["transition_mode"] = "none"
    df["geofence_style"] = "none"
    df["replay_source_start_index"] = np.nan

    return df


def generate_spoofed_dataset(input_file=INPUT_FILE):
    # Use a fixed RNG so results are reproducible across sessions.
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Load raw file and split by session ──────────────────────────
    raw = pd.read_csv(input_file)
    raw.columns = raw.columns.str.strip()

    if "session_id" not in raw.columns:
        raw["session_id"] = "session_0"

    sessions = raw["session_id"].unique()
    print(f"Found {len(sessions)} session(s): {list(sessions)}")

    all_processed = []
    all_attack_logs = []

    for sid in sessions:
        print(f"\n{'='*60}")
        print(f"Processing session: {sid}")
        session_df = raw[raw["session_id"] == sid].copy().reset_index(drop=True)
        print(f"  Rows: {len(session_df)}")

        # Use first point of each session as the allowed-zone center.
        center_lat = float(session_df["latitude"].iloc[0])
        center_lon = float(session_df["longitude"].iloc[0])

        processed_df, attack_log = _process_one_session(
            session_df, rng, center_lat, center_lon
        )

        # Restore correct session_id (prepare_input_data_from_df keeps it).
        processed_df["session_id"] = sid
        all_processed.append(processed_df)
        all_attack_logs.extend(attack_log)

        spoof_count = int((processed_df["label"] == 1).sum())
        print(f"  Spoofed rows generated: {spoof_count} / {len(processed_df)} "
              f"({spoof_count/len(processed_df)*100:.1f}%)")

    # ── Concatenate all sessions ────────────────────────────────────
    df = pd.concat(all_processed, ignore_index=True)

    print(f"\n{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Total spoofed: {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")

    # ── Save outputs ────────────────────────────────────────────────
    metadata_only_cols = [
        "true_latitude", "true_longitude", "is_generated_spoof",
        "attack_round_id", "attack_type", "attack_phase", "transition_mode",
        "geofence_style",
        "replay_source_start_index", "time_delta_sec", "_datetime",
        "true_distance_from_allowed_center_m",
        "reported_distance_from_allowed_center_m",
        "true_zone_status", "reported_zone_status", "geofence_evasion_case",
    ]

    # Metadata output (full detail, for analysis only)
    metadata_df = df.drop(columns=["_datetime"], errors="ignore")
    metadata_df.to_csv(METADATA_OUTPUT_FILE, index=False)

    # Training output (no leakage columns)
    training_df = df.drop(
        columns=[c for c in metadata_only_cols if c in df.columns], errors="ignore"
    )
    training_df.to_csv(TRAINING_OUTPUT_FILE, index=False)

    # Attack log
    attack_log_df = pd.DataFrame(all_attack_logs)
    attack_log_df.to_csv(ATTACK_LOG_FILE, index=False)

    create_plots(df)
    print_summary(df, all_attack_logs)

    print(f"\nSaved training dataset : {TRAINING_OUTPUT_FILE}")
    print(f"Saved metadata dataset : {METADATA_OUTPUT_FILE}")
    print(f"Saved attack log       : {ATTACK_LOG_FILE}")
    print(f"Saved plots folder     : {PLOTS_DIR}/")
    print("\nUse the training dataset for ML.")
    print("Use the metadata dataset only for explanation, plots, and per-attack evaluation.")
    print("Do NOT train on attack_type, attack_phase, attack_round_id, true_latitude, or true_longitude.")

    return training_df, metadata_df, attack_log_df


if __name__ == "__main__":
    generate_spoofed_dataset(INPUT_FILE)