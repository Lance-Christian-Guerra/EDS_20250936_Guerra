# main.py
import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# -------------------------
# Utility / I/O functions
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV {path}: {e}")

def save_csv(df: pd.DataFrame, path: str):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save CSV {path}: {e}")

# -------------------------
# Cleaning & preprocessing
# -------------------------
def convert_time_seconds_to_datetime(df: pd.DataFrame, t_col: str, start_iso: str) -> pd.DataFrame:
    df = df.copy()
    try:
        start_dt = pd.to_datetime(start_iso)
    except Exception as e:
        raise ValueError(f"Invalid start datetime {start_iso}: {e}")
    # ensure numeric seconds
    df[t_col] = pd.to_numeric(df[t_col], errors='coerce')
    df = df.dropna(subset=[t_col])
    df['timestamp'] = start_dt + pd.to_timedelta(df[t_col], unit='s')
    return df

def basic_cleaning(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    df = df.copy()
    # drop exact duplicates
    df = df.drop_duplicates()
    # coerce numeric columns
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # report and fill small gaps by time interpolation if timestamp present
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp').sort_index()
        # interpolate numeric columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both')
        df = df.reset_index()
    # drop columns with >30% missing
    thresh = 0.7 * len(df)
    df = df.dropna(axis=1, thresh=thresh)
    return df

# -------------------------
# Unique filter
# -------------------------
def apply_unique_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df2 = df.copy()
    start = pd.to_datetime(cfg['start_datetime'])
    end = pd.to_datetime(cfg['end_datetime'])
    df2 = df2[(df2['timestamp'] >= start) & (df2['timestamp'] <= end)]
    # motor_channel is a column name like 'I3'
    motor_ch = cfg.get('motor_channel')
    if motor_ch and motor_ch in df2.columns:
        # keep only rows where motor channel is finite
        df2 = df2[np.isfinite(df2[motor_ch])]
    return df2

# -------------------------
# Feature engineering & spike detection
# -------------------------
def compute_rolling_features(df: pd.DataFrame, motor_ch: str, window: int):
    df = df.copy()
    if motor_ch not in df.columns:
        raise KeyError(f"{motor_ch} not in dataframe columns")
    # rolling window in samples (centered)
    df['rolling_median'] = df[motor_ch].rolling(window=window, min_periods=1, center=True).median()
    df['rolling_std'] = df[motor_ch].rolling(window=window, min_periods=1, center=True).std().fillna(0.0)
    return df

def detect_spikes(df: pd.DataFrame, motor_ch: str, k: float):
    df = df.copy()
    threshold = df['rolling_median'] + k * df['rolling_std']
    df['is_spike'] = df[motor_ch] > threshold
    # group consecutive spikes into events
    df['spike_group'] = (df['is_spike'] != df['is_spike'].shift(1)).cumsum()
    spike_events = []
    for gid, g in df[df['is_spike']].groupby('spike_group'):
        start_ts = g['timestamp'].iloc[0]
        end_ts = g['timestamp'].iloc[-1]
        peak_current = g[motor_ch].max()
        duration_s = (end_ts - start_ts).total_seconds()
        spike_events.append({
            'group': int(gid),
            'start': str(start_ts),
            'end': str(end_ts),
            'peak_current': float(peak_current),
            'duration_s': float(duration_s),
            'samples': int(len(g))
        })
    return df, spike_events

# -------------------------
# Analysis
# -------------------------
def analyze_statistics(df: pd.DataFrame, motor_ch: str) -> dict:
    arr = df[motor_ch].dropna().to_numpy()
    if arr.size == 0:
        return {}
    stats_dict = {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)),
        'var': float(np.var(arr, ddof=1)),
        'skew': float(stats.skew(arr)),
        'kurtosis': float(stats.kurtosis(arr)),
        'iqr': float(stats.iqr(arr))
    }
    return stats_dict

# -------------------------
# Visualizations
# -------------------------
def plot_histogram(df: pd.DataFrame, motor_ch: str, out_path: str, threshold=None):
    plt.figure(figsize=(6,4))
    sns.histplot(df[motor_ch].dropna(), bins=60, kde=True, color='C0')
    if threshold is not None:
        plt.axvline(threshold, color='r', linestyle='--', label=f"threshold={threshold:.2f}")
        plt.legend()
    plt.xlabel(f"{motor_ch} (A)")
    plt.title(f"Histogram of {motor_ch}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_boxplot_groups(df: pd.DataFrame, motor_ch: str, out_path: str):
    df2 = df.copy()
    df2['group'] = np.where(df2['is_spike'], 'spike', 'normal')
    plt.figure(figsize=(6,4))
    sns.boxplot(x='group', y=motor_ch, data=df2)
    plt.title(f"Boxplot: {motor_ch} (spike vs normal)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, cols: list, out_path: str):
    corr = df[cols].corr(method='pearson')
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# Animation 1: time series with spike highlight
def animate_time_series(df: pd.DataFrame, motor_ch: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.set_xlabel('Time')
    ax.set_ylabel(f"{motor_ch} (A)")
    line, = ax.plot([], [], lw=1)
    spike_scatter = ax.scatter([], [], color='red', s=10)

    times = pd.to_datetime(df['timestamp'])
    y = df[motor_ch].ffill().to_numpy()
    x = np.arange(len(times))

    ax.set_xlim(0, len(x))
    ax.set_ylim(np.nanmin(y)*0.95, np.nanmax(y)*1.05)

    def init():
        line.set_data([], [])
        spike_scatter.set_offsets(np.empty((0, 2)))
        return line, spike_scatter

    def update(i):
        xi = x[:i]
        yi = y[:i]
        line.set_data(xi, yi)
        spikes_idx = np.where(df['is_spike'].iloc[:i].to_numpy())[0]
        if spikes_idx.size > 0:
            pts = np.column_stack((spikes_idx, y[spikes_idx]))
            spike_scatter.set_offsets(pts)
        else:
            spike_scatter.set_offsets(np.empty((0, 2)))
        return line, spike_scatter

    if not animation.writers.is_available('ffmpeg'):
        print(f"FFmpeg writer not available; skipping animation {out_path}")
        plt.close()
        return
    ani = animation.FuncAnimation(fig, update, frames=len(x), init_func=init, blit=True, interval=20)
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Auto'), bitrate=1800)
        ani.save(out_path, writer=writer)
    except RuntimeError:
        print(f"FFmpeg writer not available; skipping animation {out_path}")
    plt.close()

# Animation 2: moving-window histogram
def animate_moving_histogram(df: pd.DataFrame, motor_ch: str, out_path: str, window_samples=500, step=50):
    fig, ax = plt.subplots(figsize=(6,4))
    y = df[motor_ch].ffill().to_numpy()
    n_frames = max(1, (len(y) - window_samples) // step)
    def update(i):
        start = i*step
        end = start + window_samples
        ax.clear()
        ax.hist(y[start:end], bins=40, color='C0', alpha=0.8)
        ax.set_title(f"Window {start}:{end}")
        ax.set_xlabel(f"{motor_ch} (A)")
        ax.set_ylabel("Frequency")
    if not animation.writers.is_available('ffmpeg'):
        print(f"FFmpeg writer not available; skipping animation {out_path}")
        plt.close()
        return
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=200)
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=10, metadata=dict(artist='Auto'), bitrate=1200)
        ani.save(out_path, writer=writer)
    except RuntimeError:
        print(f"FFmpeg writer not available; skipping animation {out_path}")
    plt.close()

# -------------------------
# Pipeline runner
# -------------------------
def run_pipeline(cfg: dict):
    os.makedirs(cfg['out_dir'], exist_ok=True)
    start_time = time.perf_counter()

    print("Loading data...")
    df = load_data(cfg['input_csv'])

    print("Converting time column 't' to timestamps...")
    df = convert_time_seconds_to_datetime(df, t_col='t', start_iso=cfg['start_datetime'])

    numeric_cols = [c for c in df.columns if c not in ['timestamp']]
    print("Basic cleaning...")
    df = basic_cleaning(df, numeric_cols=numeric_cols)

    print("Applying unique filter...")
    df = apply_unique_filter(df, cfg)
    print(f"Rows after filter: {len(df)}")

    motor_ch = cfg['motor_channel']
    print("Computing rolling features...")
    window = cfg['spike_detection']['rolling_window_samples']
    df = compute_rolling_features(df, motor_ch, window=window)

    print("Detecting spikes...")
    k = cfg['spike_detection']['k']
    df, spike_events = detect_spikes(df, motor_ch, k=k)
    print(f"Detected {len(spike_events)} spike events")

    print("Analyzing statistics...")
    stats = analyze_statistics(df, motor_ch)
    stats['spike_event_count'] = len(spike_events)

    # Save cleaned CSV and analysis
    save_csv(df, cfg['cleaned_csv'])
    with open(os.path.join(cfg['out_dir'],'analysis_results.json'),'w') as f:
        json.dump({'stats':stats, 'spike_events': spike_events}, f, indent=2, default=str)

    # Visualizations
    print("Generating static plots...")
    threshold_vals = df['rolling_median'] + k * df['rolling_std']
    median_threshold = float(np.nanmedian(threshold_vals))
    plot_histogram(df, motor_ch, os.path.join(cfg['out_dir'],'hist_current.png'), threshold=median_threshold)
    plot_boxplot_groups(df, motor_ch, os.path.join(cfg['out_dir'],'boxplot_spike_vs_normal.png'))
    # correlation heatmap using a subset of numeric cols
    corr_cols = [motor_ch] + [c for c in ['dq1','dq2','dq3','q1','q2','q3','ddq1','ddq2','ddq3'] if c in df.columns]
    plot_correlation_heatmap(df, corr_cols, os.path.join(cfg['out_dir'],'corr_heatmap.png'))

    print("Generating animations (this may take a while)...")
    animate_time_series(df, motor_ch, os.path.join(cfg['out_dir'],'timeseries_spikes.mp4'))
    animate_moving_histogram(df, motor_ch, os.path.join(cfg['out_dir'],'moving_histogram.mp4'))

    end_time = time.perf_counter()
    print(f"Pipeline finished in {end_time - start_time:.2f} seconds")
    return {'stats': stats, 'spike_events': spike_events}

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    cfg = {
      "input_csv": "data/dataset_original.csv",
      "cleaned_csv": "data/dataset_cleaned.csv",
      "out_dir": "outputs",
      "motor_channel": "I3",
      "start_datetime": "2006-09-15T08:00:00",
      "end_datetime": "2006-11-20T00:00:00",
      "spike_detection": {
        "method": "rolling_adaptive",
        "rolling_window_samples": 101,
        "k": 4
      },
      "resample_rate_hz": None,
      "visualization_formats": ["png","mp4","html"]
    }
    results = run_pipeline(cfg)
    print(json.dumps(results, indent=2))
