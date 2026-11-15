#!/usr/bin/env python3
"""
beat_analysis.py - variable-tempo beat / onset / intensity analysis with Librosa.

Takes an arbitrary audio file and exports a JSON "musical timeline" (beats, tempo map,
downbeats, subdivisions, onsets, RMS intensity) for a downstream procedural
enemy-generation system.  It does NOT generate enemies or any gameplay.

Usage
-----
    python beat_analysis.py song.mp3 -o song.analysis.json
    python beat_analysis.py song.wav --min-bpm 70 --max-bpm 180 --start-bpm 128
    python beat_analysis.py song.flac --beats-per-bar 3      # force 3/4
    python beat_analysis.py song.ogg --no-meter              # beats only, no bar numbers

Library use
-----------
    from beat_analysis import analyze_audio, AnalysisConfig
    result = analyze_audio("song.mp3", AnalysisConfig(start_bpm=128))

Pipeline overview
-------------------------------------------------------------------------------
 1. Load mono audio at a fixed analysis rate (default 22.05 kHz). All times are derived
    from sample counts at that rate, so they are seconds from the start of the ORIGINAL file.
 2. Onset-strength envelope from a log-mel spectrogram (SuperFlux-style max filter).
 3. Continuous tempo tracking: autocorrelation tempogram -> log-spaced BPM grid ->
    Viterbi decoding (sticky but gradual drift allowed, sudden jumps allowed at a cost).
 4. Dynamic-programming beat tracker (Ellis 2007) re-implemented so the target period can
    vary frame by frame. (librosa.beat.beat_track only accepts ONE tempo.)
 5. Post-processing: silence removal, weak edge trimming, sub-frame time refinement.
 6. Tempo map: optimal piecewise-constant segmentation of the log-BPM of the tracked beats.
 7. Meter / downbeats: beat-synchronous accent features (onset, bass onset, harmonic
    change) -> meter periodicity test -> Viterbi bar-phase decoding with phase slips.
 8. Subdivision (8ths / triplets / 16ths) detection from onset energy on sub-beat grids.
 9. Onsets, RMS intensity, per-beat confidence, JSON export.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sps

import librosa

SCHEMA_VERSION = "1.0"
log = logging.getLogger("beat_analysis")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    # --- audio / feature extraction ---
    sr: int = 22050                 # analysis sample rate (audio is resampled to this)
    hop_length: int = 256           # ~11.6 ms per frame at 22.05 kHz
    n_fft: int = 1024
    n_mels: int = 128
    beat_log_gamma: float = 10.0    # log1p compression strength of the beat-tracking envelope

    # --- tempo tracking ---
    min_bpm: float = 55.0
    max_bpm: float = 215.0
    start_bpm: float = 120.0        # centre of the log-normal tempo prior (octave disambiguation)
    prior_sigma_octaves: float = 1.0
    tempo_window_sec: float = 6.0   # tempogram window: shorter = reacts faster, less precise
    tempo_step_sec: float = 0.2     # Viterbi time step for the tempo track
    tempo_bins_per_octave: int = 72
    tempo_obs_power: float = 2.0    # sharpens tempogram evidence
    tempo_transition_sigma_oct: float = 0.02   # allowed drift per step (gradual changes)
    tempo_jump_prob: float = 2e-3   # probability mass for sudden jumps per step

    # --- beat tracking (DP) ---
    tightness: float = 60.0         # higher = beats stick to the tempo, lower = follow onsets
    trim_weak_edges: bool = True

    # --- silence / pauses ---
    silence_db: float = -50.0       # RMS (dB re. loudest frame) below this counts as digital silence
    pause_db_below_typical: float = 24.0   # quiet AND onset-free: this many dB under typical loudness
    min_silence_sec: float = 0.35   # shorter dips are ignored
    rhythm_floor_rel: float = 0.04  # "no rhythmic activity": 2 s without onset flux above this x p99 ...
    rhythm_floor_abs: float = 1e-3  # ... or above this absolute level (beat-envelope units)

    # --- tempo map ---
    tempo_change_min_ratio: float = 0.04   # smallest relative BPM change worth reporting
    tempo_change_min_beats: int = 8        # a change must persist for roughly this many beats
    tempo_sudden_ratio: float = 0.10       # >= this relative jump is labelled "sudden"

    # --- meter ---
    detect_meter: bool = True
    beats_per_bar: int | None = None       # force a meter (None = estimate)
    meter_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    meter_min_strength: float = 0.20       # minimum periodicity strength (also needs >3.5 std. errors)
    meter_min_confident_beat_fraction: float = 0.6   # skip meter if most beats are low-confidence
    downbeat_slip_penalty: float = 4.0     # cost of changing bar phase (meter change / pause)
    detect_meter_changes: bool = True      # look for sections with a different time signature
    meter_window_beats: int = 32
    meter_switch_penalty: float = 1.5      # evidence needed (in summed window strengths) to switch

    # --- subdivisions ---
    detect_subdivisions: bool = True
    subdivision_threshold: float = 0.30

    # --- onsets ---
    onset_delta: float = 0.08

    # --- confidence ---
    low_confidence_threshold: float = 0.30
    periodicity_floor: float = 0.15   # normalised autocorrelation of pure noise is ~0.1
    periodicity_full: float = 0.50    # ... and of clearly rhythmic music is >= ~0.5
    drop_low_confidence: bool = False

    # --- output ---
    intensity_hz: float = 20.0
    float_decimals: int = 4


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _round(x: Any, nd: int) -> Any:
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    return x


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med))
    if mad < 1e-9:
        s = x.std()
        return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s
    return np.clip((x - med) / mad, -4.0, 4.0)


def _filter_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs of at least ``min_len`` samples."""
    lab, n = ndi.label(mask)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n + 1):
        run = lab == i
        if run.sum() >= min_len:
            out |= run
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """List of (start, end_exclusive) for the True-runs of a boolean array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _peak_near(env: np.ndarray, frames: np.ndarray, w: int = 2) -> np.ndarray:
    out = np.zeros(len(frames))
    n = len(env)
    for i, f in enumerate(frames):
        lo, hi = max(int(f) - w, 0), min(int(f) + w + 1, n)
        out[i] = env[lo:hi].max() if hi > lo else 0.0
    return out


def _parabolic_peak(env: np.ndarray, f: int) -> float:
    """Sub-frame location of the peak near frame f (returns a fractional frame index)."""
    n = len(env)
    lo, hi = max(f - 1, 0), min(f + 1, n - 1)
    f0 = lo + int(np.argmax(env[lo:hi + 1]))
    if 0 < f0 < n - 1:
        y0, y1, y2 = env[f0 - 1], env[f0], env[f0 + 1]
        den = y0 - 2 * y1 + y2
        if den < -1e-9:
            return f0 + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    return float(f0)


# --------------------------------------------------------------------------------------
# 1-2. Loading and features
# --------------------------------------------------------------------------------------
def load_audio(path: str, cfg: AnalysisConfig) -> tuple[np.ndarray, int, float, int]:
    try:
        orig_sr = int(librosa.get_samplerate(path))
    except Exception:  # pragma: no cover - some formats cannot be probed
        orig_sr = 0
    y, sr = librosa.load(path, sr=cfg.sr, mono=True)
    duration = len(y) / float(sr)
    return y, sr, duration, orig_sr


def compute_features(y: np.ndarray, cfg: AnalysisConfig) -> dict[str, Any]:
    hop = cfg.hop_length
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=hop, n_mels=cfg.n_mels, fmax=cfg.sr / 2
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # center=False: the (centred) STFT frame t is already at time t*hop/sr. Verified on synthetic
    # clicks: peaks fall within ~1 frame (12 ms) of the true transient. max_size=3 is the
    # SuperFlux max-filter, which suppresses false onsets from vibrato / tremolo.
    oenv = librosa.onset.onset_strength(
        S=S_db, sr=cfg.sr, hop_length=hop, lag=1, max_size=3, center=False
    )

    # Beat-tracking envelope. dB-compressed flux treats a faint broadband hi-hat like a loud kick
    # (and averaging over mel bands then favours the hat), which made the tracker lock onto
    # off-beats. log1p compression keeps loudness information; the max filter over frequency
    # (SuperFlux) suppresses vibrato. The dB envelope above stays in use for onsets/subdivisions,
    # where sensitivity to quiet events is wanted.
    L = np.log1p(cfg.beat_log_gamma * S / max(float(S.max()), 1e-12))
    Lmax = ndi.maximum_filter(L, size=(3, 1), mode="nearest")
    flux = np.maximum(0.0, L[:, 1:] - Lmax[:, :-1])
    oenv_beat = np.concatenate([[0.0], flux.mean(axis=0)])

    # Low-frequency (<= 200 Hz) onset strength: kick / bass accents, used for downbeats.
    freqs = librosa.mel_frequencies(n_mels=cfg.n_mels, fmin=0.0, fmax=cfg.sr / 2)
    low = freqs <= 200.0
    if low.sum() >= 2:
        oenv_low = librosa.onset.onset_strength(
            S=S_db[low], sr=cfg.sr, hop_length=hop, lag=1, max_size=1, center=False
        )
    else:  # pragma: no cover
        oenv_low = oenv.copy()

    # Same band, but in LINEAR amplitude. dB-compressed flux hides loudness differences between
    # a strong kick and a weak one; the linear version keeps them, which matters for downbeats.
    low_amp = np.sqrt(S[low].sum(axis=0)) if low.sum() >= 2 else np.sqrt(S.sum(axis=0))
    lin_low = np.maximum(0.0, np.diff(low_amp, prepend=low_amp[0]))

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    n = min(len(oenv), len(oenv_low), len(rms), len(lin_low), len(oenv_beat))
    oenv, oenv_low, rms, lin_low = oenv[:n], oenv_low[:n], rms[:n], lin_low[:n]
    oenv_beat = oenv_beat[:n]

    peak = max(float(rms.max()), 1e-12)
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10) / peak)  # dB relative to loudest frame

    return dict(
        oenv=oenv.astype(float),
        oenv_beat=oenv_beat.astype(float),
        oenv_low=oenv_low.astype(float),
        lin_low=lin_low.astype(float),
        rms=rms.astype(float),
        rms_db=rms_db,
        frame_rate=cfg.sr / hop,
        n_frames=n,
        peak_rms=peak,
    )


def frame_to_time(frames: np.ndarray | float, cfg: AnalysisConfig) -> np.ndarray | float:
    """Frame index (may be fractional) -> seconds from start of the ORIGINAL audio."""
    return np.asarray(frames, dtype=float) * cfg.hop_length / cfg.sr


# --------------------------------------------------------------------------------------
# 3. Continuous tempo tracking
# --------------------------------------------------------------------------------------
def track_tempo(oenv: np.ndarray, cfg: AnalysisConfig, frame_rate: float,
                inactive: np.ndarray | None = None) -> dict[str, Any]:
    """
    Per-frame tempo estimate using an autocorrelation tempogram + Viterbi.

    The Viterbi transition model mixes (a) a narrow Gaussian over log-tempo, which allows slow
    accelerations/decelerations, and (b) a small uniform floor, which allows sudden jumps at a
    fixed cost. A new tempo therefore has to be supported by enough evidence to pay for the jump,
    so noise does not produce phantom tempo changes.
    """
    n_frames = len(oenv)
    win = max(int(round(cfg.tempo_window_sec * frame_rate)), 64)
    # Remove the slowly varying mean first. Otherwise the DC offset of a (non-negative) onset
    # envelope makes every lag look "periodic" and the normalised autocorrelation of pure noise
    # comes out near 0.8 instead of ~0.
    detr = oenv - ndi.uniform_filter1d(oenv, size=max(int(2.0 * frame_rate), 3), mode="nearest")
    tg = librosa.feature.tempogram(
        onset_envelope=detr, sr=cfg.sr, hop_length=cfg.hop_length, win_length=win
    )  # (win, n_frames), column-normalised so lag 0 == 1  ->  values are normalised autocorrelation

    # Remove the triangular bias that windowed autocorrelation has towards short lags.
    w = sps.get_window("hann", win, fftbins=True)
    wac = np.correlate(w, w, mode="full")[win - 1:]
    wac = wac / wac[0]
    tg = np.clip(tg / np.maximum(wac[:, None], 0.3), 0.0, 1.0)

    # Time steps for Viterbi
    step = max(1, int(round(cfg.tempo_step_sec * frame_rate)))
    cols = np.arange(0, n_frames, step)
    tgc = tg[:, cols]

    # Log-spaced BPM grid, linearly interpolated from the lag axis
    n_oct = np.log2(cfg.max_bpm / cfg.min_bpm)
    n_bins = int(round(n_oct * cfg.tempo_bins_per_octave)) + 1
    grid_bpm = cfg.min_bpm * 2.0 ** (np.arange(n_bins) / cfg.tempo_bins_per_octave)
    lag_f = 60.0 * cfg.sr / (cfg.hop_length * grid_bpm)
    ok = (lag_f >= 1.0) & (lag_f <= win - 2)
    grid_bpm, lag_f = grid_bpm[ok], lag_f[ok]
    n_bins = len(grid_bpm)
    lo = np.floor(lag_f).astype(int)
    frac = lag_f - lo
    tgg = tgc[lo, :] * (1.0 - frac)[:, None] + tgc[lo + 1, :] * frac[:, None]

    # Prior over tempo (resolves octave ambiguity softly)
    prior = np.exp(-0.5 * (np.log2(grid_bpm / cfg.start_bpm) / cfg.prior_sigma_octaves) ** 2)
    obs = (np.maximum(tgg, 1e-6) * prior[:, None]) ** cfg.tempo_obs_power

    # Frames with (almost) no rhythmic energy carry no tempo information -> flat observation
    k = max(int(round(2.0 * frame_rate)), 3)
    local_energy = ndi.uniform_filter1d(oenv, size=k, mode="nearest")
    thr = 0.05 * np.percentile(local_energy, 90) if n_frames else 0.0
    inactive_steps = local_energy[cols] < thr
    if inactive is not None:
        inactive_steps = inactive_steps | inactive[cols]
    obs[:, inactive_steps] = 1.0
    obs = obs / obs.sum(axis=0, keepdims=True)

    # Transition matrix on the log grid
    idx = np.arange(n_bins)
    d_oct = (idx[:, None] - idx[None, :]) / cfg.tempo_bins_per_octave * (
        np.log2(grid_bpm[-1] / grid_bpm[0]) / ((n_bins - 1) / cfg.tempo_bins_per_octave)
    )
    T = np.exp(-0.5 * (d_oct / cfg.tempo_transition_sigma_oct) ** 2)
    T /= T.sum(axis=1, keepdims=True)
    T = (1.0 - cfg.tempo_jump_prob) * T + cfg.tempo_jump_prob / n_bins

    path = librosa.sequence.viterbi(obs, T)
    bpm_steps = grid_bpm[path]
    period_steps = tgg[path, np.arange(len(cols))]
    period_steps = np.where(inactive_steps, 0.0, period_steps)  # periodicity (confidence) track

    frames = np.arange(n_frames)
    bpm_frames = 2.0 ** np.interp(frames, cols, np.log2(bpm_steps))
    periodicity_frames = np.interp(frames, cols, period_steps)

    return dict(
        bpm=bpm_frames,
        periodicity=periodicity_frames,
        steps=cols,
        bpm_steps=bpm_steps,
        periodicity_steps=period_steps,
        inactive_steps=inactive_steps,
    )


# --------------------------------------------------------------------------------------
# 4. Variable-period dynamic-programming beat tracker
# --------------------------------------------------------------------------------------
def dp_beat_track(local: np.ndarray, period: np.ndarray, tightness: float) -> np.ndarray:
    """
    Ellis (2007) dynamic-programming beat tracker with a time-varying target period.

    cum[t] = local[t] + max_i ( cum[i] - tightness * log((t-i)/period_mid)^2 )

    ``period`` is the expected beat period in frames at every frame; the expected interval
    between candidates i -> t uses the mean of period[i] and period[t], so the interval target
    changes smoothly across tempo changes.
    """
    n = len(local)
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=int)
    for t in range(n):
        p = period[t]
        lo = max(int(np.floor(t - 2.0 * p)), 0)
        hi = min(int(np.ceil(t - 0.5 * p)), t - 1)
        if hi < lo:
            cum[t] = local[t]
            continue
        idx = np.arange(lo, hi + 1)
        pm = 0.5 * (period[idx] + p)
        s = cum[idx] - tightness * np.log((t - idx) / pm) ** 2
        j = int(np.argmax(s))
        if s[j] > 0:
            cum[t] = local[t] + s[j]
            back[t] = idx[j]
        else:
            cum[t] = local[t]

    # Pick the last beat: last local maximum of cum that is not negligible
    maxes = np.zeros(n, dtype=bool)
    maxes[1:-1] = (cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])
    if n > 1:
        maxes[-1] = cum[-1] > cum[-2]
    if not maxes.any():
        return np.array([], dtype=int)
    med = np.median(cum[maxes])
    cand = np.flatnonzero(maxes & (cum * 2.0 > med))
    if cand.size == 0:
        return np.array([], dtype=int)
    tail = int(cand[-1])
    beats = [tail]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    return np.array(beats[::-1], dtype=int)


def _trim_weak_edges(frames: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Drop leading / trailing beats that sit on very weak onset energy (e.g. beatless intros)."""
    if len(frames) < 5:
        return frames
    smooth = np.convolve(strength, np.hanning(5), mode="same")
    thr = 0.5 * np.sqrt(np.mean(smooth ** 2))
    good = np.flatnonzero(strength > thr)
    if good.size == 0:
        return frames[:0]
    return frames[good[0]: good[-1] + 1]


def _drop_short_runs(frames: np.ndarray, frame_rate: float, min_beats: int = 4,
                     min_sec: float = 2.0) -> np.ndarray:
    """Remove isolated beat runs (separated by pauses) that are too short to be a real groove."""
    if len(frames) < 2:
        return frames[:0] if len(frames) < min_beats else frames
    d = np.diff(frames)
    med = float(np.median(d))
    cuts = np.flatnonzero(d > 1.8 * med) + 1
    keep = []
    for run in np.split(frames, cuts):
        if len(run) >= min_beats and (run[-1] - run[0]) / frame_rate >= min_sec:
            keep.append(run)
    return np.concatenate(keep) if keep else frames[:0]


# --------------------------------------------------------------------------------------
# 6. Tempo segmentation (tempo map)
# --------------------------------------------------------------------------------------
def segment_piecewise_constant(x: np.ndarray, beta: float, min_len: int) -> list[tuple[int, int]]:
    """Optimal partitioning (exact DP, O(n^2)) of x into constant-mean segments; cost = SSE + beta/segment."""
    n = len(x)
    if n < 2 * min_len:
        return [(0, n)]
    c1 = np.concatenate([[0.0], np.cumsum(x)])
    c2 = np.concatenate([[0.0], np.cumsum(x * x)])
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        i = np.arange(0, j - min_len + 1)
        L = j - i
        s = c1[j] - c1[i]
        s2 = c2[j] - c2[i]
        tot = F[i] + (s2 - s * s / L) + beta
        k = int(np.argmin(tot))
        F[j] = tot[k]
        last[j] = i[k]
    segs, j = [], n
    while j > 0:
        i = int(last[j])
        segs.append((i, j))
        j = i
    return segs[::-1]


# --------------------------------------------------------------------------------------
# 7. Meter and downbeats
# --------------------------------------------------------------------------------------
def _chroma_change(
    beat_times: np.ndarray, chroma: np.ndarray, sr: int, hop_c: int, typical_interval: float
) -> np.ndarray:
    """Harmonic change at each beat: cosine distance between chroma just before and just after."""
    n_c = chroma.shape[1]
    cs = np.concatenate([np.zeros((chroma.shape[0], 1)), np.cumsum(chroma, axis=1)], axis=1)

    def mean_win(t0: float, t1: float) -> np.ndarray:
        a = int(np.clip(round(t0 * sr / hop_c), 0, n_c - 1))
        b = int(np.clip(round(t1 * sr / hop_c), a + 1, n_c))
        return (cs[:, b] - cs[:, a]) / (b - a)

    out = np.zeros(len(beat_times))
    for i, t in enumerate(beat_times):
        prev_i = (t - beat_times[i - 1]) if i > 0 else typical_interval
        next_i = (beat_times[i + 1] - t) if i + 1 < len(beat_times) else typical_interval
        prev_i = min(prev_i, 1.5 * typical_interval)
        next_i = min(next_i, 1.5 * typical_interval)
        pre = mean_win(t - prev_i, t - 0.05 * prev_i)
        post = mean_win(t + 0.10 * next_i, t + next_i)
        na, nb = np.linalg.norm(pre), np.linalg.norm(post)
        out[i] = 0.0 if na < 1e-9 or nb < 1e-9 else 1.0 - float(pre @ post) / (na * nb)
    return out


def beat_accents(
    y: np.ndarray,
    beat_frames: np.ndarray,
    beat_times: np.ndarray,
    feats: dict[str, Any],
    cfg: AnalysisConfig,
) -> np.ndarray:
    """Per-beat 'accent' score = onset + bass onset (dB and linear) + harmonic change (robust z-scored)."""
    a_onset = _peak_near(feats["oenv_beat"], beat_frames, 2)
    a_low = _peak_near(feats["oenv_low"], beat_frames, 2)
    a_lin = _peak_near(feats["lin_low"], beat_frames, 2)
    hop_c = 512
    chroma = librosa.feature.chroma_stft(y=y, sr=cfg.sr, n_fft=2048, hop_length=hop_c)
    typical = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    a_chroma = _chroma_change(beat_times, chroma, cfg.sr, hop_c, typical)
    acc = (0.5 * _robust_z(a_onset) + 0.5 * _robust_z(a_low) + 1.0 * _robust_z(a_lin)
           + 0.7 * _robust_z(a_chroma))
    s = acc.std()
    return (acc - acc.mean()) / s if s > 1e-9 else np.zeros_like(acc)


_METER_PRIOR = {2: 0.9, 3: 1.0, 4: 1.0, 5: 0.8, 6: 0.9, 7: 0.8}


def _meter_strengths(accent: np.ndarray, candidates: tuple[int, ...], max_lag: int) -> dict[int, float]:
    """P(m) = mean autocorrelation at multiples of m  -  mean autocorrelation at all other lags."""
    n = len(accent)
    a = accent - accent.mean()
    denom = float((a * a).mean())
    if n < 12 or denom < 1e-9:
        return {}
    ac = np.zeros(max_lag + 1)
    for k in range(1, min(max_lag, n - 4) + 1):
        ac[k] = float((a[:-k] * a[k:]).mean()) / denom
    lags = np.arange(1, max_lag + 1)
    out: dict[int, float] = {}
    for m in candidates:
        if n < 3 * m or m > max_lag // 2:
            continue
        out[m] = float(ac[lags[lags % m == 0]].mean() - ac[lags[lags % m != 0]].mean())
    return out


def _meter_significant(strength: float, m: int, n: int, max_lag: int) -> bool:
    """Is a periodicity strength larger than 3.5 standard errors (null: white accent noise)?"""
    n_mult = max_lag // m
    se_p = (1.0 / np.sqrt(n)) * np.sqrt(1.0 / max(n_mult, 1) + 1.0 / max(max_lag - n_mult, 1))
    return strength > 3.5 * se_p


def estimate_meter(accent: np.ndarray, cfg: AnalysisConfig) -> tuple[int | None, float, dict[int, float]]:
    """Global beats-per-bar from the periodicity of the per-beat accent sequence."""
    max_lag = 16
    strengths = _meter_strengths(accent, cfg.meter_candidates, max_lag)
    if not strengths:
        return None, 0.0, {}
    scored = {m: s * _METER_PRIOR.get(m, 0.7) for m, s in strengths.items()}
    best = max(scored, key=scored.get)
    ok = _meter_significant(strengths[best], best, len(accent), max_lag)
    conf = float(np.clip(strengths[best] / 0.5, 0.0, 1.0)) if ok else 0.0
    return best, conf, strengths


def meter_sections(accent: np.ndarray, global_m: int, cfg: AnalysisConfig) -> list[tuple[int, int, int]]:
    """
    Detect sections with a different time signature.

    Sliding windows of ``meter_window_beats`` beats each vote for the meter whose periodicity is
    significantly strongest; a Viterbi pass over the windows (switch penalty + a small bonus for
    the global meter) turns the votes into contiguous sections. Without strong, sustained
    evidence this returns a single section covering all beats, i.e. nothing is invented.
    Section boundaries are only accurate to roughly +-half a window (a couple of bars).
    Returns a list of (first_beat, end_beat_exclusive, beats_per_bar).
    """
    n, win, hop, max_lag = len(accent), cfg.meter_window_beats, 4, 12
    cands = [m for m in cfg.meter_candidates if m <= max_lag // 2]
    if n < 2 * win or global_m not in cands + [global_m]:
        return [(0, n, global_m)]
    starts = np.arange(0, n - win + 1, hop)
    emis = np.zeros((len(starts), len(cands)))
    for wi, s in enumerate(starts):
        st = _meter_strengths(accent[s:s + win], tuple(cands), max_lag)
        for ci, m in enumerate(cands):
            if m in st and _meter_significant(st[m], m, win, max_lag):
                emis[wi, ci] = st[m] * _METER_PRIOR.get(m, 0.7)
    # hysteresis: prefer the global meter when evidence is ambiguous
    if global_m in cands:
        emis[:, cands.index(global_m)] += 0.15
    K = len(cands)
    V = np.zeros((len(starts), K))
    ptr = np.zeros((len(starts), K), dtype=int)
    V[0] = emis[0]
    for wi in range(1, len(starts)):
        stay = V[wi - 1]
        arg = int(stay.argmax())
        sw = stay[arg] - cfg.meter_switch_penalty
        use_stay = stay >= sw
        V[wi] = emis[wi] + np.where(use_stay, stay, sw)
        ptr[wi] = np.where(use_stay, np.arange(K), arg)
    lab = np.zeros(len(starts), dtype=int)
    lab[-1] = int(V[-1].argmax())
    for wi in range(len(starts) - 1, 0, -1):
        lab[wi - 1] = ptr[wi, lab[wi]]
    # window label -> beat label (nearest window centre)
    centres = starts + win / 2.0
    beat_lab = np.array([cands[lab[int(np.argmin(np.abs(centres - (i + 0.5))))]] for i in range(n)])
    # runs, merging runs shorter than 12 beats into their predecessor
    runs: list[list[int]] = []
    for i, m in enumerate(beat_lab):
        if runs and runs[-1][2] == m:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1, int(m)])
    merged: list[list[int]] = []
    for r in runs:
        if merged and (r[1] - r[0] < 12 or merged[-1][2] == r[2]):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < 12:
        merged[1][0] = 0
        merged.pop(0)
    return [(a, b, m) for a, b, m in merged]


def decode_bar_phase(accent: np.ndarray, m: int, slip_penalty: float) -> np.ndarray:
    """
    Viterbi over 'position in bar' states 0..m-1 (0 = downbeat).
    Advancing 1 state per beat is free; jumping to another phase costs ``slip_penalty`` - this
    lets bar lines re-align after a pause, an odd-length bar, or a meter change.
    """
    n = len(accent)
    emis = np.empty((n, m))
    emis[:, 0] = accent
    emis[:, 1:] = (-accent / (m - 1))[:, None]
    V = np.empty((n, m))
    ptr = np.zeros((n, m), dtype=int)
    V[0] = emis[0]
    ar = np.arange(m)
    for i in range(1, n):
        prev = V[i - 1]
        adv = np.roll(prev, 1)
        arg = int(prev.argmax())
        slip = prev[arg] - slip_penalty
        use_adv = adv >= slip
        V[i] = emis[i] + np.where(use_adv, adv, slip)
        ptr[i] = np.where(use_adv, (ar - 1) % m, arg)
    states = np.zeros(n, dtype=int)
    states[-1] = int(V[-1].argmax())
    for i in range(n - 1, 0, -1):
        states[i - 1] = ptr[i, states[i]]
    return states


def local_downbeat_confidence(accent: np.ndarray, is_down: np.ndarray, m: np.ndarray) -> np.ndarray:
    """How much stronger downbeats are than other beats in a window of +-3 bars around each beat."""
    n = len(accent)
    out = np.zeros(n)
    for i in range(n):
        half = 3 * int(m[i])
        lo, hi = max(0, i - half), min(n, i + half + 1)
        d = is_down[lo:hi]
        if d.sum() < 1 or (~d).sum() < 1:
            continue
        a = accent[lo:hi]
        out[i] = np.clip((a[d].mean() - a[~d].mean()) / 1.5, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------------------
# 8. Subdivisions
# --------------------------------------------------------------------------------------
def detect_subdivisions(
    beat_times: np.ndarray,
    oenv: np.ndarray,
    onset_ref: np.ndarray,
    frame_rate: float,
    cfg: AnalysisConfig,
) -> list[dict[str, Any]]:
    """
    For every beat interval, measure onset energy at the 1/2, 1/3-2/3 and 1/4-3/4 grid points.
    Evidence is averaged over +-2 beats (musical texture is persistent) before deciding.
    Hits are normalised by a local reference (90th percentile of onset-envelope peaks nearby).
    """
    n = len(beat_times)
    if n < 3:
        return [dict(division=None, label=None, confidence=0.0, times=[]) for _ in range(n)]
    ivals = np.diff(beat_times)
    ivals = np.append(ivals, ivals[-1])
    tol = 2

    def strength_at(t: float, ref: float) -> float:
        f = int(round(t * frame_rate))
        lo, hi = max(f - tol, 0), min(f + tol + 1, len(oenv))
        if hi <= lo or ref <= 1e-9:
            return 0.0
        return float(min(1.0, oenv[lo:hi].max() / ref))

    E = np.zeros(n)
    Tr = np.zeros(n)
    S16 = np.zeros(n)
    for i in range(n):
        t0, iv, ref = beat_times[i], ivals[i], onset_ref[i]
        E[i] = strength_at(t0 + iv / 2, ref)
        Tr[i] = min(strength_at(t0 + iv / 3, ref), strength_at(t0 + 2 * iv / 3, ref))
        S16[i] = min(strength_at(t0 + iv / 4, ref), strength_at(t0 + 3 * iv / 4, ref))
    k = np.ones(5) / 5.0
    Es, Ts, Ss = (np.convolve(v, k, mode="same") for v in (E, Tr, S16))

    th = cfg.subdivision_threshold
    out = []
    for i in range(n):
        div, label, conf = None, None, 0.0
        if Ts[i] > th and Es[i] < 0.7 * Ts[i]:
            div, label, conf = 3, "triplet", Ts[i]
        elif Ss[i] > th:
            div, label, conf = 4, "sixteenth", Ss[i]
        elif Es[i] > th:
            div, label, conf = 2, "eighth", Es[i]
        times = [] if div is None else [float(beat_times[i] + k_ * ivals[i] / div) for k_ in range(1, div)]
        out.append(dict(division=div, label=label, confidence=float(conf), times=times))
    return out


# --------------------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------------------
def analyze_audio(path: str, cfg: AnalysisConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AnalysisConfig()
    nd = cfg.float_decimals
    warnings: list[str] = []

    y, sr, duration, orig_sr = load_audio(path, cfg)
    if len(y) < sr:  # < 1 s
        raise ValueError("Audio is shorter than 1 second; nothing meaningful to analyse.")

    feats = compute_features(y, cfg)
    oenv, rms, rms_db = feats["oenv"], feats["rms"], feats["rms_db"]
    benv = feats["oenv_beat"]          # beat-tracking envelope (see compute_features)
    fr = feats["frame_rate"]
    n_frames = feats["n_frames"]
    hop = cfg.hop_length

    # (a) near-digital silence, or (b) a pause: much quieter than the track's typical level AND no
    # onsets (this catches pauses that still contain a noise floor, e.g. live / vinyl / dithering).
    typical_db = float(np.percentile(rms_db, 90))
    k_win = max(int(round(cfg.min_silence_sec * fr)) | 1, 3)
    onset_activity = ndi.maximum_filter1d(oenv, size=k_win, mode="nearest")
    quiet_no_onsets = (rms_db < typical_db - cfg.pause_db_below_typical) & (
        onset_activity < 0.15 * np.percentile(oenv, 95))
    silent = (rms_db < cfg.silence_db) | quiet_no_onsets
    silent = _filter_runs(silent, int(round(cfg.min_silence_sec * fr)))
    if feats["peak_rms"] < 1e-4:
        warnings.append("Audio is almost silent; results are unreliable.")

    # (c) no rhythmic activity at all (beatless intro, sustained pad / tone): beats are not guessed
    floor = max(cfg.rhythm_floor_rel * float(np.percentile(benv, 99)), cfg.rhythm_floor_abs)
    k2 = max(int(round(2.0 * fr)) | 1, 3)
    arhythmic = ndi.maximum_filter1d(benv, size=k2, mode="nearest") < floor
    no_beats_zone = silent | arhythmic

    # ---- tempo track + beat tracking -------------------------------------------------
    tempo = track_tempo(benv, cfg, fr, inactive=no_beats_zone)
    period = 60.0 * fr / tempo["bpm"]

    sd = benv.std()
    local = ndi.gaussian_filter1d(benv / (sd if sd > 1e-9 else 1.0), 1.0)
    beat_frames = dp_beat_track(local, period, cfg.tightness)

    if len(beat_frames):
        beat_frames = beat_frames[~no_beats_zone[np.clip(beat_frames, 0, n_frames - 1)]]
    if cfg.trim_weak_edges and len(beat_frames):
        beat_frames = _trim_weak_edges(beat_frames, local[beat_frames])
    if len(beat_frames):
        beat_frames = _drop_short_runs(beat_frames, fr)

    if len(beat_frames) < 4:
        warnings.append("Fewer than 4 beats found; the track may be silent, beatless or too short.")

    # sub-frame refinement
    if len(beat_frames):
        ref_frames = np.array([_parabolic_peak(benv, int(f)) for f in beat_frames])
        beat_times = np.asarray(frame_to_time(ref_frames, cfg))
        order = np.argsort(beat_times)
        beat_times, beat_frames = beat_times[order], beat_frames[order]
    else:
        beat_times = np.array([])
    n_beats = len(beat_times)

    # ---- local BPM per beat ----------------------------------------------------------
    tempo_at_beat = tempo["bpm"][np.clip(beat_frames, 0, n_frames - 1)] if n_beats else np.array([])
    after_pause = np.zeros(n_beats, dtype=bool)
    if n_beats >= 2:
        dt = np.diff(beat_times)
        inst = 60.0 / dt
        ref_bpm = tempo_at_beat[:-1]
        invalid = (inst < 0.6 * ref_bpm) | (inst > 1.7 * ref_bpm)  # long gap (pause) or glitch
        inst = np.where(invalid, ref_bpm, inst)
        after_pause[1:] = invalid
        inst_beat = np.append(inst, inst[-1])
        local_bpm = ndi.median_filter(inst_beat, size=5, mode="nearest")
    else:
        local_bpm = tempo_at_beat.copy()

    # ---- per-beat confidence ---------------------------------------------------------
    support = np.zeros(n_beats)
    for i, f in enumerate(beat_frames):
        a, b = max(0, int(f - 4 * fr)), min(n_frames, int(f + 4 * fr))
        refv = np.percentile(benv[a:b], 95) if b > a else 0.0
        pk = benv[max(int(f) - 2, 0): int(f) + 3].max() if refv > 1e-9 else 0.0
        support[i] = min(1.0, pk / refv) if refv > 1e-9 else 0.0
    periodic = tempo["periodicity"][np.clip(beat_frames, 0, n_frames - 1)] if n_beats else np.array([])
    if n_beats:
        per_score = np.clip((periodic - cfg.periodicity_floor) /
                            (cfg.periodicity_full - cfg.periodicity_floor), 0, 1)
        confidence = np.sqrt(np.clip(support, 0, 1) * per_score)
    else:
        confidence = np.array([])
    low_conf = confidence < cfg.low_confidence_threshold if n_beats else np.array([], dtype=bool)

    if cfg.drop_low_confidence and n_beats:
        keep = ~low_conf
        beat_frames, beat_times, local_bpm, after_pause = (
            beat_frames[keep], beat_times[keep], local_bpm[keep], after_pause[keep],
        )
        confidence, low_conf, support, periodic = confidence[keep], low_conf[keep], support[keep], periodic[keep]
        n_beats = len(beat_times)

    # ---- tempo map -------------------------------------------------------------------
    tempo_segments: list[dict[str, Any]] = []
    seg_id = np.zeros(n_beats, dtype=int)
    rhythm_reliable = bool(n_beats >= 4 and np.mean(~low_conf) >= 0.5)
    if n_beats >= 4 and not rhythm_reliable:
        warnings.append(
            "Most beats are low-confidence (no stable rhythm detected): the tempo map is left empty "
            "and no global BPM is reported. Beats are still listed with their confidence.")
    if rhythm_reliable:
        x = np.log2(local_bpm)
        beta = (np.log2(1.0 + cfg.tempo_change_min_ratio) ** 2) * cfg.tempo_change_min_beats / 2.0
        min_len = max(3, cfg.tempo_change_min_beats // 2)
        segs = segment_piecewise_constant(x, beta, min_len)
        prev_bpm = None
        for k, (a, b) in enumerate(segs):
            seg_id[a:b] = k
            bpm = float(np.median(local_bpm[a:b]))
            start = 0.0 if k == 0 else float(beat_times[a])
            end = duration if k == len(segs) - 1 else float(beat_times[b])
            ratio = None if prev_bpm is None else abs(bpm - prev_bpm) / prev_bpm
            tempo_segments.append(dict(
                start_time=start, end_time=end, estimated_bpm=bpm,
                change_type=None if prev_bpm is None else (
                    "sudden" if ratio >= cfg.tempo_sudden_ratio else "gradual"),
                beat_count=int(b - a),
                confidence=float(np.mean(confidence[a:b])),
                octave_ambiguity_risk=bool(bpm < 70 or bpm > 160),
                alternate_bpm=float(bpm * 2 if bpm < 70 else bpm / 2) if (bpm < 70 or bpm > 160) else None,
            ))
            prev_bpm = bpm
    elif 1 <= n_beats < 4:
        tempo_segments.append(dict(
            start_time=0.0, end_time=duration, estimated_bpm=float(np.median(local_bpm)),
            change_type=None, beat_count=n_beats, confidence=float(np.mean(confidence)),
            octave_ambiguity_risk=False, alternate_bpm=None))

    # global BPM only if "meaningful" (tempo essentially constant)
    median_bpm = float(np.median(local_bpm)) if n_beats else None
    frac_near = float(np.mean(np.abs(local_bpm - median_bpm) / median_bpm < 0.04)) if n_beats else 0.0
    constant = bool(rhythm_reliable and len(tempo_segments) == 1 and frac_near > 0.85)
    global_bpm = median_bpm if constant else None

    # ---- meter & downbeats -----------------------------------------------------------
    meter: dict[str, Any] = dict(status="not_attempted", beats_per_measure=None, confidence=0.0,
                                 candidate_strengths={}, sections=[])
    measure = [None] * n_beats
    beat_in_measure = [None] * n_beats
    is_down: list[bool | None] = [None] * n_beats
    meter_conf = np.zeros(n_beats)
    confident_frac = float(np.mean(~low_conf)) if n_beats else 0.0
    if cfg.detect_meter and n_beats >= 12 and confident_frac < cfg.meter_min_confident_beat_fraction:
        meter["status"] = "unknown"
        warnings.append("Most beats are low-confidence, so time signature / downbeats were not "
                        "estimated (fields are null).")
    elif cfg.detect_meter and n_beats >= 12:
        accent = beat_accents(y, beat_frames, beat_times, feats, cfg)
        if cfg.beats_per_bar:
            m, mconf, strengths = cfg.beats_per_bar, 1.0, {}
            _, _, strengths = estimate_meter(accent, cfg)
            status = "user_specified"
        else:
            m, mconf, strengths = estimate_meter(accent, cfg)
            ok = m is not None and mconf > 0.0 and strengths.get(m, 0.0) >= cfg.meter_min_strength
            status = "estimated" if ok else "unknown"
            if not ok:
                m = None
        meter_secs: list[tuple[int, int, int]] = []
        if m is not None:
            n_acc = len(accent)
            meter_secs = [(0, n_acc, m)]
            if cfg.detect_meter_changes and not cfg.beats_per_bar:
                meter_secs = meter_sections(accent, m, cfg)
            states = np.zeros(n_acc, dtype=int)
            m_arr = np.zeros(n_acc, dtype=int)
            for a_, b_, mm in meter_secs:
                states[a_:b_] = decode_bar_phase(accent[a_:b_], mm, cfg.downbeat_slip_penalty)
                m_arr[a_:b_] = mm
            down = states == 0
            meter_conf = local_downbeat_confidence(accent, down, m_arr)
            meas = 0
            for i, s in enumerate(states):
                if s == 0:
                    meas += 1
                measure[i] = meas            # 0 = pickup beats before the first detected downbeat
                beat_in_measure[i] = int(s) + 1
                is_down[i] = bool(s == 0)
            if len(meter_secs) > 1:
                # dominant meter (by number of beats) is the headline value
                tot: dict[int, int] = {}
                for a_, b_, mm in meter_secs:
                    tot[mm] = tot.get(mm, 0) + (b_ - a_)
                m = max(tot, key=tot.get)
                warnings.append(
                    f"Meter appears to change ({' -> '.join(str(s[2]) for s in meter_secs)} beats per "
                    "bar). Section boundaries are accurate to about +-2 bars; bars near them may be "
                    "mislabelled.")
            slips = 0
            for a_, b_, mm in meter_secs:
                st_ = states[a_:b_]
                slips += int(np.sum(st_[1:] != (st_[:-1] + 1) % mm))
            if slips:
                warnings.append(
                    f"Bar phase was re-aligned {slips} time(s): irregular bars, pauses or a meter "
                    "change were detected. Bars of unusual length are expected there.")
        meter = dict(status=status, beats_per_measure=m, confidence=float(mconf),
                     candidate_strengths={int(k): float(v) for k, v in strengths.items()},
                     sections=[dict(start_beat=int(a_), end_beat=int(b_ - 1), beats_per_measure=int(mm),
                                    start_time=_round(beat_times[a_], nd), end_time=_round(beat_times[b_ - 1], nd))
                               for a_, b_, mm in meter_secs] if m is not None else [])
        if status == "unknown":
            warnings.append("Time signature could not be determined reliably; measure/downbeat "
                            "fields are null.")
    elif cfg.detect_meter:
        meter["status"] = "unknown"
        warnings.append("Too few beats to estimate meter; measure/downbeat fields are null.")

    # ---- onsets ----------------------------------------------------------------------
    pre_max = max(1, int(0.03 * fr))
    post_max = max(1, int(0.01 * fr) + 1)
    pre_avg = max(1, int(0.10 * fr))
    post_avg = max(1, int(0.10 * fr) + 1)
    wait = max(1, int(0.03 * fr))
    on_frames = librosa.onset.onset_detect(
        onset_envelope=oenv, sr=cfg.sr, hop_length=hop, units="frames", backtrack=False,
        pre_max=pre_max, post_max=post_max, pre_avg=pre_avg, post_avg=post_avg,
        delta=cfg.onset_delta, wait=wait,
    )
    on_frames = np.asarray(on_frames, dtype=int)
    if on_frames.size:
        on_frames = on_frames[~silent[np.clip(on_frames, 0, n_frames - 1)]]
    on_ref = float(np.percentile(oenv[oenv > 0], 99)) if np.any(oenv > 0) else 1.0
    on_fr_ref = np.array([_parabolic_peak(oenv, int(f)) for f in on_frames]) if on_frames.size else np.array([])
    on_times = np.asarray(frame_to_time(on_fr_ref, cfg)) if on_frames.size else np.array([])
    on_strength = np.clip(oenv[on_frames] / on_ref, 0, 1) if on_frames.size else np.array([])

    # ---- subdivisions ----------------------------------------------------------------
    subdivs = [dict(division=None, label=None, confidence=0.0, times=[]) for _ in range(n_beats)]
    if cfg.detect_subdivisions and n_beats >= 3:
        onset_ref = np.zeros(n_beats)
        for i, f in enumerate(beat_frames):
            a, b = int(max(0, f - 4 * fr)), int(min(n_frames, f + 4 * fr))
            seg = oenv[a:b]
            onset_ref[i] = np.percentile(seg, 97) if seg.size else 0.0
        subdivs = detect_subdivisions(beat_times, oenv, onset_ref, fr, cfg)
        subdivs = [s if (not low_conf[i] and not silent[min(int(beat_frames[i]), n_frames - 1)])
                   else dict(division=None, label=None, confidence=0.0, times=[])
                   for i, s in enumerate(subdivs)]

    # ---- intensity -------------------------------------------------------------------
    frame_t = np.asarray(frame_to_time(np.arange(n_frames), cfg))
    n_bins = int(np.ceil(duration * cfg.intensity_hz))
    bin_idx = np.minimum((frame_t * cfg.intensity_hz).astype(int), n_bins - 1)
    cnt = np.bincount(bin_idx, minlength=n_bins).astype(float)
    rms_bins = np.bincount(bin_idx, weights=rms, minlength=n_bins) / np.maximum(cnt, 1)
    p99 = float(np.percentile(rms, 99)) or 1e-12
    rms_norm = np.clip(rms_bins / p99, 0, 1)
    db_bins = 20 * np.log10(np.maximum(rms_bins, 1e-10) / feats["peak_rms"])
    loud_norm = np.clip((db_bins + 60.0) / 60.0, 0, 1)
    int_times = np.arange(n_bins) / cfg.intensity_hz

    # per-beat intensity and onset counts
    beat_intensity = np.zeros(n_beats)
    onsets_in_beat = np.zeros(n_beats, dtype=int)
    for i in range(n_beats):
        t0 = beat_times[i]
        t1 = beat_times[i + 1] if i + 1 < n_beats else min(duration, t0 + 60.0 / max(local_bpm[i], 1))
        if t1 - t0 > 3.0:  # a pause follows; don't integrate across it
            t1 = t0 + 60.0 / max(local_bpm[i], 1)
        a, b = int(t0 * fr), max(int(t1 * fr), int(t0 * fr) + 1)
        beat_intensity[i] = float(np.clip(rms[a:b].mean() / p99, 0, 1)) if b > a else 0.0
        onsets_in_beat[i] = int(np.sum((on_times >= t0) & (on_times < t1))) if on_times.size else 0

    # fractional beat position for onsets (lets a generator snap/relate events to the beat grid)
    onsets_out = []
    for t, s in zip(on_times, on_strength):
        bp = None
        if n_beats >= 2 and beat_times[0] <= t <= beat_times[-1]:
            j = int(np.searchsorted(beat_times, t, side="right") - 1)
            j = min(j, n_beats - 2)
            span = beat_times[j + 1] - beat_times[j]
            if span < 3.0:  # skip fractional position across pauses
                bp = j + (t - beat_times[j]) / span
        onsets_out.append(dict(time=_round(t, nd), strength=_round(s, 3), beat_position=_round(bp, 3)))

    # ---- assemble --------------------------------------------------------------------
    beats_out = []
    for i in range(n_beats):
        sd_ = subdivs[i]
        beats_out.append(dict(
            index=i,
            time=_round(beat_times[i], nd),
            local_bpm=_round(local_bpm[i], 2),
            tempo_segment=int(seg_id[i]),
            measure=measure[i],
            beat_in_measure=beat_in_measure[i],
            is_downbeat=is_down[i],
            meter_confidence=_round(meter_conf[i], 3) if is_down[i] is not None else None,
            confidence=_round(confidence[i], 3),
            low_confidence=bool(low_conf[i]),
            after_pause=bool(after_pause[i]),
            intensity=_round(beat_intensity[i], 3),
            onsets_in_beat=int(onsets_in_beat[i]),
            subdivision=dict(
                division=sd_["division"], label=sd_["label"],
                confidence=_round(sd_["confidence"], 3),
                times=[_round(t, nd) for t in sd_["times"]],
            ),
        ))

    tempo_changes = [dict(
        start_time=_round(s["start_time"], nd), end_time=_round(s["end_time"], nd),
        estimated_bpm=_round(s["estimated_bpm"], 2), change_type=s["change_type"],
        beat_count=s["beat_count"], confidence=_round(s["confidence"], 3),
        octave_ambiguity_risk=s["octave_ambiguity_risk"],
        alternate_bpm=_round(s["alternate_bpm"], 2),
    ) for s in tempo_segments]

    # coarse tempo curve (for visualisation / debugging), ~2 Hz
    tstep = max(1, int(round(0.5 * fr)))
    cf = np.arange(0, n_frames, tstep)
    tempo_curve = dict(
        times=[_round(t, nd) for t in frame_to_time(cf, cfg)],
        bpm=[_round(b, 2) for b in tempo["bpm"][cf]],
        periodicity=[_round(p, 3) for p in tempo["periodicity"][cf]],
        note="Raw frame-level tempo tracker output; the authoritative values are 'beats' and 'tempo_changes'.",
    )

    silences = [dict(start_time=_round(a / fr, nd), end_time=_round(b / fr, nd)) for a, b in _runs(silent)]
    arhythmic_out = [dict(start_time=_round(a / fr, nd), end_time=_round(b / fr, nd))
                     for a, b in _runs(arhythmic & ~silent)]

    result = dict(
        schema_version=SCHEMA_VERSION,
        metadata=dict(
            source_file=Path(path).name,
            duration_seconds=_round(duration, nd),
            sample_rate=int(sr),
            original_sample_rate=int(orig_sr) if orig_sr else None,
            hop_length=hop,
            time_reference="All timestamps are seconds from the start of the original audio file.",
            estimated_global_bpm=_round(global_bpm, 2),
            median_bpm=_round(median_bpm, 2),
            tempo_is_constant=constant,
            total_beats=n_beats,
            total_onsets=len(onsets_out),
            time_signature=dict(
                status=meter["status"], beats_per_measure=meter["beats_per_measure"],
                confidence=_round(meter["confidence"], 3),
                candidate_strengths={str(k): _round(v, 3) for k, v in meter["candidate_strengths"].items()},
                varies=bool(len(meter.get("sections", [])) > 1),
                sections=meter.get("sections", []),
            ),
            mean_beat_confidence=_round(float(np.mean(confidence)) if n_beats else 0.0, 3),
            low_confidence_beat_fraction=_round(float(np.mean(low_conf)) if n_beats else 0.0, 3),
            warnings=warnings,
            parameters={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        ),
        tempo_map=dict(
            is_constant_tempo=constant if rhythm_reliable else None,
            tempo_changes=tempo_changes,
            tempo_curve=tempo_curve,
        ),
        beats=beats_out,
        onsets=onsets_out,
        intensity=dict(
            sample_rate_hz=cfg.intensity_hz,
            description="Each value summarises [time, time + 1/sample_rate_hz). "
                        "rms_normalized = RMS / 99th-percentile RMS (clipped 0..1); "
                        "loudness_normalized maps 60 dB below the peak -> 0 and the peak -> 1.",
            times=[_round(t, nd) for t in int_times],
            rms_normalized=[_round(v, 3) for v in rms_norm],
            loudness_normalized=[_round(v, 3) for v in loud_norm],
        ),
        silences=silences,
        no_rhythmic_activity=arhythmic_out,
        limitations=LIMITATIONS,
    )
    return result


LIMITATIONS = [
    "Beat times come from onset-energy peaks (about +-12-25 ms accurate on clean transients); "
    "calibrate audio output latency separately in the game.",
    "Tempo is tracked from rhythmic periodicity. Songs with weak or no percussion, heavy rubato, or "
    "long ambient sections give low-confidence beats (flagged via 'confidence'/'low_confidence').",
    "Half-time / double-time cannot be resolved from audio alone; a tempo prior (start_bpm, "
    "min/max_bpm) picks the octave and 'octave_ambiguity_risk' flags segments outside 70-160 BPM.",
    "Beat phase can lock onto off-beats in heavily syncopated music (the tracker follows the "
    "strongest periodic onsets, not 'musical beat' semantics).",
    "Tempo changes are segmented from the tracked beats: sudden changes are located to within a "
    "few beats; gradual accelerations are reported as a staircase of small steps.",
    "Meter is one global estimate with local bar-phase re-alignment. Mixed/changing time "
    "signatures are NOT labelled explicitly; they appear as phase slips (see warnings).",
    "Downbeats are heuristic (bass onset + onset strength + harmonic change) and are the least "
    "reliable output; check 'time_signature.confidence' and per-beat 'meter_confidence'.",
    "Subdivisions detect straight 8ths, triplets and 16ths only; swing/shuffle and polyrhythms "
    "are not recognised.",
    "Beats inside silence or stretches with no onset activity (pauses, beatless intros, sustained "
    "pads) are removed rather than guessed; beats after a pause are marked 'after_pause'. See "
    "'silences' and 'no_rhythmic_activity'.",
]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def write_json(result: dict[str, Any], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


def _summary(r: dict[str, Any]) -> str:
    m = r["metadata"]
    lines = [
        f"File: {m['source_file']}  duration {m['duration_seconds']} s  (analysis sr {m['sample_rate']} Hz)",
        f"Beats: {m['total_beats']}   Onsets: {m['total_onsets']}   "
        f"Mean beat confidence: {m['mean_beat_confidence']}  (low-confidence: {m['low_confidence_beat_fraction']})",
        f"Global BPM: {m['estimated_global_bpm']} (constant tempo: {m['tempo_is_constant']}; median {m['median_bpm']})",
        f"Time signature: {m['time_signature']['status']} "
        f"{m['time_signature']['beats_per_measure']} beats/bar (conf {m['time_signature']['confidence']})",
        "Tempo map:",
    ]
    for t in r["tempo_map"]["tempo_changes"]:
        lines.append(f"  {t['start_time']:8.2f}s -> {t['end_time']:8.2f}s  {t['estimated_bpm']:7.2f} BPM"
                     f"  [{t['change_type'] or 'start'}]  beats={t['beat_count']}")
    for w in m["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Variable-tempo beat/onset/intensity analysis (Librosa).")
    p.add_argument("audio", help="path to an audio file (wav, mp3, flac, ogg, ...)")
    p.add_argument("-o", "--output", help="output JSON path (default: <audio>.analysis.json)")
    p.add_argument("--min-bpm", type=float, default=None)
    p.add_argument("--max-bpm", type=float, default=None)
    p.add_argument("--start-bpm", type=float, default=None, help="tempo prior centre (octave disambiguation)")
    p.add_argument("--tightness", type=float, default=None, help="beat tracker stiffness (default 60)")
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--beats-per-bar", type=int, default=None, help="force a meter instead of estimating it")
    p.add_argument("--no-meter", action="store_true", help="skip measure/downbeat estimation")
    p.add_argument("--no-subdivisions", action="store_true")
    p.add_argument("--drop-low-confidence", action="store_true", help="remove low-confidence beats from the output")
    p.add_argument("--intensity-hz", type=float, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = AnalysisConfig()
    for attr, val in [("min_bpm", a.min_bpm), ("max_bpm", a.max_bpm), ("start_bpm", a.start_bpm),
                      ("tightness", a.tightness), ("hop_length", a.hop_length),
                      ("beats_per_bar", a.beats_per_bar), ("intensity_hz", a.intensity_hz)]:
        if val is not None:
            setattr(cfg, attr, val)
    if a.no_meter:
        cfg.detect_meter = False
    if a.no_subdivisions:
        cfg.detect_subdivisions = False
    cfg.drop_low_confidence = a.drop_low_confidence

    out = a.output or str(Path(a.audio).with_suffix(".analysis.json"))
    try:
        result = analyze_audio(a.audio, cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1
    write_json(result, out)
    print(_summary(result))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
