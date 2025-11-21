# app_improved.py
import matplotlib
# Agg backend for server-side rendering
matplotlib.use('Agg')

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tempfile
import os
import base64
import textwrap
from io import BytesIO
import shutil
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def parse_exp_file(exp_path):
    """
    Parse the .exp file into a list of trial dicts.
    Tries to be robust to tabs / spaces. Heuristic for latency units handled later.
    Returns list of dict: {'id': id, 'name': name, 'type': type, 'latency_raw': int_or_none}
    """
    trials = []
    with open(exp_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n\r") for ln in f.readlines() if ln.strip() != ""]
    # Skip header lines if typical FAM2 has a short header; try to find a line containing 'trial' or numeric start
    start_idx = 0
    # Heuristic - some files have 8-line header (as you used before). If the 1st non-empty line contains alphabetic headers -> skip until numeric first column
    for i, ln in enumerate(lines[:20]):
        parts = ln.split()
        # If first token is numeric-like, assume data starts here
        if len(parts) > 0:
            try:
                _ = int(parts[0])
                start_idx = i
                break
            except Exception:
                continue

    for line in lines[start_idx:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 4:
            parts = line.strip().split()
        if len(parts) >= 2:
            try:
                t_id = parts[0].strip()
            except:
                t_id = ""
            t_name = parts[1].strip() if len(parts) > 1 else ""
            t_type = parts[3].strip() if len(parts) > 3 else ""
            # attempt to read a latency field commonly near 6th or 7th column
            t_lat = None
            for candidate_idx in (6, 5, 7, 4):
                if len(parts) > candidate_idx:
                    try:
                        t_lat = int(parts[candidate_idx].strip())
                        break
                    except Exception:
                        t_lat = None
            trials.append({
                "id": t_id,
                "name": t_name,
                "type": t_type,
                "latency_raw": t_lat
            })
    return trials


def latency_to_sample(lat_raw, raw):
    """
    Convert observed latency value (from .exp) into a sample index in raw.
    Heuristic rules:
    - If lat_raw is None -> return None
    - If lat_raw <= raw.n_times -> treat as sample index
    - Else if lat_raw <= raw.times[-1] * 1000 + 100 -> treat as ms and convert to samples
    - Else if lat_raw > 1e6 -> probably sample index (fallback)
    - Else fallback None
    """
    if lat_raw is None:
        return None
    sfreq = raw.info['sfreq']
    n_times = raw.n_times
    max_ms = raw.times[-1] * 1000.0
    # treat as sample index if it's in plausible range
    if 0 <= lat_raw <= n_times:
        return int(lat_raw)
    # treat as ms if plausible
    if 0 < lat_raw <= max_ms + 100:  # allow small buffer
        samp = int((lat_raw / 1000.0) * sfreq)
        if samp < n_times:
            return samp
    # fallback: if it's very large, maybe it's sample index anyway
    if lat_raw > n_times:
        # try dividing by 10 -> maybe in 100us units? (rare); otherwise fallback to None
        samp = int(lat_raw)
        if samp < n_times * 10 and samp > 0:
            samp_try = samp if samp < n_times else int(samp / 10)
            if samp_try < n_times:
                return samp_try
    return None


@app.get("/")
def read_root():
    return {"status": "EEG Server is Running (improved)!"}


@app.post("/analyze")
def analyze_eeg(cnt_file: UploadFile = File(...), exp_file: UploadFile = File(...)):
    """
    Improved analyze endpoint:
    - robust .exp parsing & event alignment
    - average reference
    - notch + bandpass filtering
    - resample to 250 Hz before epoching
    - basic amplitude-based artifact rejection
    - returns base64 image plus metrics (peak amplitudes/latencies)
    """
    tmp_cnt_path = None
    tmp_exp_path = None
    try:
        # 1) Save uploads to temp files
        tmp_cnt = tempfile.NamedTemporaryFile(delete=False, suffix=".cnt")
        tmp_cnt_path = tmp_cnt.name
        tmp_cnt.close()
        tmp_exp = tempfile.NamedTemporaryFile(delete=False, suffix=".exp")
        tmp_exp_path = tmp_exp.name
        tmp_exp.close()

        with open(tmp_cnt_path, "wb") as buffer:
            shutil.copyfileobj(cnt_file.file, buffer)
        with open(tmp_exp_path, "wb") as buffer:
            shutil.copyfileobj(exp_file.file, buffer)

        # 2) Read raw (do not preload to save memory yet)
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)  # preload for filtering/resampling
        sfreq_orig = raw.info['sfreq']

        # 3) Parse .exp
        trials = parse_exp_file(tmp_exp_path)
        trial_type_map = {}
        reaction_times = []
        for tr in trials:
            if tr['id'] != "":
                trial_type_map[tr['id']] = {'type': tr['type'], 'name': tr['name'], 'lat_raw': tr['latency_raw']}

        # 4) Re-reference to average (important for Pz, Oz, Fz)
        try:
            raw.set_eeg_reference('average', verbose=False)
        except Exception:
            # if this fails, continue but warn in metadata
            pass

        # 5) Notch filter (50/100Hz) THEN bandpass 0.1-30 Hz
        try:
            # Do notch first to remove mains lines
            raw.notch_filter(freqs=[50, 100], picks='eeg', verbose=False)
        except Exception:
            # On some files 100 may be above Nyquist after resample; we'll ignore failure
            pass

        # Optionally resample to reduce computation and improve SNR for ERPs
        resample_sfreq = 250.0
        if sfreq_orig > resample_sfreq:
            raw.resample(resample_sfreq, npad='auto', verbose=False)

        # Bandpass
        raw.filter(0.1, 30.0, picks='eeg', verbose=False)

        # 6) Build events from .exp latencies (preferred), fallback to annotations
        new_events_list = []
        # if trials have latencies, use them
        for tid, info in trial_type_map.items():
            lat_raw = info['lat_raw']
            sample_idx = latency_to_sample(lat_raw, raw)
            # If latency exists, create event at that sample
            if sample_idx is not None:
                evt_code = 1 if info['type'] == 'R' else 2
                new_events_list.append([int(sample_idx), 0, evt_code])
            else:
                # fallback: try to find an annotation with matching description
                # many CNT annotations contain the same trial id in description
                for ann in raw.annotations:
                    desc = str(ann['description']).strip()
                    if desc == tid or desc.endswith(tid) or tid.endswith(desc):
                        onset_sample = raw.time_as_index(ann['onset'])[0]
                        evt_code = 1 if info['type'] == 'R' else 2
                        new_events_list.append([int(onset_sample), 0, evt_code])
                        break

        # If still empty, try parsing annotations alone
        if not new_events_list and len(raw.annotations) > 0:
            for ann in raw.annotations:
                desc = str(ann['description']).strip()
                # Attempt to map description to trial type if possible
                mapped = trial_type_map.get(desc, None)
                if mapped is not None:
                    onset_sample = raw.time_as_index(ann['onset'])[0]
                    code = 1 if mapped['type'] == 'R' else 2
                    new_events_list.append([int(onset_sample), 0, code])
            # if still empty try generic mapping
            if not new_events_list:
                # map every annotation to 'Non-Target' as fallback
                for ann in raw.annotations:
                    onset_sample = raw.time_as_index(ann['onset'])[0]
                    new_events_list.append([int(onset_sample), 0, 2])

        if not new_events_list:
            raise HTTPException(status_code=400, detail="No events could be constructed from .exp/.cnt. Check your .exp file format.")

        events = np.array(new_events_list, dtype=int)
        event_ids = {'Target': 1, 'Non-Target': 2}

        # 7) Epoching with basic artifact rejection
        # conservative amplitude reject (µV units in Volts)
        # threshold configurable; 150e-6 is 150 µV
        reject_threshold_uv = 150.0
        reject = dict(eeg=(reject_threshold_uv * 1e-6))

        tmin, tmax = -0.2, 0.6
        epochs = mne.Epochs(raw, events, event_ids, tmin=tmin, tmax=tmax,
                            baseline=(None, 0), picks='eeg', preload=True,
                            reject=reject, verbose=False)

        if len(epochs) == 0:
            raise HTTPException(status_code=400, detail="No epochs left after artifact rejection.")

        # 8) Compute evokeds and difference
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()
        evoked_difference = mne.combine_evoked([evoked_target, evoked_nontarget], weights=[1, -1])

        # convert to microvolt copies for plotting & computing peaks
        ev_t_uv = evoked_target.copy()
        ev_nt_uv = evoked_nontarget.copy()
        ev_diff_uv = evoked_difference.copy()
        ev_t_uv.data *= 1e6
        ev_nt_uv.data *= 1e6
        ev_diff_uv.data *= 1e6

        # 9) Compute trial-wise CI (standard error) for shaded CI plots
        # For each condition compute mean +/- sem for the chosen channel when plotting
        # We'll compute metrics (peak amp & lat) per section too
        # Precompute picks
        ch_names = raw.ch_names

        # 10) Prepare plotting (keep your layout)
        fig = plt.figure(figsize=(12, 32))
        gs = gridspec.GridSpec(7, 1, height_ratios=[1.2, 1.0, 2.5, 1.0, 2.5, 1.0, 2.5], hspace=0.5)

        # Header
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off')
        main_title = "Neuro-UX: B2B Dashboard Analysis"
        summary_text = (
            "B2B Dashboard Analysis Summary\n\n"
            "In this experiment, we replace standard images with screenshots of your dashboard (Current vs. New) "
            "to measure how easily users can make decisions. By giving a user a specific management task (e.g., 'Find the Revenue Drop'), "
            "the EEG acts as an unbiased stress test: the P100 shows us if the visual design is too busy, "
            "the N200 highlights exactly where users get confused by complex charts, and the P300 proves "
            "how much faster the new design allows them to spot the answer and act on it. "
            "This validates your design with hard biological data, not just opinions."
        )
        ax_header.text(0.5, 0.85, main_title, ha='center', fontsize=26, weight='bold', color='#2c3e50')
        ax_header.text(0.5, 0.45, textwrap.fill(summary_text, width=90), ha='center', va='top', fontsize=14, style='italic', color='#34495e')

        sections = [
            {"comp": "P100", "ch": "OZ", "color": "green", "window": (0.08, 0.14),
             "title": "A. P100 (The 'First Glance' Test)",
             "desc": "Measures how physically overwhelming the screen is—telling us if the sheer amount of clutter is tiring the user's eyes before they even start reading."
             },
            {"comp": "N200", "ch": "FZ", "color": "yellow", "window": (0.18, 0.30),
             "title": "B. N200 (The 'Confusion' Test)",
             "desc": "Measures mental friction—revealing the exact moment a user gets stuck or frustrated because they can't instantly find the insight they need in a wall of numbers."
             },
            {"comp": "P300", "ch": "PZ", "color": "red", "window": (0.30, 0.50),
             "title": "C. P300 (The 'Confidence' Test)",
             "desc": "Measures the 'Aha!' moment—proving the user has successfully understood the data and is ready to make a confident decision, rather than hesitating."
             }
        ]
        row_indices = [(1, 2), (3, 4), (5, 6)]

        # Precompute evoked times
        times = ev_t_uv.times  # in seconds

        # Metrics to return
        component_metrics = {}

        # Precompute per-condition single-trial arrays for CI (if available)
        try:
            target_epochs = epochs['Target'].get_data() * 1e6  # n_epochs x n_chan x n_times (µV)
            nontarget_epochs = epochs['Non-Target'].get_data() * 1e6
        except Exception:
            target_epochs = None
            nontarget_epochs = None

        for i, sec in enumerate(sections):
            text_row, graph_row = row_indices[i]
            channel = sec["ch"]

            # TEXT
            ax_text = fig.add_subplot(gs[text_row])
            ax_text.axis('off')
            ax_text.text(0.5, 0.75, sec["title"], ha='center', fontsize=20, weight='bold', color='#2c3e50')
            ax_text.text(0.5, 0.25, textwrap.fill(sec["desc"], width=100), ha='center', va='top', fontsize=14, color='#7f8c8d')

            # GRAPH
            ax_graph = fig.add_subplot(gs[graph_row])
            if channel in ch_names:
                # pick channel index
                pick_idx = ch_names.index(channel)

                # Plot Target & Non-Target mean
                ax_graph.plot(times, ev_t_uv.data[pick_idx, :], label='Target (mean)', linewidth=1.5)
                ax_graph.plot(times, ev_nt_uv.data[pick_idx, :], label='Non-Target (mean)', linewidth=1.5)

                # plot difference wave as bold black
                ax_graph.plot(times, ev_diff_uv.data[pick_idx, :], label='Difference (T - NT)', color='black', linewidth=2.0)

                # CI shading if epoch data present
                if target_epochs is not None and target_epochs.shape[0] >= 2:
                    t_mean = target_epochs[:, pick_idx, :].mean(axis=0)
                    t_sem = target_epochs[:, pick_idx, :].std(axis=0, ddof=1) / np.sqrt(target_epochs.shape[0])
                    ax_graph.fill_between(times, t_mean - 1.96 * t_sem, t_mean + 1.96 * t_sem, alpha=0.15)

                if nontarget_epochs is not None and nontarget_epochs.shape[0] >= 2:
                    nt_mean = nontarget_epochs[:, pick_idx, :].mean(axis=0)
                    nt_sem = nontarget_epochs[:, pick_idx, :].std(axis=0, ddof=1) / np.sqrt(nontarget_epochs.shape[0])
                    ax_graph.fill_between(times, nt_mean - 1.96 * nt_sem, nt_mean + 1.96 * nt_sem, alpha=0.12)

                # Highlight component time window
                ax_graph.axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.12)

                # Compute peak amplitude & latency within window for Target and Difference
                wmin_idx = np.searchsorted(times, sec["window"][0])
                wmax_idx = np.searchsorted(times, sec["window"][1])

                # For polarity, choose peak method depending on component name:
                # P-components -> positive peak; N-components -> negative peak
                polarity = 'positive' if sec["comp"].upper().startswith('P') else 'negative'

                # Helper function
                def peak_info(array_1d, times_window, polarity):
                    if polarity == 'positive':
                        idx_rel = np.argmax(array_1d)
                    else:
                        idx_rel = np.argmin(array_1d)
                    amp = array_1d[idx_rel]
                    lat = times_window[idx_rel]
                    return float(amp), float(lat)

                # target peak
                try:
                    targ_win = ev_t_uv.data[pick_idx, wmin_idx:wmax_idx]
                    targ_times_win = times[wmin_idx:wmax_idx]
                    targ_amp, targ_lat = peak_info(targ_win, targ_times_win, polarity)
                except Exception:
                    targ_amp, targ_lat = None, None

                # diff peak
                try:
                    diff_win = ev_diff_uv.data[pick_idx, wmin_idx:wmax_idx]
                    diff_times_win = times[wmin_idx:wmax_idx]
                    diff_amp, diff_lat = peak_info(diff_win, diff_times_win, 'positive' if sec['comp'].upper().startswith('P') else 'negative')
                except Exception:
                    diff_amp, diff_lat = None, None

                component_metrics[sec['comp']] = {
                    'channel': channel,
                    'target_peak_uv': targ_amp,
                    'target_latency_s': targ_lat,
                    'difference_peak_uv': diff_amp,
                    'difference_latency_s': diff_lat
                }

                # Styling
                ax_graph.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
                ax_graph.set_xlim(tmin, tmax)
                ax_graph.set_ylabel("Amplitude (µV)", fontsize=12, weight='bold')
                ax_graph.set_xlabel("Time (s)", fontsize=12, weight='bold')
                ax_graph.grid(True, linestyle=':', alpha=0.4)
                ax_graph.ticklabel_format(style='plain', axis='y')
                ax_graph.tick_params(axis='both', which='major', labelsize=10)
                ax_graph.spines['top'].set_visible(False)
                ax_graph.spines['right'].set_visible(False)
                ax_graph.legend(loc='upper right', fontsize=9)
                ax_graph.text(0.02, 0.98, f'{sec["comp"]} @ {channel}', transform=ax_graph.transAxes, fontsize=11, weight='bold', verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_graph.text(0.5, 0.5, f'Channel {channel} not found in data', ha='center', va='center', fontsize=14, color='red')
                ax_graph.axis('off')

        # Footer metadata and counts
        target_count = len(epochs["Target"])
        nontarget_count = len(epochs["Non-Target"])
        balance_note = ""
        if target_count < 10 or nontarget_count < 10:
            balance_note = " ⚠️ Low trial count detected"
        fig.text(0.5, 0.01,
                 f'Total Epochs (sum conditions): {len(epochs)} | Target: {target_count} | Non-Target: {nontarget_count}{balance_note}',
                 ha='center', fontsize=10, style='italic', color='#7f8c8d')

        # Optionally add small topomaps for verification (P100/P300 distribution)
        # We'll add P300 topomap at peak latency from difference wave if available
        try:
            # find global peak latency of diff in 0.3-0.5s
            peak_win = (0.30, 0.50)
            wi0 = np.searchsorted(times, peak_win[0])
            wi1 = np.searchsorted(times, peak_win[1])
            data_diff = ev_diff_uv.data
            # find absolute peak across channels in that window
            mean_win = data_diff[:, wi0:wi1].mean(axis=1)
            chan_peak_idx = np.argmax(np.abs(mean_win))
            # latency within that window on that channel
            ch_ts = data_diff[chan_peak_idx, wi0:wi1]
            peak_rel = np.argmax(np.abs(ch_ts))
            peak_time = times[wi0 + peak_rel]
            # create an inset axis near the end of figure for topomap
            ax_topo = fig.add_axes([0.82, 0.06, 0.15, 0.15])
            mne.viz.plot_topomap(ev_diff_uv.data[:, np.searchsorted(times, peak_time)], ev_diff_uv.info, axes=ax_topo, show=False)
            ax_topo.set_title(f"Diff topomap\n{peak_time*1000:.0f} ms", fontsize=8)
        except Exception:
            pass

        # 11) Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")

        # 12) Build response
        response = {
            "status": "success",
            "image": img_str,
            "metadata": {
                "total_epochs": len(epochs),
                "target_epochs": target_count,
                "nontarget_epochs": nontarget_count,
                "channels_analyzed": [sec["ch"] for sec in sections if sec["ch"] in ch_names],
                "balance_warning": target_count < 10 or nontarget_count < 10,
                "artifact_rejection": "enabled",
                "reject_threshold_uv": reject_threshold_uv,
                "resample_sfreq": resample_sfreq,
                "original_sfreq": sfreq_orig,
                "component_metrics": component_metrics
            }
        }
        return response

    except HTTPException as he:
        # Return clean HTTPException info
        return {"error": str(he.detail)}
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": tb
        }
    finally:
        # Cleanup temp files
        try:
            if tmp_cnt_path and os.path.exists(tmp_cnt_path):
                os.remove(tmp_cnt_path)
        except Exception:
            pass
        try:
            if tmp_exp_path and os.path.exists(tmp_exp_path):
                os.remove(tmp_exp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
