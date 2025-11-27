import matplotlib
matplotlib.use('Agg')

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import numpy as np
from scipy.signal import find_peaks
import tempfile
import os
import base64
import textwrap
from io import BytesIO
import shutil
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'filter': {
        'low': 0.5,
        'high': 30.0,
        'n_jobs': 1  # Safer for cloud instances
    },
    'epoch': {
        'tmin': -0.2,
        'tmax': 0.6,
        'baseline': (None, 0)
    },
    'rejection': {
        'eeg': 100e-6  # 100 microvolts
    },
    'p300': {
        'search_window': (0.25, 0.6),
        'score_range': (250, 600),  # milliseconds
        'window_duration': 0.2,
        'peak_prominence': 0.5e-6
    },
    'figure': {
        'size': (12, 32),
        'dpi': 150,
        'hspace': 0.75
    },
    'text': {
        'title_wrap': 95,
        'desc_wrap': 100
    },
    'thresholds': {
        'min_trial_count': 10,
        'low_trial_warning': 15
    }
}

# Analysis section definitions
ANALYSIS_SECTIONS = [
    {
        "comp": "P100",
        "ch": "OZ",
        "color": "green",
        "window": (0.08, 0.14),
        "bg_color": "#f0f8ff",
        "title": "A. P100 (The 'First Glance' Test)", 
        "desc": "This test measures the brain's immediate, subconscious reaction to seeing the screen. It tells us if the visual elements are striking enough to instantaneously grab the brain's attention—much like the primary visual cortex's swift response to early attention tasks—to ensure the design registers immediately."
    },
    {
        "comp": "N200",
        "ch": "FZ",
        "color": "yellow",
        "window": (0.20, 0.30),
        "bg_color": "#fffef0",
        "title": "B. N200 (The 'Mental Roadblock' Test)",
        "desc": "This measures mental friction by revealing the precise moment a user hits a cognitive barrier or is momentarily stuck. It shows whether the user is struggling or experiencing frustration because they cannot instantly recognize or classify the insight they need in complex data. This aligns with objective indicators of a high cognitive load or poor experience."
    },
    {
        "comp": "P300",
        "ch": "PZ",
        "color": "red",
        "window": (0.30, 0.50),
        "bg_color": "#fff5f5",
        "title": "C. P300 (The 'Confirmation' Test)",
        "desc": "This component marks the ultimate 'Aha!' moment of successful understanding. It confirms that the user has fully processed the key information and is cognitively ready to make a confident decision or take action, rather than hesitating due to uncertainty."
    }
]

# ============================================================================
# FASTAPI SETUP
# ============================================================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_uploaded_files(cnt_file: UploadFile, exp_file: UploadFile):
    """Validate that uploaded files have correct extensions."""
    if not cnt_file.filename.endswith('.cnt'):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected .cnt file")
    if not exp_file.filename.endswith('.exp'):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected .exp file")


def save_upload_to_temp(upload_file: UploadFile, suffix: str) -> str:
    """Save an uploaded file to a temporary location and return the path."""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.close()
    
    with open(tmp_file.name, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return tmp_file.name


def parse_experiment_file(exp_path: str):
    """
    Parse the .exp file to extract trial mappings and reaction times.
    
    Returns:
        tuple: (trial_type_map, reaction_times)
            - trial_type_map: dict mapping BOTH trial IDs and Trigger Codes to types
            - reaction_times: list of (latency_ms, trial_id, trial_name) tuples
    """
    trial_type_map = {}
    reaction_times = []
    
    with open(exp_path, 'r') as f:
        lines = f.readlines()
        
        for line in lines[8:]:  # Skip header lines
            parts = line.strip().split('\t')
            if len(parts) < 7:
                parts = line.strip().split()
            
            if len(parts) >= 7:
                # Column indices based on your file format
                trial_id = parts[0].strip()      # '1'
                trial_name = parts[1].strip()    # 'Underwater Photography'
                trial_type = parts[3].strip()    # 'R' or 'C'
                
                # Column 5 contains the TRIGGER CODE (e.g. '12001')
                try:
                    trigger_code = parts[5].strip()
                except:
                    trigger_code = None

                # Column 6 contains LATENCY
                try:
                    latency = int(parts[6].strip())
                except:
                    latency = 1000
                
                # --- KEY FIX: Map BOTH the ID and the Trigger Code ---
                # This ensures we catch the event whether EEG calls it '1' or '12001'
                trial_type_map[trial_id] = trial_type
                if trigger_code:
                    trial_type_map[trigger_code] = trial_type
                
                if trial_type == 'R' and latency < 1000:
                    reaction_times.append((latency, trial_id, trial_name))
    
    return trial_type_map, reaction_times


def calculate_task_extremes(reaction_times):
    """Identify easiest and toughest tasks based on reaction times."""
    if not reaction_times:
        return "N/A", "N/A"
    
    best = min(reaction_times, key=lambda x: x[0])
    worst = max(reaction_times, key=lambda x: x[0])
    
    easiest_txt = f"Trial {best[1]}: '{best[2]}' ({best[0]}ms)"
    toughest_txt = f"Trial {worst[1]}: '{worst[2]}' ({worst[0]}ms)"
    
    return easiest_txt, toughest_txt


def map_events_to_codes(raw, trial_type_map):
    """
    Map raw annotations to event codes based on trial type.
    Includes robust string cleaning.
    """
    new_events_list = []
    found_descriptions = set()
    
    for annot in raw.annotations:
        # Clean up description (e.g. remove "Stimulus")
        raw_desc = str(annot['description'])
        clean_id = raw_desc.replace('Stimulus', '').strip()
        found_descriptions.add(clean_id)
        
        trial_type = trial_type_map.get(clean_id, "Unknown")
        
        if trial_type == "Unknown":
            continue
        
        code = 1 if trial_type == 'R' else 2
        event_sample = raw.time_as_index(annot['onset'])[0]
        new_events_list.append([event_sample, 0, code])
    
    if not new_events_list:
        # Generate helpful error message if no matches found
        sample_eeg = list(found_descriptions)[:5]
        sample_map = list(trial_type_map.keys())[:5]
        raise ValueError(
            f"No matching events found. "
            f"EEG file has events: {sample_eeg}. "
            f"Experiment file mapping expects: {sample_map}..."
        )
    
    custom_events = np.array(new_events_list)
    event_ids = {'Target': 1, 'Non-Target': 2}
    
    return custom_events, event_ids


def calculate_rejection_stats(total_events: int, epochs):
    """Calculate epoch rejection statistics."""
    good_epochs = len(epochs)
    dropped_epochs = total_events - good_epochs
    drop_percentage = (dropped_epochs / total_events) * 100 if total_events > 0 else 0
    
    return {
        'total_events': total_events,
        'good_epochs': good_epochs,
        'dropped_epochs': dropped_epochs,
        'drop_percentage': drop_percentage
    }


def detect_p300_peak(evoked_target, channel: str):
    """Detect P300 peak using robust peak-finding algorithm."""
    if channel not in evoked_target.ch_names:
        return None
    
    ch_idx = evoked_target.ch_names.index(channel)
    data = evoked_target.data[ch_idx, :]
    times = evoked_target.times
    
    # Extract data within P300 search window
    window_start, window_end = CONFIG['p300']['search_window']
    mask = (times >= window_start) & (times <= window_end)
    
    if not np.any(mask):
        return None
    
    window_data = data[mask]
    window_times = times[mask]
    
    # Find peaks with minimum prominence to avoid noise
    peaks, properties = find_peaks(
        window_data, 
        prominence=CONFIG['p300']['peak_prominence']
    )
    
    if len(peaks) > 0:
        # Select the most prominent peak
        peak_idx = peaks[np.argmax(properties['prominences'])]
        return window_times[peak_idx]
    else:
        # Fallback to simple max if no prominent peaks found
        peak_idx = np.argmax(window_data)
        return window_times[peak_idx]


def calculate_p300_score(peak_latency_seconds: float):
    """Calculate Neural Confidence Score based on P300 latency."""
    latency_ms = peak_latency_seconds * 1000
    min_lat, max_lat = CONFIG['p300']['score_range']
    
    # Linear scoring: faster = better
    raw_score = 100 - ((latency_ms - min_lat) / (max_lat - min_lat) * 100)
    score = max(0, min(100, raw_score))
    
    return score, latency_ms


def create_header_section(ax, title: str, summary: str):
    """Render the report header with title and description."""
    ax.axis('off')
    
    ax.text(0.5, 0.85, title, 
            ha='center', fontsize=26, weight='bold', color='#2c3e50')
    
    wrapped_summary = textwrap.fill(summary, width=CONFIG['text']['title_wrap'])
    ax.text(0.5, 0.35, wrapped_summary, 
            ha='center', va='top', fontsize=13, style='italic', 
            color='#34495e', linespacing=1.5)


def create_section_text(ax, section: dict):
    """Render section title and description."""
    ax.axis('off')
    ax.set_facecolor(section['bg_color'])
    
    ax.text(0.5, 0.75, section['title'], 
            ha='center', fontsize=20, weight='bold', color='#2c3e50')
    
    wrapped_desc = textwrap.fill(section['desc'], width=CONFIG['text']['desc_wrap'])
    ax.text(0.5, 0.25, wrapped_desc, 
            ha='center', va='top', fontsize=14, color='#7f8c8d')


def plot_erp_comparison(ax, evoked_target, evoked_nontarget, section: dict, 
                        highlight_window: tuple, p300_info: dict = None):
    """Plot ERP comparison with highlighting and optional P300 scoring."""
    channel = section['ch']
    
    # Plot ERPs
    mne.viz.plot_compare_evokeds(
        {'Target': evoked_target, 'Non-Target': evoked_nontarget}, 
        picks=channel, 
        axes=ax, 
        show=False, 
        show_sensors=False, 
        legend=False,  # We'll add custom legend
        title=None
    )
    
    # Add custom legend with transparency
    ax.legend(loc='upper left', framealpha=0.8, fontsize=10)
    
    # Highlight analysis window
    ax.axvspan(highlight_window[0], highlight_window[1], 
               color=section['color'], alpha=0.15, 
               label=f'{section["comp"]} Window')
    
    # Styling
    ax.set_xlim(-0.2, 0.6)
    ax.set_xticks(np.arange(-0.2, 0.7, 0.1))
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4, which='both')
    ax.minorticks_on()
    
    # Convert y-axis to microvolts
    ax.ticklabel_format(style='plain', axis='y')
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f'{val*1e6:.1f}' for val in y_ticks])
    
    ax.set_ylabel("Amplitude (µV)", fontsize=12, weight='bold')
    ax.set_xlabel("Time (s)", fontsize=12, weight='bold')
    
    # Add component label
    ax.text(0.02, 0.98, f'{section["comp"]} @ {channel}', 
            transform=ax.transAxes, fontsize=11, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add P300 score box if provided
    if p300_info and section['comp'] == 'P300':
        score_text = (
            f"P300 Latency: {p300_info['latency_ms']:.0f} ms\n"
            f"Neural Confidence Score: {p300_info['score']:.0f}%"
        )
        ax.text(0.98, 0.05, score_text, 
                transform=ax.transAxes, ha='right', va='bottom', 
                fontsize=11, color='black',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', 
                          ec='black', alpha=0.9))
        
        # Add research footnote
        footnote = (
            "* Studies using objective EEG measures show that high cognitive load "
            "is closely related to an increase in task completion time. Furthermore, "
            "successful decision-making is strongly connected to levels of self-reported "
            "confidence, where low confidence ratings often reflect a hard or uncertain "
            "cognitive pursuit."
        )
        ax.text(0.5, -0.32, footnote,
                transform=ax.transAxes, ha='center', fontsize=10, 
                style='italic', color='#e74c3c', wrap=True,
                bbox=dict(boxstyle='round,pad=0.7', fc='#fff5f5', alpha=0.8))


def create_report_figure(evoked_target, evoked_nontarget, sections, 
                         rejection_stats, balance_note):
    """Generate complete visualization report with all sections."""
    fig = plt.figure(figsize=CONFIG['figure']['size'])
    
    gs = gridspec.GridSpec(
        7, 1, 
        height_ratios=[1.2, 1.0, 2.5, 1.0, 2.5, 1.0, 2.5], 
        hspace=CONFIG['figure']['hspace']
    )
    
    # Header
    ax_header = fig.add_subplot(gs[0])
    main_title = "Neuro-UX Analyzer"
    summary_text = (
        "We analyze your business dashboard versions (Current vs. New) by showing them to users "
        "while they complete common management tasks, like \"Spot the revenue drop.\" Using an "
        "objective brain activity monitor (EEG), we perform a neurological stress test that bypasses "
        "unreliable subjective opinions.\n\n"
        "The results show how the design performs across three key stages of comprehension: the P100 "
        "reveals if the initial visual design is instantly effective; the N200 pinpoints exactly where "
        "the user gets mentally stuck or confused by complex charts; and the P300 proves how quickly "
        "the new design allows them to spot the right answer and confidently act on it. This provides "
        "you with quantifiable, biological data to validate your design choices."
    )
    create_header_section(ax_header, main_title, summary_text)
    
    # Section rows
    row_indices = [(1, 2), (3, 4), (5, 6)]
    p300_score_txt = "N/A"
    
    for i, section in enumerate(sections):
        text_row, graph_row = row_indices[i]
        channel = section["ch"]
        
        # Render text section
        ax_text = fig.add_subplot(gs[text_row])
        create_section_text(ax_text, section)
        
        # Render graph section
        if channel in evoked_target.ch_names:
            ax_graph = fig.add_subplot(gs[graph_row])
            
            # Handle dynamic P300 window
            highlight_window = section["window"]
            p300_info = None
            
            if section["comp"] == "P300":
                p300_peak_time = detect_p300_peak(evoked_target, channel)
                
                if p300_peak_time is not None:
                    # Adjust window to start at peak
                    highlight_window = (
                        p300_peak_time, 
                        p300_peak_time + CONFIG['p300']['window_duration']
                    )
                    
                    # Calculate score
                    score, latency_ms = calculate_p300_score(p300_peak_time)
                    p300_score_txt = f"{score:.0f}%"
                    p300_info = {'score': score, 'latency_ms': latency_ms}
            
            plot_erp_comparison(ax_graph, evoked_target, evoked_nontarget, 
                              section, highlight_window, p300_info)
        else:
            ax_graph = fig.add_subplot(gs[graph_row])
            ax_graph.text(0.5, 0.5, f'Channel {channel} not found', 
                          ha='center', fontsize=14, color='red')
            ax_graph.axis('off')
    
    # Footer metadata
    stats = rejection_stats
    footer_line1 = (
        f'Clean Epochs: {stats["good_epochs"]} '
        f'(Target: {len(evoked_target.nave)} | Non-Target: {len(evoked_nontarget.nave)})'
        f'{balance_note}'
    )
    footer_line2 = (
        f'Rejected: {stats["dropped_epochs"]}/{stats["total_events"]} '
        f'({stats["drop_percentage"]:.1f}%) | '
        f'Threshold: {CONFIG["rejection"]["eeg"]*1e6:.0f}µV | '
        f'Filter: {CONFIG["filter"]["low"]}-{CONFIG["filter"]["high"]}Hz'
    )
    
    fig.text(0.5, 0.02, footer_line1, ha='center', fontsize=10, color='#7f8c8d')
    fig.text(0.5, 0.005, footer_line2, ha='center', fontsize=9, 
             style='italic', color='#95a5a6')
    
    # Export to base64
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', 
                dpi=CONFIG['figure']['dpi'])
    plt.close(fig)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    
    return img_str, p300_score_txt


def cleanup_resources(raw, epochs, evoked_target, evoked_nontarget):
    """Clean up memory resources after analysis."""
    try:
        del raw, epochs, evoked_target, evoked_nontarget
        plt.close('all')
        gc.collect()
    except:
        pass

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
def read_root():
    return {"status": "EEG Server is Running!"}


@app.post("/analyze")
def analyze_eeg(cnt_file: UploadFile = File(...), exp_file: UploadFile = File(...)):
    """Analyze EEG data and generate Neuro-UX report."""
    tmp_cnt_path = None
    tmp_exp_path = None
    raw = None
    epochs = None
    evoked_target = None
    evoked_nontarget = None
    
    try:
        # Validate inputs
        validate_uploaded_files(cnt_file, exp_file)
        
        # Save uploads to temporary files
        tmp_cnt_path = save_upload_to_temp(cnt_file, ".cnt")
        tmp_exp_path = save_upload_to_temp(exp_file, ".exp")
        
        # Load EEG data
        try:
            raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to load .cnt file: {str(e)}"
            )
        
        # Parse experiment file
        trial_type_map, reaction_times = parse_experiment_file(tmp_exp_path)
        easiest_txt, toughest_txt = calculate_task_extremes(reaction_times)
        
        # Map events to codes
        custom_events, event_ids = map_events_to_codes(raw, trial_type_map)
        
        # Apply bandpass filter
        raw.filter(
            CONFIG['filter']['low'], 
            CONFIG['filter']['high'], 
            picks='eeg', 
            n_jobs=CONFIG['filter']['n_jobs'], 
            verbose=False
        )
        
        # Create epochs with artifact rejection
        epochs = mne.Epochs(
            raw, 
            custom_events, 
            event_ids, 
            tmin=CONFIG['epoch']['tmin'], 
            tmax=CONFIG['epoch']['tmax'], 
            baseline=CONFIG['epoch']['baseline'], 
            picks='eeg', 
            reject=CONFIG['rejection'],
            preload=True, 
            verbose=False
        )
        
        # Calculate rejection statistics
        rejection_stats = calculate_rejection_stats(len(custom_events), epochs)
        
        # Check if any epochs survived
        if len(epochs) == 0:
            return {"error": "All trials were rejected due to artifacts (too much noise)."}
        
        # Check trial balance
        target_count = len(epochs['Target'])
        nontarget_count = len(epochs['Non-Target'])
        
        if (target_count < CONFIG['thresholds']['low_trial_warning'] or 
            nontarget_count < CONFIG['thresholds']['low_trial_warning']):
            print(f"WARNING: Low trial count may affect reliability")
        
        # Average epochs to get ERPs
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()
        
        # Generate report figure
        img_str, p300_score_txt = create_report_figure(
            evoked_target, 
            evoked_nontarget, 
            ANALYSIS_SECTIONS,
            rejection_stats,
            balance_note=""
        )
        
        # Clean up resources
        cleanup_resources(raw, epochs, evoked_target, evoked_nontarget)
        
        return {
            "status": "success", 
            "image": img_str,
            "easiest": easiest_txt,
            "toughest": toughest_txt,
            "neural_confidence_score": p300_score_txt,
            "metadata": {
                "total_events_found": rejection_stats['total_events'],
                "clean_epochs_kept": rejection_stats['good_epochs'],
                "rejected_epochs": rejection_stats['dropped_epochs'],
                "drop_percentage": round(rejection_stats['drop_percentage'], 2),
                "target_epochs": target_count,
                "nontarget_epochs": nontarget_count,
                "rejection_threshold_uv": CONFIG['rejection']['eeg'] * 1e6,
                "filter_range_hz": f"{CONFIG['filter']['low']}-{CONFIG['filter']['high']}"
            }
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        
        # Clean up on error
        if raw is not None or epochs is not None:
            cleanup_resources(raw, epochs, evoked_target, evoked_nontarget)
        
        return {"error": str(e), "details": error_details}
    
    finally:
        # Clean up temporary files
        if tmp_cnt_path and os.path.exists(tmp_cnt_path):
            try:
                os.remove(tmp_cnt_path)
            except:
                pass
        if tmp_exp_path and os.path.exists(tmp_exp_path):
            try:
                os.remove(tmp_exp_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)