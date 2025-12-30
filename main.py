import matplotlib
# OPTIMIZATION: Agg backend for server-side rendering
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "EEG Server is Running!"}

@app.post("/analyze")
def analyze_eeg(cnt_file: UploadFile = File(...), exp_file: UploadFile = File(...)):
    
    # 1. SAVE UPLOADS TEMP
    try:
        tmp_cnt = tempfile.NamedTemporaryFile(delete=False, suffix=".cnt")
        tmp_cnt.close() 
        tmp_cnt_path = tmp_cnt.name

        tmp_exp = tempfile.NamedTemporaryFile(delete=False, suffix=".exp")
        tmp_exp.close()
        tmp_exp_path = tmp_exp.name

        with open(tmp_cnt_path, "wb") as buffer:
            shutil.copyfileobj(cnt_file.file, buffer)
            
        with open(tmp_exp_path, "wb") as buffer:
            shutil.copyfileobj(exp_file.file, buffer)

        # 2. LOAD DATA
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        
        # 3. PARSE EXP
        trial_type_map = {}
        reaction_times = []
        
        with open(tmp_exp_path, 'r') as f:
            lines = f.readlines()
            for line in lines[8:]: 
                parts = line.strip().split('\t')
                if len(parts) < 7: parts = line.strip().split()
                if len(parts) >= 7:
                    t_id = parts[0].strip()
                    t_name = parts[1].strip()
                    t_type = parts[3].strip()
                    try: t_lat = int(parts[6].strip())
                    except: t_lat = 1000
                    
                    trial_type_map[t_id] = t_type
                    if t_type == 'R' and t_lat < 1000:
                        reaction_times.append((t_lat, t_id, t_name))

        # Frontend Text Logic
        easiest_txt = "N/A"
        toughest_txt = "N/A"
        if reaction_times:
            best = min(reaction_times, key=lambda x: x[0])
            worst = max(reaction_times, key=lambda x: x[0])
            easiest_txt = f"Trial {best[1]}: '{best[2]}' ({best[0]}ms)"
            toughest_txt = f"Trial {worst[1]}: '{worst[2]}' ({worst[0]}ms)"

        # 4. EVENTS
        new_events_list = []
        for annot in raw.annotations:
            clean_id = str(annot['description']).strip()
            sType = trial_type_map.get(clean_id, "Unknown")
            if sType == "Unknown": continue
            code = 1 if sType == 'R' else 2
            new_events_list.append([raw.time_as_index(annot['onset'])[0], 0, code])

        if not new_events_list:
            return {"error": "No matching events found in .exp file"}

        custom_events = np.array(new_events_list)
        event_ids = {'Target': 1, 'Non-Target': 2}

        # 5. FILTER & EPOCH (No Artifact Rejection)
        raw.filter(0.1, 30.0, picks='eeg', n_jobs=1, verbose=False)  # Changed n_jobs=-1 to n_jobs=1 to avoid joblib warning
        
        # Create epochs WITHOUT artifact rejection to keep all trials
        epochs = mne.Epochs(
            raw, 
            custom_events, 
            event_ids, 
            tmin=-0.2, 
            tmax=0.6, 
            baseline=(None, 0), 
            picks='eeg', 
            preload=True, 
            verbose=False
        )
        
        if len(epochs) == 0:
            return {"error": "All trials were rejected due to artifacts (too much noise)."}

        # Get evoked responses (these are in VOLTS by default)
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()
        
        # CRITICAL: Calculate DIFFERENCE WAVE (Target - Non-Target)
        # This is the proper way to isolate the P300 component
        evoked_difference = mne.combine_evoked([evoked_target, evoked_nontarget], weights=[1, -1])

        # 6. PLOT (REFINED GRID LAYOUT)
        
        fig = plt.figure(figsize=(12, 32)) 
        
        gs = gridspec.GridSpec(7, 1, height_ratios=[1.2, 1.0, 2.5, 1.0, 2.5, 1.0, 2.5], hspace=0.5)

        # --- ROW 0: MAIN HEADER ---
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

        # Section Definitions
        sections = [
            {
                "comp": "P100", "ch": "OZ", "color": "green", "window": (0.08, 0.14),
                "title": "A. P100 (The 'First Glance' Test)",
                "desc": "Measures how physically overwhelming the screen is—telling us if the sheer amount of clutter is tiring the user's eyes before they even start reading.",
                "show_difference": False  # P100 - show both conditions
            },
            {
                "comp": "N200", "ch": "FZ", "color": "yellow", "window": (0.20, 0.30),
                "title": "B. N200 (The 'Confusion' Test)",
                "desc": "Measures mental friction—revealing the exact moment a user gets stuck or frustrated because they can't instantly find the insight they need in a wall of numbers.",
                "show_difference": False  # N200 - show both conditions
            },
            {
                "comp": "P300", "ch": "PZ", "color": "red", "window": (0.30, 0.50),
                "title": "C. P300 (The 'Confidence' Test)",
                "desc": "Measures the 'Aha!' moment—proving the user has successfully understood the data and is ready to make a confident decision, rather than hesitating. The difference wave (Target - Non-Target) isolates the cognitive decision-making response.",
                "show_difference": True  # P300 - show difference wave!
            }
        ]
        
        row_indices = [(1, 2), (3, 4), (5, 6)]

        for i, sec in enumerate(sections):
            text_row, graph_row = row_indices[i]
            channel = sec["ch"]
            
            # --- TEXT ROW (CENTERED) ---
            ax_text = fig.add_subplot(gs[text_row])
            ax_text.axis('off') 
            
            ax_text.text(0.5, 0.75, sec["title"], ha='center', fontsize=20, weight='bold', color='#2c3e50')
            ax_text.text(0.5, 0.25, textwrap.fill(sec["desc"], width=100), ha='center', va='top', fontsize=14, color='#7f8c8d')

            # --- GRAPH ROW ---
            if channel in raw.ch_names:
                ax_graph = fig.add_subplot(gs[graph_row])
                
                # CRITICAL: Scale evoked data to microvolts BEFORE plotting
                # Create copies to avoid modifying original data
                evoked_target_uv = evoked_target.copy()
                evoked_target_uv.data *= 1e6  # Convert V to µV
                
                evoked_nontarget_uv = evoked_nontarget.copy()
                evoked_nontarget_uv.data *= 1e6  # Convert V to µV
                
                evoked_difference_uv = evoked_difference.copy()
                evoked_difference_uv.data *= 1e6  # Convert V to µV
                
                # For P300: Show DIFFERENCE WAVE (the proper P300 measurement)
                # For P100/N200: Show both conditions
                if sec["show_difference"]:
                    # Plot difference wave + both conditions for context
                    mne.viz.plot_compare_evokeds(
                        {
                            'Difference (T-NT)': evoked_difference_uv,
                            'Target': evoked_target_uv, 
                            'Non-Target': evoked_nontarget_uv
                        }, 
                        picks=channel, 
                        axes=ax_graph, 
                        show=False, 
                        show_sensors=False, 
                        legend='upper right',
                        title=None,
                        colors={'Difference (T-NT)': 'purple', 'Target': 'blue', 'Non-Target': 'orange'},
                        linestyles={'Difference (T-NT)': '-', 'Target': '--', 'Non-Target': '--'}
                    )
                    
                    # Calculate P300 amplitude (peak in difference wave)
                    ch_idx = evoked_difference_uv.ch_names.index(channel)
                    time_mask = (evoked_difference_uv.times >= sec["window"][0]) & (evoked_difference_uv.times <= sec["window"][1])
                    p300_amplitude = evoked_difference_uv.data[ch_idx, time_mask].max()
                    p300_latency = evoked_difference_uv.times[time_mask][evoked_difference_uv.data[ch_idx, time_mask].argmax()]
                    
                    # Add P300 measurement annotation
                    ax_graph.text(0.98, 0.02, f'P300: {p300_amplitude:.2f}µV @ {p300_latency*1000:.0f}ms',
                                 transform=ax_graph.transAxes,
                                 fontsize=11, weight='bold',
                                 ha='right', va='bottom',
                                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                else:
                    # Plot both conditions normally
                    mne.viz.plot_compare_evokeds(
                        {'Target': evoked_target_uv, 'Non-Target': evoked_nontarget_uv}, 
                        picks=channel, 
                        axes=ax_graph, 
                        show=False, 
                        show_sensors=False, 
                        legend='upper right',
                        title=None
                    )
                
                # Remove scientific notation
                ax_graph.ticklabel_format(style='plain', axis='y')
                
                # Highlight component time window
                ax_graph.axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.15, label=f'{sec["comp"]} Window')
                
                # Set x-limits only - let y-axis auto-scale
                ax_graph.set_xlim(-0.2, 0.6)
                
                # Add zero line for reference
                ax_graph.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
                
                # Styling
                ax_graph.spines['top'].set_visible(False)
                ax_graph.spines['right'].set_visible(False)
                ax_graph.grid(True, linestyle=':', alpha=0.4)
                ax_graph.set_ylabel("Amplitude (µV)", fontsize=12, weight='bold')
                ax_graph.set_xlabel("Time (s)", fontsize=12, weight='bold')
                ax_graph.tick_params(axis='both', which='major', labelsize=10)
                
                # Add component label on graph
                ax_graph.text(0.02, 0.98, f'{sec["comp"]} @ {channel}', 
                              transform=ax_graph.transAxes, 
                              fontsize=11, weight='bold', 
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                ax_graph = fig.add_subplot(gs[graph_row])
                ax_graph.text(0.5, 0.5, f'Channel {channel} not found in data', 
                              ha='center', va='center', fontsize=14, color='red')
                ax_graph.axis('off')

        # Add metadata footer with trial balance info
        target_count = len(epochs["Target"])
        nontarget_count = len(epochs["Non-Target"])
        
        # Calculate P300 amplitude for metadata
        ch_idx = evoked_difference.ch_names.index('PZ')
        time_mask = (evoked_difference.times >= 0.30) & (evoked_difference.times <= 0.50)
        p300_amplitude = (evoked_difference.data[ch_idx, time_mask].max()) * 1e6  # Convert to µV
        
        balance_note = ""
        
        # Alert if severely imbalanced (less than 10 trials in either condition)
        if target_count < 10 or nontarget_count < 10:
            balance_note = " ⚠️ Low trial count detected"
        
        fig.text(0.5, 0.01, 
                f'Total Epochs: {len(epochs)} | Target: {target_count} | Non-Target: {nontarget_count} | P300 Amplitude: {p300_amplitude:.2f}µV{balance_note}', 
                ha='center', fontsize=10, style='italic', color='#7f8c8d')

        # 7. CONVERT TO IMAGE
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)  # Reduced from 300 to 150 for faster transfer 
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "status": "success", 
            "image": img_str,
            "easiest": easiest_txt,
            "toughest": toughest_txt,
            "metadata": {
                "total_epochs": len(epochs),
                "target_epochs": target_count,
                "nontarget_epochs": nontarget_count,
                "channels_analyzed": [sec["ch"] for sec in sections if sec["ch"] in raw.ch_names],
                "balance_warning": target_count < 10 or nontarget_count < 10,
                "artifact_rejection": "disabled",
                "p300_amplitude_uv": float(p300_amplitude),
                "p300_interpretation": "Strong" if p300_amplitude > 5 else "Moderate" if p300_amplitude > 2 else "Weak"
            }
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")  # Print to server console
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "error_details": error_details
        }
    finally:
        if 'tmp_cnt_path' in locals() and os.path.exists(tmp_cnt_path): 
            os.remove(tmp_cnt_path)
        if 'tmp_exp_path' in locals() and os.path.exists(tmp_exp_path): 
            os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)