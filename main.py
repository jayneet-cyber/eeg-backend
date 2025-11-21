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

# --- FastAPI Initialization ---
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
    
    tmp_cnt_path = None
    tmp_exp_path = None
    
    try:
        # 1. SAVE UPLOADS TEMP
        # Create and manage temporary files securely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cnt") as tmp_cnt:
            tmp_cnt_path = tmp_cnt.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".exp") as tmp_exp:
            tmp_exp_path = tmp_exp.name

        with open(tmp_cnt_path, "wb") as buffer:
            shutil.copyfileobj(cnt_file.file, buffer)
            
        with open(tmp_exp_path, "wb") as buffer:
            shutil.copyfileobj(exp_file.file, buffer)

        # 2. LOAD DATA
        # MNE's read_raw_cnt loads the Neuroscan .cnt file
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        
        # 3. PARSE EXP (Get trial type and Reaction Time info)
        trial_type_map = {}
        reaction_times = []
        
        with open(tmp_exp_path, 'r') as f:
            lines = f.readlines()
            # Assuming trial data starts after the 8th line
            for line in lines[8:]: 
                parts = line.strip().split('\t')
                if len(parts) < 7: parts = line.strip().split() # Handle space-delimited fallback
                
                if len(parts) >= 7:
                    t_id = parts[0].strip()      # Trial ID
                    t_name = parts[1].strip()    # Trial Name/Stimulus
                    t_type = parts[3].strip()    # Trial Type (e.g., R for Target)
                    try: 
                        t_lat = int(parts[6].strip()) # Reaction Latency (ms)
                    except ValueError: 
                        t_lat = 1000 # Use a high value for missing/invalid RT

                    trial_type_map[t_id] = t_type
                    # Collect valid RTs for "Target" trials (assuming R means required response)
                    if t_type == 'R' and t_lat < 1000:
                        reaction_times.append((t_lat, t_id, t_name))

        # Frontend Text Logic for RT Extremes
        easiest_txt = "N/A"
        toughest_txt = "N/A"
        if reaction_times:
            best = min(reaction_times, key=lambda x: x[0])
            worst = max(reaction_times, key=lambda x: x[0])
            easiest_txt = f"Trial {best[1]}: '{best[2]}' ({best[0]}ms)"
            toughest_txt = f"Trial {worst[1]}: '{worst[2]}' ({worst[0]}ms)"

        # 4. EVENTS (Map .cnt annotations to .exp trial types)
        new_events_list = []
        for annot in raw.annotations:
            clean_id = str(annot['description']).strip()
            sType = trial_type_map.get(clean_id, "Unknown")
            if sType == "Unknown": continue
            code = 1 if sType == 'R' else 2 # 1=Target (Response Req.), 2=Non-Target
            # [sample_index, 0, event_code]
            new_events_list.append([raw.time_as_index(annot['onset'])[0], 0, code])

        if not new_events_list:
            raise HTTPException(status_code=400, detail="No matching events found in .exp file")

        custom_events = np.array(new_events_list)
        event_ids = {'Target': 1, 'Non-Target': 2}

        # 5. FILTER & EPOCH (NO ARTIFACT REJECTION)
        # Apply a bandpass filter (0.1 to 30 Hz) typical for ERP analysis
        raw.filter(0.1, 30.0, picks='eeg', n_jobs=-1, verbose=False)
        
        # Create epochs: tmin=-200ms, tmax=600ms, baseline to pre-stimulus period
        # Artifact rejection explicitly NOT applied here (reject=None is default)
        epochs = mne.Epochs(
            raw, 
            custom_events, 
            event_ids, 
            tmin=-0.2, 
            tmax=0.6, 
            baseline=(None, 0), # Baseline is from tmin to 0s
            picks='eeg', 
            preload=True, 
            verbose=False
        )
        
        # Get evoked responses (averaged ERPs)
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()
        
        target_count = len(epochs["Target"])
        nontarget_count = len(epochs["Non-Target"])
        
        if len(epochs) == 0:
            raise HTTPException(status_code=400, detail="No epochs were successfully created. Check event timing or file format.")

        # 6. PLOT (REFINED GRID LAYOUT for 3-Component ERP Analysis)
        
        fig = plt.figure(figsize=(12, 32)) 
        
        # Define grid for 7 rows: Header, Text A, Graph A, Text B, Graph B, Text C, Graph C
        gs = gridspec.GridSpec(7, 1, height_ratios=[1.2, 1.0, 2.5, 1.0, 2.5, 1.0, 2.5], hspace=0.5)

        # --- ROW 0: MAIN HEADER ---
        ax_header = fig.add_subplot(gs[0])
        ax_header.axis('off') 
        
        main_title = "Neuro-UX: B2B Dashboard Analysis"
        summary_text = (
            "B2B Dashboard Analysis Summary\n\n"
            "This analysis uses key ERP components to test dashboard usability: "
            "The **P100** gauges visual clutter, the **N200** detects moments of user confusion, "
            "and the **P300** measures cognitive resource allocation and decision confidence. "
            "A faster P300 latency and higher P300 amplitude for 'Target' trials indicate superior design and lower cognitive load."
        )
        
        ax_header.text(0.5, 0.85, main_title, ha='center', fontsize=26, weight='bold', color='#2c3e50')
        ax_header.text(0.5, 0.45, textwrap.fill(summary_text, width=90), ha='center', va='top', fontsize=14, style='italic', color='#34495e')

        # Component Definitions
        sections = [
            {
                "comp": "P100", "ch": "OZ", "color": "green", "window": (0.08, 0.14),
                "title": "A. P100 (The 'First Glance' Test) @ OZ",
                "desc": "Measures how physically overwhelming the screen is—telling us if the sheer amount of clutter is tiring the user's eyes before they even start reading. Amplitude is sensitive to low-level visual features."
            },
            {
                "comp": "N200", "ch": "FZ", "color": "yellow", "window": (0.20, 0.30),
                "title": "B. N200 (The 'Confusion' Test) @ FZ",
                "desc": "Measures mental friction and conflict—revealing the exact moment a user gets stuck or frustrated because they can't instantly resolve the information needed for the task."
            },
            {
                "comp": "P300", "ch": "PZ", "color": "red", "window": (0.30, 0.50),
                "title": "C. P300 (The 'Confidence' Test) @ PZ",
                "desc": "Measures the 'Aha!' moment and resource allocation—proving the user has successfully understood the data and is ready to make a confident decision. Faster latency means quicker insight."
            }
        ]
        
        row_indices = [(1, 2), (3, 4), (5, 6)]

        for i, sec in enumerate(sections):
            text_row, graph_row = row_indices[i]
            channel = sec["ch"]
            
            # --- TEXT ROW (COMPONENT HEADER) ---
            ax_text = fig.add_subplot(gs[text_row])
            ax_text.axis('off') 
            
            ax_text.text(0.5, 0.75, sec["title"], ha='center', fontsize=20, weight='bold', color='#2c3e50')
            ax_text.text(0.5, 0.25, textwrap.fill(sec["desc"], width=100), ha='center', va='top', fontsize=14, color='#7f8c8d')

            # --- GRAPH ROW ---
            if channel in raw.ch_names:
                ax_graph = fig.add_subplot(gs[graph_row])
                
                # Plot the ERP comparison
                mne.viz.plot_compare_evokeds(
                    {'Target': evoked_target, 'Non-Target': evoked_nontarget}, 
                    picks=channel, 
                    axes=ax_graph, 
                    show=False, 
                    show_sensors=False, 
                    legend='upper right',
                    title=None,
                    scalings=dict(eeg=1e6) # CRITICAL: Display in microvolts (µV)
                )
                
                # Formatting and Highlighting
                ax_graph.ticklabel_format(style='plain', axis='y')
                ax_graph.axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.15, label=f'{sec["comp"]} Window')
                ax_graph.set_xlim(-0.2, 0.6)
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
                # Handle case where required channel is not present
                ax_graph = fig.add_subplot(gs[graph_row])
                ax_graph.text(0.5, 0.5, f'Channel {channel} not found in data', 
                              ha='center', va='center', fontsize=14, color='red')
                ax_graph.axis('off')

        # Add metadata footer with trial balance info
        balance_note = ""
        # Alert if severely imbalanced (less than 10 trials in either condition)
        if target_count < 10 or nontarget_count < 10:
            balance_note = " ⚠️ LOW TRIAL COUNT WARNING"
        
        fig.text(0.5, 0.01, 
                 f'Total Epochs: {len(epochs)} | Target: {target_count} | Non-Target: {nontarget_count}{balance_note}', 
                 ha='center', fontsize=10, style='italic', color='#7f8c8d')

        # 7. CONVERT TO IMAGE (PNG)
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=300) 
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
                "artifact_rejection": "disabled (all trials included)"
            }
        }

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred: {e}")
        # Raise a 500 HTTPException for external facing errors
        raise HTTPException(status_code=500, detail=f"Server analysis error: {str(e)}")
        
    finally:
        # Clean up temporary files
        if tmp_cnt_path and os.path.exists(tmp_cnt_path): 
            os.remove(tmp_cnt_path)
        if tmp_exp_path and os.path.exists(tmp_exp_path): 
            os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)