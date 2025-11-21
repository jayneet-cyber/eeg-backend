import matplotlib
# OPTIMIZATION 3: Set backend to 'Agg' to prevent GUI errors on headless servers
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

# OPTIMIZATION 1: Use synchronous 'def' to run in thread pool (prevents blocking)
@app.post("/analyze")
def analyze_eeg(cnt_file: UploadFile = File(...), exp_file: UploadFile = File(...)):
    
    # --- 1. FILE HANDLING ---
    # Save uploads to temporary files on disk (MNE requires file paths)
    try:
        tmp_cnt = tempfile.NamedTemporaryFile(delete=False, suffix=".cnt")
        tmp_cnt.close()
        tmp_cnt_path = tmp_cnt.name

        tmp_exp = tempfile.NamedTemporaryFile(delete=False, suffix=".exp")
        tmp_exp.close()
        tmp_exp_path = tmp_exp.name

        # Write content using shutil for efficiency
        with open(tmp_cnt_path, "wb") as buffer:
            shutil.copyfileobj(cnt_file.file, buffer)
            
        with open(tmp_exp_path, "wb") as buffer:
            shutil.copyfileobj(exp_file.file, buffer)

        # --- 2. LOAD & PARSE ---
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        
        # Parse .exp file to map Trial IDs -> Conditions (R vs C) and get Reaction Times
        trial_type_map = {}
        reaction_times = []
        
        with open(tmp_exp_path, 'r') as f:
            lines = f.readlines()
            for line in lines[8:]: # Skip 8 header lines
                parts = line.strip().split('\t')
                if len(parts) < 7: parts = line.strip().split() # Fallback split
                if len(parts) >= 7:
                    t_id = parts[0].strip()      # Trial ID
                    t_name = parts[1].strip()    # Trial Name/Stimulus
                    t_type = parts[3].strip()    # Trial Type (e.g., R for Target)
                    try: 
                        t_lat = int(parts[6].strip()) # Reaction Latency (ms)
                    except ValueError: 
                        t_lat = 1000 # Use a high value for missing/invalid RT

                    trial_type_map[t_id] = t_type
                    # Collect valid reaction times for Targets (R)
                    if t_type == 'R' and t_lat < 1000:
                        reaction_times.append((t_lat, t_id, t_name))

        # --- 3. CALCULATE METRICS (Easiest/Toughest) ---
        easiest_txt = "N/A"
        toughest_txt = "N/A"
        if reaction_times:
            best = min(reaction_times, key=lambda x: x[0]) # Min latency
            worst = max(reaction_times, key=lambda x: x[0]) # Max latency
            
            easiest_txt = f"Trial {best[1]}: '{best[2]}' ({best[0]}ms)"
            toughest_txt = f"Trial {worst[1]}: '{worst[2]}' ({worst[0]}ms)"

        # --- 4. CREATE MNE EVENTS ---
        new_events_list = []
        for annot in raw.annotations:
            clean_id = str(annot['description']).strip()
            sType = trial_type_map.get(clean_id, "Unknown")
            if sType == "Unknown": continue
            code = 1 if sType == 'R' else 2 # 1=Target, 2=Non-Target
            new_events_list.append([raw.time_as_index(annot['onset'])[0], 0, code])

        if not new_events_list:
            raise HTTPException(status_code=400, detail="No matching events found in .exp file")

        custom_events = np.array(new_events_list)
        event_ids = {'Target': 1, 'Non-Target': 2}

        # --- 5. PREPROCESSING (Filter & Epoch) ---
        # OPTIMIZATION 2: n_jobs=-1 uses all CPU cores
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

        # --- 6. PLOTTING (REPORT STYLE) ---
        
        # CONFIG: Figure Size
        # height=30 gives us a long, scrollable report format
        fig, ax = plt.subplots(3, 1, figsize=(12, 30))
        
        # CONTENT: Header Text
        main_title = "Neuro-UX: B2B Dashboard Analysis"
        summary_text = (
            "B2B Dashboard Analysis Summary\n\n"
            "This analysis uses key ERP components to test dashboard usability: "
            "The **P100** gauges visual clutter, the **N200** detects moments of user confusion, "
            "and the **P300** measures cognitive resource allocation and decision confidence. "
            "A faster P300 latency and higher P300 amplitude for 'Target' trials indicate superior design and lower cognitive load."
        )
        
        # PLACE HEADER
        # 0.97 is near the very top edge. 0.93 is slightly below it.
        fig.text(0.5, 0.97, main_title, ha='center', fontsize=24, weight='bold', color='#2c3e50')
        fig.text(0.5, 0.93, textwrap.fill(summary_text, width=95), ha='center', va='top', fontsize=13, style='italic', color='#34495e')

        # CONTENT: Section Definitions
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
        
        # LOOP: Create graphs
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
                # Plot Graph lines
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
                
                # Highlight Window (Colored background)
                ax[i].axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.1, label=f"{sec['comp']} Window")
                
                # --- TEXT PLACEMENT ---
                # Title Position: 1.4 (High above graph)
                ax[i].text(0.5, 1.4, sec["title"], transform=ax[i].transAxes, ha='center', va='bottom', fontsize=18, weight='bold', color='#2c3e50')
                
                # Description Position: 1.15 (Between title and graph)
                wrapped_desc = textwrap.fill(sec["desc"], width=85)
                ax[i].text(0.5, 1.15, wrapped_desc, transform=ax[i].transAxes, ha='center', va='top', fontsize=12, color='#7f8c8d',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#bdc3c7", alpha=0.8))

        # --- LAYOUT ADJUSTMENT ---
        # top=0.78: Pushes graphs down to make room for the Main Header text.
        # hspace=0.8: Adds vertical gap between graphs A, B, and C.
        plt.subplots_adjust(top=0.78, hspace=0.8, bottom=0.05)
        
        # --- 7. EXPORT ---
        buf = BytesIO()
        # dpi=150 ensures text is crisp
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=150) 
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
        return {"error": str(e)}
    finally:
        # Cleanup temp files to save space
        if 'tmp_cnt_path' in locals() and os.path.exists(tmp_cnt_path): 
            os.remove(tmp_cnt_path)
        if tmp_exp_path and os.path.exists(tmp_exp_path): 
            os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)