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

        # 5. FILTER & EPOCH (With Artifact Rejection)
        raw.filter(0.1, 30.0, picks='eeg', n_jobs=-1, verbose=False)
        
        # Define rejection threshold (e.g., 100 microvolts)
        reject_criteria = dict(eeg=100e-6) 
        
        epochs = mne.Epochs(
            raw, 
            custom_events, 
            event_ids, 
            tmin=-0.2, 
            tmax=0.6, 
            baseline=(None, 0), 
            picks='eeg', 
            preload=True, 
            reject=reject_criteria, 
            verbose=False
        )
        
        if len(epochs) == 0:
            return {"error": "All trials were rejected due to artifacts (too much noise)."}

        # FIX: Removed manual scaling block. MNE handles Volts -> Microvolts automatically.
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()

        # 6. PLOT (REFINED GRID LAYOUT WITH FIXES)
        
        fig = plt.figure(figsize=(12, 32)) 
        
        # GridSpec: Rows 1,3,5 (Text) are height 1.0. Rows 2,4,6 (Graphs) are height 2.5.
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
                "desc": "Measures how physically overwhelming the screen is—telling us if the sheer amount of clutter is tiring the user's eyes before they even start reading."
            },
            {
                "comp": "N200", "ch": "FZ", "color": "yellow", "window": (0.20, 0.30),
                "title": "B. N200 (The 'Confusion' Test)",
                "desc": "Measures mental friction—revealing the exact moment a user gets stuck or frustrated because they can't instantly find the insight they need in a wall of numbers."
            },
            {
                "comp": "P300", "ch": "PZ", "color": "red", "window": (0.30, 0.50),
                "title": "C. P300 (The 'Confidence' Test)",
                "desc": "Measures the 'Aha!' moment—proving the user has successfully understood the data and is ready to make a confident decision, rather than hesitating."
            }
        ]
        
        row_indices = [(1, 2), (3, 4), (5, 6)]

        # Calculate dynamic y-limits based on actual data
        # FIX: We manually scale by 1e6 HERE just for the Y-axis MATH, 
        # but we do NOT modify the 'evoked' objects themselves.
        all_data = np.concatenate([
            evoked_target.get_data(picks=sec["ch"]).flatten() * 1e6
            if sec["ch"] in evoked_target.ch_names else np.array([0])
            for sec in sections
        ] + [
            evoked_nontarget.get_data(picks=sec["ch"]).flatten() * 1e6
            if sec["ch"] in evoked_nontarget.ch_names else np.array([0])
            for sec in sections
        ])
        
        # Set y-limits with 20% padding
        y_min = all_data.min() * 1.2 if all_data.min() < 0 else all_data.min() * 0.8
        y_max = all_data.max() * 1.2
        
        # Ensure reasonable limits (fallback to defaults if data is too flat)
        if abs(y_max - y_min) < 1:
            y_min, y_max = -10, 35

        for i, sec in enumerate(sections):
            text_row, graph_row = row_indices[i]
            channel = sec["ch"]
            
            # --- TEXT ROW (CENTERED) ---
            ax_text = fig.add_subplot(gs[text_row])
            ax_text.axis('off') 
            
            # Centered text with better vertical positioning
            ax_text.text(0.5, 0.75, sec["title"], ha='center', fontsize=20, weight='bold', color='#2c3e50')
            ax_text.text(0.5, 0.25, textwrap.fill(sec["desc"], width=100), ha='center', va='top', fontsize=14, color='#7f8c8d')

            # --- GRAPH ROW ---
            if channel in raw.ch_names:
                ax_graph = fig.add_subplot(gs[graph_row])
                
                # Plot with legend on all graphs for clarity
                mne.viz.plot_compare_evokeds(
                    {'Target': evoked_target, 'Non-Target': evoked_nontarget}, 
                    picks=channel, 
                    axes=ax_graph, 
                    show=False, 
                    show_sensors=False, 
                    legend='upper right',
                    title=None
                )
                
                # MNE plots in uV automatically, so we don't need to change data.
                # Just ensure tick labels are simple numbers.
                ax_graph.ticklabel_format(style='plain', axis='y')
                
                # Highlight component time window
                ax_graph.axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.15, label=f'{sec["comp"]} Window')
                
                # FIXED: Use dynamic y-limits (calculated in uV above)
                ax_graph.set_xlim(-0.2, 0.6)
                ax_graph.set_ylim(y_min, y_max)
                
                # Add zero line for reference
                ax_graph.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
                
                # Styling - Lighter Grid
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

        # Add metadata footer
        fig.text(0.5, 0.01, f'Total Epochs: {len(epochs)} | Target: {len(epochs["Target"])} | Non-Target: {len(epochs["Non-Target"])}', 
                 ha='center', fontsize=10, style='italic', color='#7f8c8d')

        # 7. CONVERT TO IMAGE
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
                "target_epochs": len(epochs["Target"]),
                "nontarget_epochs": len(epochs["Non-Target"]),
                "channels_analyzed": [sec["ch"] for sec in sections if sec["ch"] in raw.ch_names]
            }
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'tmp_cnt_path' in locals() and os.path.exists(tmp_cnt_path): 
            os.remove(tmp_cnt_path)
        if 'tmp_exp_path' in locals() and os.path.exists(tmp_exp_path): 
            os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)