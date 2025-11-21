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

        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()

        # 6. PLOT (IMPROVED GRID SPEC LAYOUT)
        
        fig = plt.figure(figsize=(12, 35)) # Increased height to 35 for more breathing room
        
        # 7 Rows: Header, TextA, GraphA, TextB, GraphB, TextC, GraphC
        # Adjusted ratios to give Text more height and Graphs more isolation
        gs = gridspec.GridSpec(7, 1, height_ratios=[0.8, 0.5, 2, 0.5, 2, 0.5, 2], hspace=0.6)

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
        
        # Align Header Center
        ax_header.text(0.5, 0.7, main_title, ha='center', fontsize=26, weight='bold', color='#2c3e50')
        ax_header.text(0.5, 0.3, textwrap.fill(summary_text, width=90), ha='center', va='top', fontsize=14, style='italic', color='#34495e')

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
                "desc": "Measures mental friction—revealing the exact moment a user gets stuck or frustrated because they can’t instantly find the insight they need in a wall of numbers."
            },
            {
                "comp": "P300", "ch": "PZ", "color": "red", "window": (0.30, 0.50),
                "title": "C. P300 (The 'Confidence' Test)",
                "desc": "Measures the 'Aha!' moment—proving the user has successfully understood the data and is ready to make a confident decision, rather than hesitating."
            }
        ]
        
        row_indices = [(1, 2), (3, 4), (5, 6)]

        for i, sec in enumerate(sections):
            text_row, graph_row = row_indices[i]
            channel = sec["ch"]
            
            # --- TEXT ROW ---
            ax_text = fig.add_subplot(gs[text_row])
            ax_text.axis('off') 
            
            # ALIGNMENT FIX: x=0.05 aligns text with the left edge of the graph (approximately)
            ax_text.text(0.05, 0.6, sec["title"], ha='left', fontsize=20, weight='bold', color='#2c3e50')
            
            # WIDTH FIX: width=100 makes the text span the full page width
            ax_text.text(0.05, 0.2, textwrap.fill(sec["desc"], width=100), ha='left', va='top', fontsize=14, color='#7f8c8d')

            # --- GRAPH ROW ---
            if channel in raw.ch_names:
                ax_graph = fig.add_subplot(gs[graph_row])
                
                # Plot
                mne.viz.plot_compare_evokeds(
                    {'Target': evoked_target, 'Non-Target': evoked_nontarget}, 
                    picks=channel, 
                    axes=ax_graph, 
                    show=False, 
                    show_sensors=False, 
                    legend='upper right' if i == 0 else None,
                    title=None
                )
                
                # Highlight
                ax_graph.axvspan(sec["window"][0], sec["window"][1], color=sec["color"], alpha=0.1)
                
                # STRICT AXIS LIMITS
                ax_graph.set_xlim(-0.2, 0.6)
                ax_graph.set_ylim(-10, 35) 
                
                # Clean up graph styling
                ax_graph.spines['top'].set_visible(False)
                ax_graph.spines['right'].set_visible(False)
                ax_graph.grid(True, linestyle='--', alpha=0.5)
                ax_graph.set_ylabel("Amplitude (µV)", fontsize=12)
                ax_graph.set_xlabel("Time (s)", fontsize=12)
                
                # Increase tick label size for readability
                ax_graph.tick_params(axis='both', which='major', labelsize=10)

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
            "toughest": toughest_txt
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