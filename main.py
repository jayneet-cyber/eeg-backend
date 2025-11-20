import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mne
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os
import base64
from io import BytesIO

app = FastAPI()

# CRITICAL: Allow Lovable (and everyone else) to talk to this server
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
async def analyze_eeg(cnt_file: UploadFile = File(...), exp_file: UploadFile = File(...)):
    # 1. SAVE UPLOADS TEMP
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cnt") as tmp_cnt:
        tmp_cnt.write(await cnt_file.read())
        tmp_cnt_path = tmp_cnt.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".exp") as tmp_exp:
        tmp_exp.write(await exp_file.read())
        tmp_exp_path = tmp_exp.name

    try:
        # 2. LOAD DATA
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        
        # 3. PARSE EXP & FIND REACTION TIMES
        trial_type_map = {}
        reaction_times = []  # List to store (Latency, TrialID, ImageName)
        
        with open(tmp_exp_path, 'r') as f:
            lines = f.readlines()
            for line in lines[8:]: # Skip header
                parts = line.strip().split('\t')
                # Fallback if tab split fails
                if len(parts) < 7: parts = line.strip().split()
                
                # We need at least 7 columns now to get Latency (Col 6) and Name (Col 1)
                if len(parts) >= 7:
                    t_id = parts[0].strip()
                    t_name = parts[1].strip()
                    t_type = parts[3].strip()
                    
                    # Try to parse latency, default to 1000 if fails
                    try:
                        t_lat = int(parts[6].strip())
                    except:
                        t_lat = 1000
                    
                    trial_type_map[t_id] = t_type
                    
                    # LOGIC: "Toughness" is determined by Reaction Time on TARGETS (R)
                    # We only count it if they actually pressed the button (Latency < 1000)
                    if t_type == 'R' and t_lat < 1000:
                        reaction_times.append((t_lat, t_id, t_name))

        # Calculate Easiest/Toughest based on sorted reaction times
        easiest_txt = "N/A"
        toughest_txt = "N/A"
        
        if reaction_times:
            reaction_times.sort() # Sorts by Latency (first item in tuple)
            
            # Easiest = Fastest Time (Smallest Number)
            best = reaction_times[0]
            easiest_txt = f"Trial {best[1]}: '{best[2]}' ({best[0]}ms)"
            
            # Toughest = Slowest Time (Largest Number)
            worst = reaction_times[-1]
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

        # 5. FILTER & EPOCH
        raw.filter(0.1, 30.0, picks='eeg', verbose=False)
        epochs = mne.Epochs(raw, custom_events, event_ids, tmin=-0.2, tmax=0.6, baseline=(None, 0), picks='eeg', preload=True, verbose=False)
        
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()

        # 6. PLOT
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        components = {
            "P100 (Visual Input)": "OZ", 
            "N200 (Categorization)": "FZ", 
            "P300 (Decision)": "PZ"
        }
        
        for i, (name, ch) in enumerate(components.items()):
            if ch in raw.ch_names:
                # Plot the ERP lines
                mne.viz.plot_compare_evokeds(
                    {'Target': evoked_target, 'Non-Target': evoked_nontarget}, 
                    picks=ch, 
                    axes=ax[i], 
                    show=False, 
                    show_sensors=False, 
                    title=name,
                    legend='upper left' if i == 0 else None
                )
                
                # --- METRICS CALCULATION ---
                target_data = evoked_target.get_data(picks=ch)[0]
                times = evoked_target.times
                
                metric_text = ""
                
                if "P300" in name:
                    # Look for peak between 300ms and 500ms
                    mask = (times >= 0.3) & (times <= 0.5)
                    # Highlight Window
                    ax[i].axvspan(0.30, 0.50, color='red', alpha=0.1, label="P300 Window")
                    
                    # Find Max Amplitude and Index
                    window_data = target_data[mask]
                    window_times = times[mask]
                    
                    if len(window_data) > 0:
                        peak_amp = np.max(window_data) * 1e6 
                        peak_lat = window_times[np.argmax(window_data)] * 1000 
                        metric_text = f"Target Peak: {peak_amp:.2f} µV\nLatency: {peak_lat:.0f} ms"

                elif "P100" in name:
                    ax[i].axvspan(0.08, 0.14, color='green', alpha=0.1)
                elif "N200" in name:
                    ax[i].axvspan(0.20, 0.30, color='yellow', alpha=0.1)

                # Add Text Box to Plot
                if metric_text:
                    ax[i].text(0.95, 0.05, metric_text, 
                             transform=ax[i].transAxes, 
                             fontsize=12, 
                             verticalalignment='bottom', 
                             horizontalalignment='right',
                             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        
        # 7. CONVERT TO IMAGE
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")

        # RETURN IMAGE + NEW STATS
        return {
            "status": "success", 
            "image": img_str,
            "easiest": easiest_txt,
            "toughest": toughest_txt
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(tmp_cnt_path): os.remove(tmp_cnt_path)
        if os.path.exists(tmp_exp_path): os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)