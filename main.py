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
    # We need to save the uploaded bytes to physical files because MNE reads from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cnt") as tmp_cnt:
        tmp_cnt.write(await cnt_file.read())
        tmp_cnt_path = tmp_cnt.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".exp") as tmp_exp:
        tmp_exp.write(await exp_file.read())
        tmp_exp_path = tmp_exp.name

    try:
        # 2. LOAD DATA
        # Read the raw EEG data
        raw = mne.io.read_raw_cnt(tmp_cnt_path, preload=True, verbose=False)
        
        # 3. PARSE EXP FILE
        # Read the metadata to find out which event is Target (R) vs Non-Target (C)
        trial_type_map = {}
        with open(tmp_exp_path, 'r') as f:
            lines = f.readlines()
            for line in lines[8:]: # Skip the first 8 header lines
                parts = line.strip().split('\t')
                if len(parts) < 4: parts = line.strip().split()
                if len(parts) >= 4:
                    trial_type_map[parts[0].strip()] = parts[3].strip()

        # 4. MAP EVENTS
        # Match the EEG triggers with the Experiment Map
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

        # 5. PROCESSING (Filter -> Epoch -> Average)
        # Filter: Keep 0.1 Hz to 30 Hz
        raw.filter(0.1, 30.0, picks='eeg', verbose=False)
        
        # Epoch: Cut from -0.2s to +0.6s
        epochs = mne.Epochs(raw, custom_events, event_ids, tmin=-0.2, tmax=0.6, baseline=(None, 0), picks='eeg', preload=True, verbose=False)
        
        evoked_target = epochs['Target'].average()
        evoked_nontarget = epochs['Non-Target'].average()

        # 6. PLOTTING
        # Create the graph for P100, N200, P300
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        components = {
            "P100 (Visual Input)": "OZ", 
            "N200 (Categorization)": "FZ", 
            "P300 (Decision)": "PZ"
        }
        
        for i, (name, ch) in enumerate(components.items()):
            if ch in raw.ch_names:
                mne.viz.plot_compare_evokeds({'Target': evoked_target, 'Non-Target': evoked_nontarget}, picks=ch, axes=ax[i], show=False, title=name)

        plt.tight_layout()
        
        # 7. RETURN IMAGE
        # Convert the plot to a string (Base64) so the website can display it
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")

        return {"status": "success", "image": img_str}

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Cleanup: Delete the temporary files
        if os.path.exists(tmp_cnt_path): os.remove(tmp_cnt_path)
        if os.path.exists(tmp_exp_path): os.remove(tmp_exp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)