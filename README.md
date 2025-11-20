🧠 EEG Cognitive Load Backend

This is the computational code for the NeuroLoad Analyzer project. It is a Python-based API built with FastAPI and MNE-Python.

What it does

This backend acts as a bridge between a web frontend (Lovable/React) and complex scientific Python libraries.

Receives Files: Accepts raw EEG data (.cnt) and experiment logs (.exp) via API uploads.

Processes Data: Runs a full neuroscience pipeline

Preprocessing: Applies Bandpass filters (0.1–30 Hz).

Epoching: Slices the continuous data into time-locked windows. As per your exp or log table file.

Averaging: Computes the Grand Average ERP (Event-Related Potential).

Visualizes: Generates scientific plots for P100 (Visual), N200 (Categorization), and P300 (Decision/Attention) components.

Returns Results on the webpage


Tech Stack

Framework: FastAPI

Neuroscience: MNE-Python

Data Science: NumPy, Matplotlib, SciPy
