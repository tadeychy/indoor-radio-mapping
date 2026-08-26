# indoor-radio-mapping
# Quick Start Guide

## 1. Setup Environment (Python 3.11)
Open your terminal in the project folder and run:

```bash
# Create and activate environment
python3.11 -m venv venv
source venv/bin/activate      # Mac/Linux
.\venv\Scripts\activate       # Windows

# Install packages
pip install -r requirements.txt
```
[Download 3D Models (pcd-2.zip) and extract them into ply folder](https://www.dropbox.com/scl/fi/6efup52i97dt29odejgwr/pcd-2.zip?rlkey=d3a7yv1icinnfz9j2iyq78le7&st=9abtapqd&dl=0)
## 2. Mandatory Pre-Run Steps
You must manually configure your layout tracking, asset paths, and threshold tolerances before launching scripts:

### A. Format & Set Up Your 3D Scan Data (`prepare_model.py` / `train.py`)
1. **Mesh Prep:** If you scanned multiple rooms, use `ply/merging.py` to join them. Convert point clouds to meshes using `point_to_triangle` from `functions.py`, clean up artifacts using a cutting function, and use the rotate tool (`-90°` on X, `205°` on Y for KIRI Engine scans) if your mesh orientation is misaligned.
2. **Matrix Configuration:** Open `prepare_dataset.py` or your training script. Adjust the spatial coordinate matrix parameters, feature scaling profiles, your targeted input `.csv` log dataset paths, and the output file title to synchronize with your specific grid layout.

### B. Map Prediction Parameters (`final.py`)
Open `final.py` and modify the inline parameters to point to your specific assets:
* Set your active evaluation mesh file path.
* Set your verified neural network weight checkpoint file path.
* Set your target custom signal coverage threshold restriction (in dBm).

## 3. Run the Project
Once configuration parameters match your target scenario, execute the scripts sequentially:

```bash
# Step 1: Process geometric features and compile model weights
python prepare_dataset.py

# Step 2: Calculate coordinates and render the interactive 3D map
python final.py
```
