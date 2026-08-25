import os
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import functions as IJS

# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class RSSIEstimator(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # Single RSSI output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ==========================================
# 2. CONFIGURATION & CONSTANTS
# ==========================================
FREQ_BAND = "2.4"
PLY_MESH_PATH = "ply/mesh/final_mesh_rotated_cut.ply"
CSV_DATA_PATH = f"messurment data/2.4.csv"
PRETRAINED_MODEL_PATH = f"pt model/rssi_estimator{FREQ_BAND}G.pt"
SAVED_MODEL_PATH = f"pt model/rssi_estimator{FREQ_BAND}G_updated.pt"

# Grid measurement layout matrix
AP_LOCATIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23, 25]
MATRIX_INDS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 18, 0, 0, 0, 19, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 15, 0, 0, 0, 16, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 24, 0, 0, 0, 25, 0, 0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 21, 0, 0, 0, 22, 0, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

# Spatial offset scaling parameters (in meters)
GRID_SPACING = 0.25
X_OFFSET = 4.5
Y_OFFSET = 3.5
Z_OFFSET = 0.5


# ==========================================
# 3. MAIN TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    print("--- 1. Loading Mesh and RSSI Measurement Data ---")
    mesh = o3d.io.read_triangle_mesh(PLY_MESH_PATH)

    # Read CSV RSSI strength data
    df = pd.read_csv(CSV_DATA_PATH, encoding="utf-16")
    cleaned_x = [int(i.replace("dBm", "")) for i in df["Strength"].tolist()]

    # Construct measurement data groups (setups)
    data_groups = []
    matrix = MATRIX_INDS
    for setup in range(20):
        index_search = setup - 1 if setup <= 20 else setup - 21
        matrix2 = [row[:] for row in matrix]
        for r in range(27):
            for c in range(30):
                if matrix2[r][c] != 0:
                    matrix2[r][c] = cleaned_x[(matrix[r][c] - 1) + (26 * index_search)]
        data_groups.append(np.array(matrix2, dtype=np.float64))

    print(f"Loaded {len(data_groups)} AP measurement setups.")

    print("\n--- 2. Computing 3D Spatial Feature Maps (Raycasting) ---")
    feature_maps = IJS.generate_ap_feature_maps(
        data_groups, MATRIX_INDS, AP_LOCATIONS, mesh,
        grid_spacing=GRID_SPACING, x_offset=X_OFFSET, y_offset=Y_OFFSET, z_offset=Z_OFFSET, ap_z=0
    )

    # Prepare feature matrix X (Distance, Wall Count) and targets y (RSSI)
    X, y, setup_indices = IJS.prepare_dataset(data_groups, feature_maps)

    # Split dataset by setup configurations (train: 0-15, val: 16-17, test: 18-19)
    train_mask = setup_indices <= 15
    val_mask   = (setup_indices == 16) | (setup_indices == 17)
    test_mask  = (setup_indices == 18) | (setup_indices == 19)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"Dataset split size:")
    print(f"  Train : {len(X_train)} samples")
    print(f"  Val   : {len(X_val)} samples")
    print(f"  Test  : {len(X_test)} samples")

    # Fit feature scaling
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s   = scaler_X.transform(X_val)
    X_test_s  = scaler_X.transform(X_test)

    # PyTorch DataLoaders
    train_loader = DataLoader(IJS.to_tensors(X_train_s, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(IJS.to_tensors(X_val_s,   y_val),   batch_size=32)
    test_loader  = DataLoader(IJS.to_tensors(X_test_s,  y_test),  batch_size=32)

    print("\n--- 3. Loading Pre-trained PyTorch Model ---")
    model = RSSIEstimator(input_dim=2, hidden_dims=[64, 128, 64], dropout=0.2)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        checkpoint = torch.load(PRETRAINED_MODEL_PATH, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded existing model weights from: {PRETRAINED_MODEL_PATH}")
    else:
        print(f"No checkpoint found at {PRETRAINED_MODEL_PATH}. Initializing fresh weights.")

    print("\n--- 4. Fine-Tuning Model with Additional Data ---")
    # Fine-tune with a lower learning rate
    model, history = IJS.train_model(
        model,
        train_loader,
        val_loader,
        epochs=150,
        lr=1e-4,
        patience=25
    )

    print("\n--- 5. Model Evaluation ---")
    preds_val,  targets_val,  metrics_val  = IJS.evaluate_model(model, val_loader,  scaler_X, None, "Validation")
    preds_test, targets_test, metrics_test = IJS.evaluate_model(model, test_loader, scaler_X, None, "Test")

    # Visualize training history and prediction correlation
    IJS.plot_results(history, preds_val, targets_val, preds_test, targets_test)

    print("\n--- 6. Saving Fine-Tuned Model ---")
    torch.save({
        "model_state": model.state_dict(),
        "scaler_X": scaler_X,
        "metrics_val": metrics_val,
        "metrics_test": metrics_test
    }, SAVED_MODEL_PATH)
    print(f"Successfully saved updated model to: {SAVED_MODEL_PATH}")
