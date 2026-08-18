from scipy.optimize import differential_evolution
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import functions as IJS
list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 23, 25]
matrix_inds = [[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  7,  0,  0,  0,  8,  0,  0,  0,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  4,  0,  0,  0,  5,  0,  0,  0,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  11,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
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
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0],
 [ 0,  0,  0,  21,  0,  0,  0,  22,  0,  0,  0,  23,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0]]

# ﷿﷿﷿﷿﷿﷿ MLP Model ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿
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
        layers.append(nn.Linear(prev_dim, 1))  # single RSSI output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)
mesh = o3d.io.read_triangle_mesh('pt model/MHAB_rotated.ply')
matrix_ind_np = np.array(matrix_inds, dtype=np.float64)
y_indices, x_indices = np.nonzero(matrix_ind_np)
#SCALING
grid_spacing = 0.25
x_offset = 4.5
y_offset = 3.5
z_offset = 0.4

# Generate point array from scaling
points_meas = np.column_stack((
    -y_indices * grid_spacing + x_offset,
    -x_indices * grid_spacing + y_offset,
    np.full(len(x_indices),  z_offset)
))
data_groups = []
df = pd.read_csv("sorted office messurment data/2.4.csv", encoding="utf-16")
x = df["Strength"].tolist()
cleaned_x = [int(i.replace("dBm", "")) for i in x]
for i in range(20):
    setup = i
    if setup <= 20:
        index_search = setup - 1
    else:
        index_search = setup - 21
    matrix2 = [row[:] for row in matrix_inds]
    for i in range(0, 27):
        for col in range(0, 30):
            if matrix2[i][col] != 0:
                matrix2[i][col] = cleaned_x[(matrix_inds[i][col] - 1) + (26 * (index_search))]
    data_group = np.array(matrix2, dtype=np.float64)
    data_groups.append(data_group)
feature_maps = IJS.generate_ap_feature_maps(data_groups, matrix_inds, list, mesh,
                                        grid_spacing=grid_spacing, x_offset=x_offset, y_offset=y_offset, z_offset=z_offset, ap_z=0)

# Prepare full dataset
X, y, setup_indices = IJS.prepare_dataset(data_groups, feature_maps)
print(setup_indices)
print(X)
print(y)
# Split by setup index: train=0-15, val=16-17, test=18-19
train_mask = setup_indices <= 15
val_mask   = (setup_indices == 16) | (setup_indices == 17)
test_mask  = (setup_indices == 18) | (setup_indices == 19)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s   = scaler_X.transform(X_val)
X_test_s  = scaler_X.transform(X_test)
test_loader  = DataLoader(IJS.to_tensors(X_test_s,  y_test),  batch_size=32)
# Define search space ﷿﷿﷿ AP can be anywhere within the room bounds
# Load the entire model in one line

checkpoint = torch.load("rssi_estimator2.4G.pt", map_location=torch.device('cpu'), weights_only=False)

# 2. Recreate the blank model skeleton (Replace MyMLPClass with your actual class name!)
input_dim = X_train.shape[1]
print(input_dim)
model = RSSIEstimator(input_dim=input_dim)

# 3. Pull the weights out of the dictionary (change 'model_state_dict' if your print showed a different key)
model.load_state_dict(checkpoint['model_state'])
results, estimators = IJS.evaluate_all(
    X_train_s, y_train,
    X_val_s,   y_val,
    X_test_s,  y_test,
    mlp_model=model,
    test_loader=test_loader
)
print(results)
print(type(results))
IJS.plot_comparison(results)
#x_bounds = [(0, 30000)]  # in mm
#y_bounds = [(0, 28000)]
#z_bounds = [(1000, 2500)]     #realistic AP mounting height
svr_model = estimators[('SVR')]
# points_meas is already (N, 3) in the correct coordinate system
grid_points = points_meas.astype(np.float32)

x_min, x_max = grid_points[:, 0].min(), grid_points[:, 0].max()
y_min, y_max = grid_points[:, 1].min(), grid_points[:, 1].max()
#z_bounds = [(1000 + z_offset, 2500 + z_offset)]  # account for z_offset
z_min, z_max = grid_points[:, 2].min(), grid_points[:,2].max()# account for z_offset
z_bounds = [(z_min, z_max)]
scaler_X = StandardScaler()
result = differential_evolution(
    IJS.coverage_score,
    bounds=[(x_min, x_max), (y_min, y_max)] + [(z_min, z_max)],
#    bounds=[x_bounds, y_bounds] +z_bounds,
    args=(mesh, grid_points, svr_model, scaler_X),
    maxiter=100,
    popsize=15,
    seed=42,
    disp=True
)
print(result.x)
optimal_ap = result.x
print(f"Optimal AP position: {optimal_ap}")
print(f"Coverage score: {-result.fun:.1%}")
#Plot vizualization
pv_mesh = IJS.visualize_optimal_ap(
    mesh        = mesh,
    grid_points = grid_points,
    optimal_ap  = optimal_ap,
    svr_model   = svr_model,
    scaler_X    = scaler_X,
    rssi_threshold = -50,
    z_offset    = z_offset
)
