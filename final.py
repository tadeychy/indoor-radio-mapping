import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from scipy.optimize import differential_evolution
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
        layers.append(nn.Linear(prev_dim, 1))  # single RSSI output
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)





freq_band = ("2.4")

"""
#loading pcd
file_path_pcd = "ply/pcd/final_pcd.ply"
pcd = o3d.io.read_point_cloud(file_path)
o3d.visualization.draw_geometries([pcd])
#rotating pcd
pcd_rot = IJS.rotate(x, -90, 205,0,save="ply/pcd/final_pcd_rot.ply")
o3d.visualization.draw_geometries([pcd_rot])
#pcd->mesh
IJS.point_to_triangle("ply/pcd/final_pcd_rot.ply",save="ply/mesh/final_pcd_rot2.ply")
mesh = o3d.io.read_triangle_mesh("ply/mesh/final_pcd_rot2.ply")
o3d.visualization.draw_geometries([mesh])
#cuting
IJS.cuting("ply/mesh/final_pcd_rot2.ply",0.2,"ply/mesh/final_pcd_rot3.ply")
"""
mesh = o3d.io.read_triangle_mesh("ply/mesh/final_pcd_rot3.ply")
o3d.visualization.draw_geometries([mesh])

# ==========================================
# 2. PURE 3D GEOMETRY HELPER FUNCTIONS
# ==========================================

def generate_grid_from_mesh(mesh, spacing=0.25, height=0.4):
    """
    Generates evaluation grid points across room bounds derived directly from 3D mesh.
    """
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b = bounds.get_min_bound()
    max_b = bounds.get_max_bound()

    x_coords = np.arange(min_b[0], max_b[0], spacing)
    y_coords = np.arange(min_b[1], max_b[1], spacing)

    xx, yy = np.meshgrid(x_coords, y_coords)
    zz = np.full_like(xx, height)

    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)





# ==========================================
# 3. OPTIMIZATION OBJECTIVE
# ==========================================
def mlp_coverage_score(ap_position, mesh, grid_points, model, scaler_X, rssi_threshold=-60, scene=None):
    """
    Surrogate-based objective function evaluating PyTorch model predictions.

    Args:
        scene (o3d.t.geometry.RaycastingScene, optional): Pre-built raycasting
            scene for `mesh`. Pass this so the BVH is built once outside the
            optimizer instead of once per candidate AP position (differential_evolution
            calls this hundreds of times).
    """
    # 1. Feature extraction strictly from 3D geometry
    X_raw = IJS.extract_features(ap_position, grid_points, mesh, scene=scene)
    X_scaled = scaler_X.transform(X_raw)

    # 2. Neural network inference
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        rssi_pred = model(X_tensor).numpy()

    # 3. Compute coverage score fraction
    coverage = np.mean(rssi_pred >= rssi_threshold)
    return -coverage  # Negative because scipy minimizes


# ==========================================
# 4. VISUALIZATION
# ==========================================



# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- Step 1: Load 3D Model ---
    midpoint_z = float(mesh.get_axis_aligned_bounding_box().get_min_bound()[2]+((mesh.get_axis_aligned_bounding_box().get_max_bound()[2]-mesh.get_axis_aligned_bounding_box().get_min_bound()[2])/2))
    # --- Step 2: Auto-Generate Grid Points ---
    grid_points = generate_grid_from_mesh(mesh, spacing=1, height=midpoint_z)
    mesh_verts_2d = np.asarray(mesh.vertices)[:, :2]

    # Clearing out points which float in not dense areas.Remove/comment if you plan on stealing your neighbors wifi:)
    for idx in range(len(grid_points)):
        point = grid_points[idx]

        has_density = np.any(np.linalg.norm(mesh_verts_2d - point[:2], axis=1) <= 0.05)

        if not has_density:
            grid_points[idx] = np.nan


    grid_points = grid_points[~np.isnan(grid_points).any(axis=1)]
    # --- Step 3: Load Pre-trained PyTorch MLP Model ---
    model = RSSIEstimator(input_dim=2)
    checkpoint = torch.load(f"pt model/rssi_estimator{freq_band}G.pt", map_location=torch.device('cpu'), weights_only=False)
    # Handle dict checkpoint structure vs direct state dict
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # --- Step 4: Fit Feature Scaler Dynamically from Mesh Geometry ---
    bounds = mesh.get_axis_aligned_bounding_box()
    min_b, max_b = bounds.get_min_bound(), bounds.get_max_bound()

    # --- Step 4b: Build the raycasting BVH once, reuse for every evaluation ---
    # estimate_wall_count previously rebuilt this from scratch on every call
    # (~930 rebuilds over the optimization run). Building it once here and
    # threading it through mlp_coverage_score -> extract_features cuts that
    # to a single build.
    raycast_scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    raycast_scene.add_triangles(mesh_t)

    # Generate sample feature set from center AP position to fit realistic scale
    sample_ap = (min_b + max_b) / 2.0
    sample_features = IJS.extract_features(sample_ap, grid_points, mesh, scene=raycast_scene)

    if isinstance(checkpoint, dict) and 'scaler_X' in checkpoint:
        print("Loaded StandardScaler directly from checkpoint file.")
        scaler_X = checkpoint['scaler_X']

    # --- Step 5: Sanitize Bounds & Run Optimization ---
    x_min, x_max = min(min_b[0], max_b[0]), max(min_b[0], max_b[0])
    y_min, y_max = min(min_b[1], max_b[1]), max(min_b[1], max_b[1])
    z_min, z_max = min(min_b[2], max_b[2]), max(min_b[2], max_b[2])
    ap_search_bounds = [
        (x_min, x_max),
        (y_min, y_max),
        (z_min, z_max)
    ]
    print("Starting Differential Evolution optimization...")
    result = differential_evolution(
        mlp_coverage_score,
        bounds=ap_search_bounds,
        args=(mesh, grid_points, model, scaler_X, -60, raycast_scene),
        maxiter=30,
        popsize=10,
        seed=42,
        polish=False,  # Prevents L-BFGS-B bounds validation error
        disp=True
    )
    optimal_ap = result.x
    # --- Step 6: Visualize Result ---
    IJS.visualize_results(mesh, grid_points, optimal_ap, model, scaler_X, rssi_threshold=-50)
