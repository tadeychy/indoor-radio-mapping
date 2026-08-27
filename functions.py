import pyvista as pv
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from scipy.optimize import curve_fit
import warnings
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np


def _predict(model, x_scaled):
    """
    Helper to predict using either Scikit-Learn or PyTorch models.
    """
    if isinstance(model, torch.nn.Module):
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tensor_input = torch.tensor(x_scaled, dtype=torch.float32).to(device)
            return model(tensor_input).cpu().numpy().squeeze()
    else:
        return model.predict(x_scaled)


# Mesh editing
def rotate(pcd, x, y, z, save=""):
    """
    Rotates a mesh file.

    Args:
        pcd (str): Filename.
        x (int): Rotation of x axis in degrees.
        y (int): Rotation of y axis in degrees.
        z (int): Rotation of z axis in degrees.
        save (bool, optional): Save to file. Defaults to False.
    """
    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.radians(x), np.radians(y), np.radians(z))))
        if save!="":
            o3d.io.write_point_cloud(save, pcd)
        return pcd

    elif isinstance(pcd, o3d.geometry.TriangleMesh):
        pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.radians(x), np.radians(y), np.radians(z))))
        if save!="":
            o3d.io.write_triangle_mesh(save, pcd)
        return pcd



def cut_bellow(mesh, x=None, y=None, z=None, save=None):
    """
    Crops a 3D triangle mesh by raising its lower boundary along specified axes.

    Args:
        mesh (str): Path to the input 3D triangle mesh file.
        x (float, optional): Minimum X-axis cutoff coordinate. Vertices with
            X values below this threshold are removed. If None, the original
            mesh minimum X is kept. Defaults to None.
        y (float, optional): Minimum Y-axis cutoff coordinate. Vertices with
            Y values below this threshold are removed. If None, the original
            mesh minimum Y is kept. Defaults to None.
        z (float, optional): Minimum Z-axis cutoff coordinate. Vertices with
            Z values below this threshold are removed. If None, the original
            mesh minimum Z is kept. Defaults to None.
        save (str, optional): File path where the cropped mesh should be saved.
            If None, the mesh is not saved to disk. Defaults to None.

    Returns:
        open3d.geometry.TriangleMesh: The cropped triangle mesh.
    """
    mesh = o3d.io.read_triangle_mesh(mesh)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    if x is not None:
        min_bound[0] = x
    if y is not None:
        min_bound[1] = y
    if z is not None:
        min_bound[2] = z
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cut_mesh = mesh.crop(bbox)
    if save is not None:
        o3d.io.write_triangle_mesh(save, cut_mesh)
    return cut_mesh


def cut_above(mesh, x=None, y=None, z=None, save=None):
    """
    Crops a 3D triangle mesh by lowering its upper boundary along specified axes.

    Args:
        mesh (str): Path to the input 3D triangle mesh file.
        x (float, optional): Maximum X-axis cutoff coordinate. Vertices with
            X values above this threshold are removed. If None, the original
            mesh maximum X is kept. Defaults to None.
        y (float, optional): Maximum Y-axis cutoff coordinate. Vertices with
            Y values above this threshold are removed. If None, the original
            mesh maximum Y is kept. Defaults to None.
        z (float, optional): Maximum Z-axis cutoff coordinate. Vertices with
            Z values above this threshold are removed. If None, the original
            mesh maximum Z is kept. Defaults to None.
        save (str, optional): File path where the cropped mesh should be saved.
            If None, the mesh is not saved to disk. Defaults to None.

    Returns:
        open3d.geometry.TriangleMesh: The cropped triangle mesh.
    """
    mesh = o3d.io.read_triangle_mesh(mesh)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    if x is not None:
        max_bound[0] = x
    if y is not None:
        max_bound[1] = y
    if z is not None:
        max_bound[2] = z
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cut_mesh = mesh.crop(bbox)
    if save is not None:
        o3d.io.write_triangle_mesh(save, cut_mesh)
    return cut_mesh


def cuting(pcd_or_mesh_input, offset, save):
    """Opens an interactive Open3D window to select 2 points and crops

        the geometry outside the bounding box defined by those points plus an offset.

        Args:
            pcd_or_mesh_input (str | o3d.geometry.PointCloud | o3d.geometry.TriangleMesh):
                File path to a point cloud/mesh, or an Open3D geometry object.
            offset (float):
                Margin/padding added around the bounding coordinates (min/max bounds)
                defined by the two picked points.
            save (str):
                File path or identifier where the cropped geometry will be saved.

        Raises:
            ValueError: If fewer than 2 points are selected in the visualizer window.

        Returns:
            None: Saves the cropped point cloud/mesh via external `IJS` utility calls.
        """
    # 1. Load input data
    if isinstance(pcd_or_mesh_input, str):
        geometry = o3d.io.read_point_cloud(pcd_or_mesh_input)
        if len(geometry.points) == 0:
            geometry = o3d.io.read_triangle_mesh(pcd_or_mesh_input)
    else:
        geometry = pcd_or_mesh_input

    # Convert mesh vertices to point cloud if needed
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = geometry.vertices
    else:
        pcd = geometry

    # 2. Print Instructions
    print("\n" + "=" * 50)
    print("Hold [SHIFT] + Left Click to pick 2 points.")
    print(" Close the window when finished.")
    print("=" * 50 + "\n")

    # 3. Open Interactive Visualizer
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Select 2 Points ([SHIFT] + Left Click)",
        width=1280,
        height=720,
    )
    vis.add_geometry(pcd)
    vis.run()  # Pauses until window is closed
    vis.destroy_window()

    # 4. Extract picked points
    picked_indices = vis.get_picked_points()

    if len(picked_indices) < 2:
        raise ValueError(
            f"You selected {len(picked_indices)} points. Please select at least 2 points."
        )

    # 5. Extract coordinates as variables
    points_array = np.asarray(pcd.points)
    p1 = points_array[picked_indices[0]]  # First point variable [x, y, z]
    p2 = points_array[picked_indices[1]]  # Second point variable [x, y, z]
    cut_above(pcd_or_mesh_input, x=max(p1[0], p2[0]) + offset, y=max(p1[1], p2[1]) + offset,
              z=max(p1[2], p2[2]) + offset, save=save)
    cut_bellow(save, x=min(p1[0], p2[0]) - offset, y=min(p1[1], p2[1]) - offset, z=min(p1[2], p2[2]) - offset,
               save=save)


def point_to_triangle(pcd, depth=14, rem_den=5, radius=500, k=30, max_nn=30, save=None, ):
    """
    Transforms a point cloud file into a 3D triangle mesh using Poisson surface reconstruction.

    Args:
        pcd (str): Path to the input point cloud file.
        depth (int, optional): Tree depth used for Poisson surface reconstruction.
            Higher values capture finer detail but increase memory usage. Defaults to 14.
        rem_den (int or float, optional): Percentile threshold for low-density vertex
            removal to clean up mesh artifacts. Set to 0 to disable. Defaults to 5.
        radius (float, optional): Search radius for hybrid KDTree normal estimation. Defaults to 500.
        k (int, optional): Number of nearest neighbors used to construct the Riemannian
            graph for consistent normal orientation. Defaults to 30.
        max_nn (int, optional): Maximum nearest neighbors considered within the search radius
            during normal estimation. Defaults to 30.
        save (str, optional): File path where the resulting triangle mesh should be saved.
            If None, the mesh is not saved to disk. Defaults to None.

    Returns:
        open3d.geometry.TriangleMesh: The generated triangle mesh.
    """
    pcd = o3d.io.read_point_cloud(pcd)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(k=k)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    if rem_den != 0:
        densities = np.asarray(densities)
        density_threshold = np.percentile(densities, rem_den)  # Remove bottom 5%
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
    if save != None:
        o3d.io.write_triangle_mesh(save, mesh)
    return mesh


# AI
def train_model(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=20):
    """
    Train the MLP with early stopping based on validation loss.

        model: RSSIEstimator
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        epochs (int): Maximum number of epochs
        lr (float): Learning rate
       patience (int): Early stopping patience

    Returns:
        model: Trained model
        history (dict): Training and validation loss history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1:4d} | Train RMSE: {np.sqrt(train_loss):.3f} dBm"
                  f" | Val RMSE: {np.sqrt(val_loss):.3f} dBm")

    # Restore best weights
    model.load_state_dict(best_weights)
    return model, history


def evaluate_model(model, loader, scaler_X, scaler_y, split_name="Test"):
    """
    Evaluate the model and print MAE, RMSE, and R2 metrics.
    """
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            pred = model(X_batch)
            preds.append(pred.numpy())
            targets.append(y_batch.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Inverse transform if targets were scaled
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"\n{split_name} Results:")
    print(f"  MAE:  {mae:.3f} dBm")
    print(f"  RMSE: {rmse:.3f} dBm")
    print(f"  R﷿﷿:   {r2:.4f}")

    return preds, targets, {"mae": mae, "rmse": rmse, "r2": r2}


def to_tensors(X, y):
    return TensorDataset(torch.tensor(X), torch.tensor(y))


def log_distance_model(distances, n, A, eps=1e-6):
    """Standard log-distance path loss model."""
    d_meters = np.maximum(distances / 1000, eps)  # avoid log10(0)
    return A - 10 * n * np.log10(d_meters)


def itu_model(distances, walls, n, A, L_wall, eps=1e-6):
    """ITU indoor path loss model with wall attenuation."""
    d_meters = np.maximum(distances / 1000, eps)  # avoid log10(0)
    return A - 20 * np.log10(d_meters) - walls * L_wall


def fit_log_distance(X_train, y_train):
    """Fit log-distance model by least squares."""
    distances = X_train[:, 0]

    # Better initial guess: n=2 (free space), A=estimated from closest point
    p0 = [2.0, np.max(y_train)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov = curve_fit(
                lambda d, n, A: log_distance_model(d, n, A),
                distances, y_train,
                p0=p0,
                bounds=([0.5, -120], [6.0, 0]),  # n in [0.5, 6], A in [-120, 0] dBm
                maxfev=10000
            )
            # Check if fit converged properly
            if np.any(np.isinf(np.diag(pcov))):
                print("  Log-distance: covariance estimation failed, using p0 as fallback")
                return p0
        except RuntimeError:
            print("  Log-distance: curve_fit failed, using initial guess as fallback")
            return p0

    print(f"  Log-distance fitted: n={popt[0]:.2f}, A={popt[1]:.2f}")
    return popt


def fit_itu(X_train, y_train):
    """Fit ITU model by least squares."""
    distances, walls = X_train[:, 0], X_train[:, 1]

    p0 = [2.0, np.max(y_train), 3.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, pcov = curve_fit(
                lambda X, n, A, L: itu_model(X[0], X[1], n, A, L),
                (distances, walls), y_train,
                p0=p0,
                bounds=([0.5, -120, 0.0], [6.0, 0, 20.0]),  # L_wall in [0, 20] dBm
                maxfev=10000
            )
            if np.any(np.isinf(np.diag(pcov))):
                print("  ITU: covariance estimation failed, using p0 as fallback")
                return p0
        except RuntimeError:
            print("  ITU: curve_fit failed, using initial guess as fallback")
            return p0

    print(f"  ITU fitted: n={popt[0]:.2f}, A={popt[1]:.2f}, L_wall={popt[2]:.2f}")
    return popt


def coverage_score(ap_position, mesh, grid_points, model, scaler_X, rssi_threshold=-90, scene=None):
    """
    Surrogate-based coverage score for a given AP position.
    Returns fraction of grid points above RSSI threshold (higher = better).

    Args:
        scene (o3d.t.geometry.RaycastingScene, optional): Pre-built raycasting
            scene for `mesh` to avoid rebuilding the BVH on every call inside
            an optimization loop.
    """
    ap_position = np.array(ap_position)

    # Compute features for all grid points
    distances = np.linalg.norm(grid_points - ap_position, axis=1)
    wall_counts = estimate_wall_count(mesh, ap_position, grid_points, scene=scene)
    X = np.column_stack([distances, wall_counts]).astype(np.float32)
    X_scaled = scaler_X.fit_transform(X)

    # Predict RSSI using surrogate
    # rssi_pred = model.predict(X_scaled)  # use SVR or GP as surrogate
    rssi_pred = _predict(model, X_scaled)
    coverage = np.mean(rssi_pred >= rssi_threshold)
    return -coverage  # negative because scipy minimizes


def evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test,
                 mlp_model=None, test_loader=None):
    results = {}

    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.5),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42),
        "Gaussian Process": GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), normalize_y=True
        ),
    }

    for name, est in estimators.items():
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds),
        }
        print(f"{name:25s} | MAE: {results[name]['mae']:.3f} | "
              f"RMSE: {results[name]['rmse']:.3f} | R﷿﷿: {results[name]['r2']:.4f}")

    # Physics-based models
    n, A = fit_log_distance(X_train, y_train)
    preds_ld = log_distance_model(X_test[:, 0], n, A)
    results["Log-Distance"] = {
        "mae": mean_absolute_error(y_test, preds_ld),
        "rmse": np.sqrt(mean_squared_error(y_test, preds_ld)),
        "r2": r2_score(y_test, preds_ld),
    }
    print(f"{'Log-Distance':25s} | MAE: {results['Log-Distance']['mae']:.3f} | "
          f"RMSE: {results['Log-Distance']['rmse']:.3f} | "
          f"R﷿﷿: {results['Log-Distance']['r2']:.4f} | n={n:.2f}, A={A:.2f}")

    n, A, L = fit_itu(X_train, y_train)
    preds_itu = itu_model(X_test[:, 0], X_test[:, 1], n, A, L)
    results["ITU Indoor"] = {
        "mae": mean_absolute_error(y_test, preds_itu),
        "rmse": np.sqrt(mean_squared_error(y_test, preds_itu)),
        "r2": r2_score(y_test, preds_itu),
    }

    print(f"{'ITU Indoor':25s} | MAE: {results['ITU Indoor']['mae']:.3f} | "
          f"RMSE: {results['ITU Indoor']['rmse']:.3f} | "
          f"R﷿﷿: {results['ITU Indoor']['r2']:.4f} | "
          f"n={n:.2f}, A={A:.2f}, L_wall={L:.2f}")

    # MLP
    if mlp_model is not None and test_loader is not None:
        mlp_model.eval()
        preds_mlp, targets_mlp = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds_mlp.append(mlp_model(X_batch).numpy())
                targets_mlp.append(y_batch.numpy())
        preds_mlp = np.concatenate(preds_mlp)
        targets_mlp = np.concatenate(targets_mlp)
        results["MLP"] = {
            "mae": mean_absolute_error(targets_mlp, preds_mlp),
            "rmse": np.sqrt(mean_squared_error(targets_mlp, preds_mlp)),
            "r2": r2_score(targets_mlp, preds_mlp),
        }
        print(f"{'MLP':25s} | MAE: {results['MLP']['mae']:.3f} | "
              f"RMSE: {results['MLP']['rmse']:.3f} | R﷿﷿: {results['MLP']['r2']:.4f}")

    return results, estimators


# mesh measuring
def estimate_wall_count(mesh, ap_point, measurement_points, max_walls=10, scene=None):
    """
    Estimate the number of surfaces (walls) between the AP and each
    measurement point using ray casting.

    Args:
        mesh (o3d.geometry.TriangleMesh): Reconstructed surface mesh. Only used
            to build a scene if `scene` is not provided.
        ap_point (np.ndarray): AP position as [x, y, z].
        measurement_points (np.ndarray): Measurement positions as (N, 3) array.
        max_walls (int): Maximum number of walls to count per ray (default 10).
        scene (o3d.t.geometry.RaycastingScene, optional): Pre-built raycasting
            scene (BVH already constructed) for `mesh`. Passing this avoids
            rebuilding the BVH on every call, which is the dominant cost when
            this function runs inside an optimization loop. If None, a scene
            is built from `mesh` as before (unchanged fallback behavior).

    Returns:
        wall_counts (np.ndarray): Number of walls crossed for each measurement point.
    """
    # Reuse a pre-built scene if given; otherwise build one from mesh (old behavior)
    if scene is None:
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

    ap = np.array(ap_point, dtype=np.float32)
    wall_counts = np.zeros(len(measurement_points), dtype=int)

    for i, meas_pt in enumerate(measurement_points):
        meas_pt = np.array(meas_pt, dtype=np.float32)

        # Direction vector from AP to measurement point
        direction = meas_pt - ap
        total_distance = np.linalg.norm(direction)
        direction_normalized = direction / total_distance

        # Cast ray and collect all intersections along it
        n_intersections = 0
        current_origin = ap.copy()
        epsilon = 0.01  # small offset in mm to avoid re-hitting the same surface

        for _ in range(max_walls):
            ray = o3d.core.Tensor(
                [[*current_origin, *direction_normalized]],
                dtype=o3d.core.Dtype.Float32
            )
            result = scene.cast_rays(ray)
            hit_distance = result["t_hit"].numpy()[0]

            # No more intersections or hit is beyond the measurement point
            if np.isinf(hit_distance) or hit_distance > total_distance:
                break

            n_intersections += 1

            # Move origin just past the hit point to find the next intersection
            current_origin = current_origin + direction_normalized * (hit_distance + epsilon)

            # Update remaining distance
            total_distance -= (hit_distance + epsilon)

        wall_counts[i] = n_intersections

    return wall_counts


def prepare_dataset(data_groups, feature_maps):
    """
    Prepare features (distance, wall count) and targets (RSSI) for all setups.

    Args:
        data_groups (list): List of RSSI measurement matrices.
        feature_maps (list): Output of generate_ap_feature_maps.

    Returns:
        X (np.ndarray): Features array (N, 2) ﷿﷿﷿ [distance, wall_count]
        y (np.ndarray): RSSI targets array (N,)
        setup_indices (np.ndarray): Setup index for each sample
    """
    X_all, y_all, setup_idx_all = [], [], []

    for i, (matrix, fm) in enumerate(zip(data_groups, feature_maps)):
        y_indices, x_indices = fm["grid_indices"]

        for j, (r, c) in enumerate(zip(y_indices, x_indices)):
            rssi = matrix[r, c]
            if np.isnan(rssi):
                continue

            distance = fm["distances"][j]
            walls = fm["wall_counts"][j]

            X_all.append([distance, walls])
            y_all.append(rssi)
            setup_idx_all.append(i)

    return (np.array(X_all, dtype=np.float32),
            np.array(y_all, dtype=np.float32),
            np.array(setup_idx_all))


def generate_ap_feature_maps(data_groups, matrix_inds, ap_locations, mesh,
                             grid_spacing=1.0, x_offset=9200, y_offset=25000,
                             z_offset=-2300, ap_z=1200):
    """
    Generate a feature map for each AP configuration, containing:
        - Distance from AP to each measurement point
        - Estimated number of walls between AP and each measurement point

    Args:
        data_groups (list): List of measurement matrices.
        matrix_inds (list): List of index matrices.
        ap_locations (list): List of AP location identifiers.
        mesh (o3d.geometry.TriangleMesh): Reconstructed surface mesh.
        grid_spacing (float): Grid spacing in meters (default 1.0).
        x_offset (float): X offset in mm (default 9200).
        y_offset (float): Y offset in mm (default 25000).
        z_offset (float): Z offset in mm (default -2300).
        ap_z (float): AP height in mm (default 1200).

    Returns:
        feature_maps (list of dict): One dict per AP config with keys:
            - "distance_map": 2D array of distances (mm)
            - "wall_map": 2D array of wall counts
            - "ap_point": AP position as [x, y, z]
            - "points_meas": measurement points as (N, 3) array
            - "grid_indices": (y_indices, x_indices) of valid measurement positions
    """
    # Build raycasting scene once (reused for all AP configs)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    feature_maps = []

    for setup_idx in range(len(data_groups)):
        print(f"\nProcessing AP configuration {setup_idx + 1}/{len(data_groups)}...")

        matrix = data_groups[setup_idx]
        matrix_ind = matrix_inds
        ap_loc = ap_locations[setup_idx]

        # Measurement points
        y_indices, x_indices = np.nonzero(matrix)
        points_meas = np.column_stack((
            -y_indices * grid_spacing + x_offset,
            -x_indices * grid_spacing + y_offset,
            np.full(len(x_indices), z_offset)
        ))

        # AP point
        ap_r, ap_c = np.where(np.array(matrix_ind) == np.array(ap_loc))
        ap_point = np.array([
            -ap_r[0] * grid_spacing + x_offset,
            -ap_c[0] * grid_spacing + y_offset,
            ap_z + z_offset
        ])
        # Distance map
        print(f"  Computing distances...")
        distances = np.linalg.norm(points_meas - ap_point, axis=1) * 2

        # Wall count map
        print(f"  Casting rays for wall estimation...")
        wall_counts = estimate_wall_count(mesh, ap_point, points_meas, scene=scene)

        # Fill 2D grids
        distance_map = np.full(matrix.shape, np.nan)
        wall_map = np.full(matrix.shape, np.nan)

        for idx, (r, c) in enumerate(zip(y_indices, x_indices)):
            distance_map[r, c] = distances[idx]
            wall_map[r, c] = wall_counts[idx]

        feature_maps.append({
            "distance_map": distance_map,
            "wall_map": wall_map,
            "ap_point": ap_point,
            "points_meas": points_meas,
            "grid_indices": (y_indices, x_indices),
            "distances": distances,
            "wall_counts": wall_counts,
        })

        print(f"  Distance range: [{distances.min():.0f}, {distances.max():.0f}] mm")
        print(f"  Wall count range: [{wall_counts.min()}, {wall_counts.max()}]")

    return feature_maps


# visualisation
def visualize_optimal_ap(mesh, grid_points, optimal_ap, svr_model, scaler_X,
                         rssi_threshold=-40, z_offset=-2300):
    """
    Visualize the optimal AP position on the reconstructed point cloud,
    with measurement points colored by predicted RSSI.

    Args:
        mesh: o3d.geometry.TriangleMesh ﷿﷿﷿ reconstructed surface
        grid_points: (N, 3) array of measurement positions
        optimal_ap: (3,) array ﷿﷿﷿ optimal AP position from optimizer
        svr_model: fitted surrogate model
        scaler_X: fitted StandardScaler
        rssi_threshold: RSSI threshold used in optimization
        z_offset: z offset used in point cloud alignment
"""
    # Predict RSSI for all grid points from optimal AP position
    distances = np.linalg.norm(grid_points - optimal_ap, axis=1)

    wall_counts = estimate_wall_count(mesh, optimal_ap, grid_points)
    X_opt = scaler_X.transform(np.column_stack([distances, wall_counts]))
    rssi_pred = svr_model.predict(X_opt)
    above_threshold = rssi_pred >= rssi_threshold
    coverage = np.mean(above_threshold)
    print(f"Predicted coverage at optimal AP: {coverage:.1%}")
    print(f"RSSI range: [{rssi_pred.min():.1f}, {rssi_pred.max():.1f}] dBm")

    # ﷿﷿﷿﷿﷿﷿ Convert mesh to PyVista ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles])
    pv_mesh = pv.PolyData(vertices, faces)

    # Height-based coloring for mesh
    z_vals = vertices[:, 2]
    z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
    pv_mesh["height"] = z_norm

    # ﷿﷿﷿﷿﷿﷿ Measurement points colored by predicted RSSI ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿
    meas_pv = pv.PolyData(grid_points)
    # for i in range(len(grid_points)):
    #    grid_points[i][2] = 0.4
    meas_pv["RSSI (dBm)"] = rssi_pred
    # optimal_ap[2] = 0.4
    # ﷿﷿﷿﷿﷿﷿ Optimal AP marker ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿
    ap_sphere = pv.Sphere(radius=0.2, center=optimal_ap)

    # Ray lines from AP to each measurement point
    lines = []
    for pt in grid_points:
        line = pv.Line(optimal_ap, pt)
        lines.append(line)
    ray_lines = pv.MultiBlock(lines).combine()

    # ﷿﷿﷿﷿﷿﷿ Plot ﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿﷿
    plotter = pv.Plotter()
    plotter.set_background("white")

    # Mesh ﷿﷿﷿ semi-transparent so you can see inside
    plotter.add_mesh(pv_mesh, scalars="height", cmap="gray",
                     opacity=0.3, show_scalar_bar=False, label="Room mesh")
    # Measurement points colored by RSSI
    plotter.add_mesh(meas_pv, scalars="RSSI (dBm)", cmap="RdYlGn",
                     point_size=20, render_points_as_spheres=True,
                     clim=[rssi_pred.min(), rssi_pred.max()],
                     scalar_bar_args={"title": "Predicted RSSI (dBm)"},
                     label="Measurement points")

    # Optimal AP ﷿﷿﷿ large green sphere
    plotter.add_mesh(ap_sphere, color="lime", label="Optimal AP")
    plotter.add_point_labels(
        [optimal_ap], [f"Optimal AP\n({optimal_ap[0]:.0f}, {optimal_ap[1]:.0f}, {optimal_ap[2]:.0f}) mm"],
        font_size=12, text_color="black", shape_opacity=0.5
    )

    # Ray lines ﷿﷿﷿ colored by above/below threshold
    plotter.add_mesh(ray_lines, color="gray", opacity=0.1, label="LOS rays")

    plotter.add_text(
        f"Optimal AP Position | Coverage: {coverage:.1%} above {rssi_threshold} dBm",
        position="upper_edge", font_size=11
    )
    plotter.add_legend()
    plotter.show()
    return pv_mesh


def extract_features(ap_position, grid_points, mesh, scene=None):
    """
    Computes [distance, wall_count] feature array X without CSV file context.

    Args:
        scene (o3d.t.geometry.RaycastingScene, optional): Pre-built raycasting
            scene for `mesh`. Pass this when calling extract_features inside
            an optimization loop (e.g. differential_evolution) so the BVH is
            built once instead of once per candidate AP position.
    """
    distances = np.linalg.norm(grid_points - ap_position, axis=1) * 2  # Kiri seems to halve sizes in 3d scans
    wall_counts = estimate_wall_count(mesh, ap_position, grid_points, scene=scene)
    return np.column_stack([distances, wall_counts]).astype(np.float32)


def visualize_results(mesh, grid_points, optimal_ap, model, scaler_X, rssi_threshold=-60):
    X_raw = extract_features(optimal_ap, grid_points, mesh)
    X_scaled = scaler_X.transform(X_raw)

    model.eval()
    with torch.no_grad():
        rssi_pred = model(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    coverage = np.mean(rssi_pred >= rssi_threshold)

    # Build PyVista mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    faces = np.hstack([np.full((len(triangles), 1), 3), triangles])
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh["height"] = (vertices[:, 2] - vertices[:, 2].min()) / (vertices[:, 2].max() - vertices[:, 2].min())

    meas_pv = pv.PolyData(grid_points)
    meas_pv["RSSI (dBm)"] = rssi_pred
    ap_sphere = pv.Sphere(radius=0.15, center=optimal_ap)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_mesh(pv_mesh, scalars="height", cmap="gray", opacity=0.3, show_scalar_bar=False)
    plotter.add_mesh(meas_pv, scalars="RSSI (dBm)", cmap="RdYlGn", point_size=24, render_points_as_spheres=True)
    plotter.add_mesh(ap_sphere, color="purple", label="Optimal AP")
    plotter.add_text(f"Optimal AP | Coverage: {coverage:.1%} above {rssi_threshold} dBm", position="upper_edge",
                     font_size=11)
    plotter.add_legend()
    plotter.show()


def plot_comparison(results):
    names = list(results.keys())
    maes = [results[n]["mae"] for n in names]
    rmses = [results[n]["rmse"] for n in names]
    r2s = [results[n]["r2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, values, label, color in zip(
            axes,
            [maes, rmses, r2s],
            ["MAE (dBm)", "RMSE (dBm)", "R﷿﷿"],
            ["steelblue", "coral", "mediumseagreen"]
    ):
        bars = ax.barh(names, values, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlabel(label)
        ax.set_title(label)
        ax.invert_yaxis()

    plt.suptitle("Model Comparison on Test Set (Setups 19-20)", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_results(history, preds_val, targets_val, preds_test, targets_test):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Loss curves
    axes[0].plot(history["train"], label="Train")
    axes[0].plot(history["val"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training History")
    axes[0].legend()

    # Validation: predicted vs actual
    axes[1].scatter(targets_val, preds_val, alpha=0.5, s=20)
    lims = [min(targets_val.min(), preds_val.min()),
            max(targets_val.max(), preds_val.max())]
    axes[1].plot(lims, lims, "r--", label="Ideal")
    axes[1].set_xlabel("Actual RSSI (dBm)")
    axes[1].set_ylabel("Predicted RSSI (dBm)")
    axes[1].set_title("Validation: Predicted vs Actual")
    axes[1].legend()

    # Test: predicted vs actual
    axes[2].scatter(targets_test, preds_test, alpha=0.5, s=20, color="orange")
    lims = [min(targets_test.min(), preds_test.min()),
            max(targets_test.max(), preds_test.max())]
    axes[2].plot(lims, lims, "r--", label="Ideal")
    axes[2].set_xlabel("Actual RSSI (dBm)")
    axes[2].set_ylabel("Predicted RSSI (dBm)")
    axes[2].set_title("Test: Predicted vs Actual")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_feature_maps(feature_maps, matrix_inds, ap_locations, grid_spacing=1.0):
    """
    Plot distance and wall count maps side by side for each AP configuration.

    Args:
        feature_maps (list of dict): Output of generate_ap_feature_maps.
        matrix_inds (list): List of index matrices (for AP position marking).
        ap_locations (list): List of AP location identifiers.
        grid_spacing (float): Grid spacing in meters.
    """
    n_configs = len(feature_maps)
    fig, axes = plt.subplots(n_configs, 2, figsize=(12, 5 * n_configs))

    # Handle single config case
    if n_configs == 1:
        axes = axes[np.newaxis, :]

    for i, fm in enumerate(feature_maps):
        ap_r, ap_c = np.where(
            matrix_inds == np.array(ap_locations[i]))  # same story as on line 85 just with [i] on matrix_inds

        # Distance map
        ax = axes[i, 0]
        im = ax.imshow(fm["distance_map"], cmap="viridis", interpolation="nearest")
        ax.scatter(ap_c[0], ap_r[0], c="red", s=200, marker="*",
                   label="AP", zorder=5)
        plt.colorbar(im, ax=ax, label="Distance (mm)")
        ax.set_title(f"AP Config {i + 1} ﷿﷿﷿ Distance to AP")
        ax.legend()

        # Wall count map
        ax = axes[i, 1]
        im = ax.imshow(fm["wall_map"], cmap="plasma", interpolation="nearest",
                       vmin=0, vmax=np.nanmax(fm["wall_map"]))
        ax.scatter(ap_c[0], ap_r[0], c="lime", s=200, marker="*",
                   label="AP", zorder=5)
        plt.colorbar(im, ax=ax, label="Number of walls")
        ax.set_title(f"AP Config {i + 1} ﷿﷿﷿ Wall Count")
        ax.legend()

    plt.tight_layout()
    plt.show()


def rssi_contour_plot(matrix, matrix_ind, ap_location, setup_num, save_figure=True):
    # Get the coordinates of nonzero values

    matrix = np.flipud(matrix)
    matrix_ind = np.flipud(matrix_ind)

    y_indices, x_indices = np.nonzero(matrix)  # (row, column)
    z_values = matrix[y_indices, x_indices]  # Extract nonzero values

    # Create a contour plot
    plt.tricontourf(x_indices, y_indices, z_values, levels=32, cmap="viridis")
    plt.colorbar(label="dBm")

    # Scatter plot for known data points
    plt.scatter(x_indices, y_indices, s=25, marker="o", edgecolors="k", color="red", label="Measurement Points")

    # Add enumeration at each data point
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x, y] != 0:
                plt.text(y + 0.2, x + 0.2, str(int(matrix_ind[x, y])), fontsize=9, color='cyan')

    # Add the AP location (The Gold Star from the estimated plot)
    ap_x, ap_y = np.where(matrix_ind == ap_location)
    # ap_r_flipped = (matrix.shape[0] - 1) - ap_r_orig
    plt.scatter([ap_y], [ap_x], s=250, marker="*", color="gold",
                edgecolors="black", linewidths=1.5, label=f"AP (setup={setup_num})")

    # Set axis limits to show more space
    padding = 2  # Adjust this value to control extra space
    plt.xlim(min(x_indices) - padding, max(x_indices) + padding)
    plt.ylim(min(y_indices) - padding, max(y_indices) + padding)

    # Labels and title
    plt.title("RSSI Values, setup {}, AP location {}".format(setup_num, ap_location))
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.legend()

    if save_figure:
        plt.savefig("rssi_{}.png".format(setup_num), dpi=300, bbox_inches="tight", transparent=True)

    # Show plot
    plt.show()
