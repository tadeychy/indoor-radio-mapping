import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import copy
import open3d as o3d

def manual_alignment_mouse(source, target, output_path="aligned_pcd.ply"):
    """
    Manually align source to target point cloud using Polyscope's
    interactive mouse-draggable transform gizmo.

    Controls:
        - Drag the gizmo ARROWS to translate
        - Drag the gizmo RINGS to rotate
        - Left drag   = rotate view
        - Right drag  = zoom
        - Middle drag = pan
        - Click 'Save Alignment' button when done, then close window

    Args:
        source (o3d.geometry.PointCloud): Source point cloud (will be moved).
        target (o3d.geometry.PointCloud): Target point cloud (stays fixed).
        output_path (str): Path to save the merged point cloud.

    Returns:
        merged (o3d.geometry.PointCloud): Merged aligned point cloud.
        transformation (np.ndarray): 4x4 transformation matrix.
    """
    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    src_colors = np.asarray(source.colors) if source.has_colors() \
        else np.tile([1.0, 0.3, 0.3], (len(src_pts), 1))
    tgt_colors = np.asarray(target.colors) if target.has_colors() \
        else np.tile([0.3, 0.3, 1.0], (len(tgt_pts), 1))

    result = {"saved": False}

    ps.init()
    ps.set_up_dir("z_up")

    # Register point clouds
    ps_source = ps.register_point_cloud("source (move me)", src_pts)
    ps_source.add_color_quantity("color", src_colors, enabled=True)
    ps_source.set_transform(np.eye(4))
    ps_source.set_transform_gizmo_enabled(True)   # ← mouse draggable gizmo

    ps_target = ps.register_point_cloud("target (fixed)", tgt_pts)
    ps_target.add_color_quantity("color", tgt_colors, enabled=True)

    def callback():
        psim.TextUnformatted("Drag the gizmo to align the red cloud to the blue cloud.")
        psim.Separator()
        if not result["saved"]:
            if psim.Button("Save Alignment"):
                result["transform"] = ps_source.get_transform()
                result["saved"]     = True
                print("Alignment saved — close the window to continue.")
        else:
            psim.TextUnformatted("Saved! Close the window to continue.")

    ps.set_user_callback(callback)
    ps.show()  # blocks until window is closed

    if result["saved"]:
        T = result["transform"]

        aligned = copy.deepcopy(source)
        aligned.transform(T)
        merged = aligned + target

        o3d.io.write_point_cloud(output_path, merged)
        print(f"Merged point cloud saved to: {output_path}")

        return merged, T
    else:
        print("Alignment discarded.")
        return None, None
first = input("First point cloud name: ")
second = input("Second point cloud name: ")
loaded_pcd1 = o3d.io.read_point_cloud(first)
loaded_pcd2 = o3d.io.read_point_cloud(second)

merged, T = manual_alignment_mouse(
    source      = loaded_pcd2,
    target      = loaded_pcd1,
    output_path = "MHAB.ply"
)