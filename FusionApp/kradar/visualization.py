"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# 3D Visualization utilities for radar point cloud and detections
--------------------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path

# Try to import Open3D, but don't fail if it's not available
try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

# Import matplotlib as fallback
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .util_geometry import Object3D


def visualize_detections(
    pc_array,
    detections,
    save_path=None,
    interactive=False,
    conf_threshold=0.3,
    view_angle="perspective",
):
    """
    Visualize radar point cloud with 3D bounding box detections.

    Args:
        pc_array: Nx4 numpy array (x, y, z, power/intensity)
        detections: Dict with keys:
                    - 'boxes': Nx7 array [x, y, z, dx, dy, dz, heading]
                    - 'scores': N array of confidence scores
                    - 'labels': N array of class indices
                    - 'class_names': List of class names
        save_path: Path to save visualization image (without extension)
                   Multiple views will be saved as {save_path}_view.png
        interactive: If True, show interactive 3D viewer (Open3D only)
        conf_threshold: Minimum confidence threshold for displaying boxes
        view_angle: 'top', 'side', 'perspective', or 'all' (saves all views)

    Returns:
        None (saves images and/or shows interactive viewer)
    """
    if pc_array.shape[0] == 0:
        print("Warning: Empty point cloud, skipping visualization")
        return

    # Filter detections by confidence threshold
    # Convert to numpy arrays if needed (inference returns lists)
    boxes = np.array(detections.get("boxes", []))
    scores = np.array(detections.get("scores", []))
    labels = np.array(detections.get("labels", []))
    class_names_list = detections.get("class_names", [])

    if len(boxes) > 0 and len(scores) > 0:
        mask = scores >= conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Get class names for each detection
        filtered_class_names = [class_names_list[int(label)] for label in labels]

        print(f"Visualizing {len(boxes)} detections (threshold: {conf_threshold:.2f})")
        for i, (cls_name, score) in enumerate(zip(filtered_class_names, scores)):
            print(f"  [{i+1}] {cls_name}: {score:.3f}")
    else:
        filtered_class_names = []
        print("No detections to visualize")

    # Try Open3D first only if interactive mode is requested
    # Otherwise use matplotlib (more reliable in headless/WSL environments)
    if OPEN3D_AVAILABLE and interactive:
        try:
            _visualize_with_open3d(
                pc_array,
                boxes,
                filtered_class_names,
                save_path,
                view_angle,
                interactive,
            )
            return
        except Exception as e:
            print(f"Open3D visualization failed ({e}), falling back to matplotlib")

    # Use matplotlib for visualization (default for non-interactive)
    _visualize_with_matplotlib(
        pc_array, boxes, filtered_class_names, scores, save_path, view_angle
    )


def _visualize_with_matplotlib(
    pc_array, boxes, class_names, scores, save_path, view_angle
):
    """
    Visualize using matplotlib (works in headless environments).
    """
    print("Using matplotlib for visualization...")

    # Default colors for different classes
    color_map = {
        "Sedan": "#17D0F9",  # Cyan
        "Bus or Truck": "#0033FF",  # Blue
        "Motorcycle": "#FF0000",  # Red
        "Bicycle": "#FFFF00",  # Yellow
        "Pedestrian": "#FF0000",  # Red
        "Pedestrian Group": "#FF0066",  # Pink
    }

    # Define views to render
    views_config = {}
    if view_angle in ["top", "all"]:
        views_config["top"] = {"elev": 90, "azim": 0}
    if view_angle in ["side", "all"]:
        views_config["side"] = {"elev": 0, "azim": 90}
    if view_angle in ["perspective", "all"] or view_angle not in ["top", "side"]:
        views_config["perspective"] = {"elev": 30, "azim": 45}

    print(f"Rendering {len(views_config)} view(s): {list(views_config.keys())}")

    for view_name, view_params in views_config.items():
        print(f"  Rendering {view_name} view...")
        try:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection="3d")

            # Plot point cloud (subsample for performance)
            stride = max(1, len(pc_array) // 5000)  # Max 5000 points
            pc_sub = pc_array[::stride]
            print(f"    Plotting {len(pc_sub)} points (stride={stride})")
            ax.scatter(
                pc_sub[:, 0],
                pc_sub[:, 1],
                pc_sub[:, 2],
                c="gray",
                s=0.5,
                alpha=0.3,
                label="Point Cloud",
            )

            # Plot bounding boxes
            print(f"    Plotting {len(boxes)} bounding boxes")
            for i, (box, cls_name) in enumerate(zip(boxes, class_names)):
                color = color_map.get(cls_name, "#FF0000")
                score = scores[i] if i < len(scores) else 0.0

                # Create Object3D to get corners
                x, y, z, dx, dy, dz, heading = box
                obj = Object3D(x, y, z, dx, dy, dz, heading)
                corners = obj.corners

                # Draw the 12 edges of the bounding box
                edges = [
                    [0, 1],
                    [2, 3],
                    [4, 5],
                    [6, 7],  # Top and bottom edges
                    [0, 4],
                    [1, 5],
                    [2, 6],
                    [3, 7],  # Vertical edges
                    [0, 2],
                    [1, 3],
                    [4, 6],
                    [5, 7],  # Side edges
                ]

                for edge in edges:
                    p1, p2 = corners[edge[0]], corners[edge[1]]
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color=color,
                        linewidth=2,
                    )

                # Add label at center
                ax.text(
                    x,
                    y,
                    z + dz / 2 + 1,
                    f"{cls_name}\n{score:.2f}",
                    color=color,
                    fontsize=8,
                    ha="center",
                )

            # Set labels and limits
            ax.set_xlabel("X (m) - Forward", fontsize=10)
            ax.set_ylabel("Y (m) - Left", fontsize=10)
            ax.set_zlabel("Z (m) - Up", fontsize=10)
            ax.set_title(
                f"Radar Detections - {view_name.capitalize()} View", fontsize=14
            )

            # Set view angle
            ax.view_init(elev=view_params["elev"], azim=view_params["azim"])

            # Set reasonable axis limits
            x_data = pc_array[:, 0]
            y_data = pc_array[:, 1]
            z_data = pc_array[:, 2]

            ax.set_xlim([max(0, x_data.min() - 5), min(100, x_data.max() + 5)])
            ax.set_ylim([y_data.min() - 5, y_data.max() + 5])
            ax.set_zlim([z_data.min() - 2, z_data.max() + 2])

            # Equal aspect ratio
            ax.set_box_aspect([1, 0.5, 0.3])

            # Grid
            ax.grid(True, alpha=0.3)

            # Save
            if save_path:
                output_path = f"{save_path}_{view_name}.png"
                print(f"    Saving to: {output_path}")
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                print(f"    Saved {view_name} view: {output_path}")

            plt.close(fig)
            print(f"  Completed {view_name} view")
        except Exception as e:
            print(f"  Error rendering {view_name} view: {e}")
            import traceback

            traceback.print_exc()


def _visualize_with_open3d(
    pc_array, boxes, class_names, save_path, view_angle, interactive
):
    """
    Visualize using Open3D (may not work in headless environments).
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available")

    from .util_geometry import get_pc_for_vis, get_bbox_for_vis

    # Create Open3D point cloud
    pcd = get_pc_for_vis(pc_array, color="gray")

    # Create bounding box line sets
    if len(boxes) > 0:
        line_sets_bbox = get_bbox_for_vis(boxes, class_names=class_names)
    else:
        line_sets_bbox = []

    # Combine geometries
    geometries = [pcd] + line_sets_bbox

    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0]
    )
    geometries.append(coord_frame)

    # Interactive visualization
    if interactive:
        print("Opening interactive 3D viewer (close window to continue)...")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Radar Point Cloud with Detections",
            width=1280,
            height=720,
            left=50,
            top=50,
        )

    # Save static images
    if save_path is not None:
        save_path = Path(save_path)

        # Define camera views
        views = {}

        if view_angle in ["top", "all"]:
            views["top"] = {
                "front": [0, 0, 1],
                "lookat": [20, 0, 0],
                "up": [1, 0, 0],
                "zoom": 0.5,
            }

        if view_angle in ["side", "all"]:
            views["side"] = {
                "front": [0, -1, 0],
                "lookat": [20, 0, 0],
                "up": [0, 0, 1],
                "zoom": 0.5,
            }

        if view_angle in ["perspective", "all"] or view_angle not in ["top", "side"]:
            views["perspective"] = {
                "front": [-1, -1, 1],
                "lookat": [20, 0, 0],
                "up": [0, 0, 1],
                "zoom": 0.5,
            }

        # Render and save each view
        for view_name, view_params in views.items():
            output_path = f"{save_path}_{view_name}.png"
            _save_view(geometries, output_path, view_params)
            print(f"Saved {view_name} view: {output_path}")


def _save_view(geometries, output_path, view_params):
    """
    Save a rendered view of the geometries to an image file using offscreen rendering.

    Args:
        geometries: List of Open3D geometry objects
        output_path: Path to save the image
        view_params: Dict with 'front', 'lookat', 'up', 'zoom' keys
    """
    # Use OffscreenRenderer for headless environments (WSL, no display)
    try:
        # Try using the visualizer first (works if display is available)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1920, height=1080)

        for geom in geometries:
            vis.add_geometry(geom)

        # Set view parameters
        ctr = vis.get_view_control()
        if ctr is not None:
            ctr.set_front(view_params["front"])
            ctr.set_lookat(view_params["lookat"])
            ctr.set_up(view_params["up"])
            ctr.set_zoom(view_params["zoom"])

        # Render and save
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(output_path), do_render=True)
        vis.destroy_window()
    except Exception as e:
        # Fallback to offscreen rendering for headless environments
        print(f"Note: Using offscreen rendering (display not available)")

        # Create offscreen renderer
        width, height = 1920, 1080
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        # Set up material for point cloud and lines
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0
        mat.line_width = 2.0

        # Add geometries to scene
        for i, geom in enumerate(geometries):
            renderer.scene.add_geometry(f"geom_{i}", geom, mat)

        # Set up camera
        center = view_params["lookat"]
        eye = [
            center[0] + view_params["front"][0] * 50,
            center[1] + view_params["front"][1] * 50,
            center[2] + view_params["front"][2] * 50,
        ]
        up = view_params["up"]

        renderer.setup_camera(60.0, center, eye, up)

        # Render and save
        img = renderer.render_to_image()
        o3d.io.write_image(str(output_path), img)

        # Cleanup
        renderer.scene.clear_geometry()


def create_detection_summary_text(detections, conf_threshold=0.3):
    """
    Create a text summary of detections for logging or overlay.

    Args:
        detections: Detection dictionary
        conf_threshold: Minimum confidence threshold

    Returns:
        String with formatted detection summary
    """
    # Convert to numpy arrays if needed (inference returns lists)
    boxes = np.array(detections.get("boxes", []))
    scores = np.array(detections.get("scores", []))
    labels = np.array(detections.get("labels", []))
    class_names_list = detections.get("class_names", [])

    if len(boxes) == 0:
        return "No detections"

    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    summary_lines = [
        f"Detections: {len(boxes)} objects (threshold: {conf_threshold:.2f})"
    ]

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cls_name = class_names_list[int(label)]
        x, y, z, dx, dy, dz, heading = box
        summary_lines.append(
            f"  [{i+1}] {cls_name} ({score:.3f}): "
            f"pos=({x:.1f}, {y:.1f}, {z:.1f}) "
            f"size=({dx:.1f}x{dy:.1f}x{dz:.1f}) "
            f"heading={np.degrees(heading):.1f}°"
        )

    return "\n".join(summary_lines)
