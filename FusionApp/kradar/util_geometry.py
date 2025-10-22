"""
# -*- coding: utf-8 -*-
--------------------------------------------------------------------------------
# Utility functions for 3D geometry and visualization
# Adapted from K-Radar dataset utilities
--------------------------------------------------------------------------------
"""

import numpy as np
import open3d as o3d


class Object3D:
    """
    Represents a 3D bounding box for object detection visualization.

    The box is defined by center position, dimensions, and rotation around Z-axis.
    Generates 8 corner points of the 3D bounding box.
    """

    def __init__(self, xc, yc, zc, xl, yl, zl, rot_rad):
        """
        Args:
            xc, yc, zc: Center coordinates (x: forward, y: left, z: up)
            xl, yl, zl: Dimensions (length, width, height)
            rot_rad: Rotation around Z-axis in radians
        """
        self.xc, self.yc, self.zc = xc, yc, zc
        self.xl, self.yl, self.zl = xl, yl, zl
        self.rot_rad = rot_rad

        # Generate 8 corners of the box (before rotation and translation)
        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2

        self.corners = np.row_stack((corners_x, corners_y, corners_z))

        # Apply rotation around Z-axis
        rotation_matrix = np.array(
            [
                [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        # Rotate and translate to final position
        self.corners = rotation_matrix.dot(self.corners).T + np.array(
            [[self.xc, self.yc, self.zc]]
        )


def get_pc_for_vis(pc, color=None):
    """
    Convert numpy point cloud to Open3D PointCloud object for visualization.

    Args:
        pc: Nx3 or Nx4 numpy array (x, y, z, [power/intensity])
        color: Color specification - 'black', 'gray', 'power', or RGB list [r, g, b] in [0, 1]
               If 'power' is specified and pc has 4 columns, colors by power (light gray to black)

    Returns:
        o3d.geometry.PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    num_points, _ = pc.shape

    if color == "black":
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pc[:, :3]))
    elif color == "gray":
        pcd.colors = o3d.utility.Vector3dVector(
            np.repeat(np.array([0.8, 0.8, 0.8])[np.newaxis, :], num_points, axis=0)
        )
    elif color == "power" and pc.shape[1] >= 4:
        # Color code by power: normalize power values and map to grayscale
        # Medium gray (low power) to black (high power)
        power = pc[:, 3]
        power_min, power_max = power.min(), power.max()
        if power_max > power_min:
            power_norm = (power - power_min) / (power_max - power_min)
        else:
            power_norm = np.zeros_like(power)
        # Map to grayscale: medium gray (0.5) to black (0.0)
        # Higher power = darker color
        gray_values = 0.5 - (
            power_norm * 0.5
        )  # Range from 0.5 (medium gray) to 0.0 (black)
        colors_array = np.repeat(gray_values[:, np.newaxis], 3, axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors_array)
    elif color is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.repeat(np.array(color)[np.newaxis, :], num_points, axis=0)
        )

    return pcd


def get_bbox_for_vis(bboxes, class_names=None, colors=None):
    """
    Convert bounding box list to Open3D LineSet objects for visualization.

    Args:
        bboxes: Nx7 numpy array [x, y, z, dx, dy, dz, heading] or
                List of [class_name, class_idx, [x,y,z,theta,l,w,h], obj_idx]
        class_names: List of class names (optional)
        colors: Dict mapping class names to RGB colors (optional)

    Returns:
        List of o3d.geometry.LineSet objects
    """
    # Define edges connecting the 8 corners
    lines = [
        [0, 1],
        [2, 3],  # Top edges
        [4, 5],
        [6, 7],  # Bottom edges
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # Vertical edges
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],  # Side edges
    ]

    # Default color scheme for different classes
    default_colors = {
        "Sedan": [0.09, 0.82, 0.99],  # Cyan
        "Bus or Truck": [0.0, 0.2, 1.0],  # Blue
        "Motorcycle": [1.0, 0.0, 0.0],  # Red
        "Bicycle": [1.0, 1.0, 0.0],  # Yellow
        "Pedestrian": [1.0, 0.0, 0.0],  # Red
        "Pedestrian Group": [1.0, 0.0, 0.4],  # Pink
        "default": [1.0, 0.0, 0.0],  # Red for unknown classes
    }

    if colors is not None:
        default_colors.update(colors)

    bboxes_o3d = []
    line_sets_bbox = []

    # Handle different input formats
    if isinstance(bboxes, np.ndarray):
        # Format: Nx7 [x, y, z, dx, dy, dz, heading]
        for i, bbox in enumerate(bboxes):
            x, y, z, dx, dy, dz, heading = bbox
            bboxes_o3d.append(Object3D(x, y, z, dx, dy, dz, heading))

            # Get color for this bbox
            if class_names is not None and i < len(class_names):
                cls_name = class_names[i]
                bbox_color = default_colors.get(cls_name, default_colors["default"])
            else:
                bbox_color = default_colors["default"]

            colors_bbox = [bbox_color for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bboxes_o3d[-1].corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
            line_sets_bbox.append(line_set)
    else:
        # Format: List of [class_name, class_idx, [x,y,z,theta,l,w,h], obj_idx]
        for obj in bboxes:
            cls_name, _, [x, y, z, theta, l, w, h], _ = obj
            bboxes_o3d.append(Object3D(x, y, z, l, w, h, theta))

            bbox_color = default_colors.get(cls_name, default_colors["default"])
            colors_bbox = [bbox_color for _ in range(len(lines))]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(bboxes_o3d[-1].corners)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors_bbox)
            line_sets_bbox.append(line_set)

    return line_sets_bbox
