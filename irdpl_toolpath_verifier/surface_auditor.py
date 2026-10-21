"""
Safety-Critical Geometric & Surface Continuity Auditor
Part of irdpl_toolpath_verifier package.
"""

from enum import Enum
from typing import List, Dict, Tuple, Any
import numpy as np
import open3d as o3d


class GapClassification(Enum):
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    STRUCTURAL_VOID = "STRUCTURAL_VOID"
    GEOMETRIC_ANOMALY = "GEOMETRIC_ANOMALY"


class SurfaceAuditor:
    def __init__(
        self,
        standoff_distance: float = 0.150,
        expected_point_spacing: float = 0.005,
        gap_detection_threshold: float = 0.015,
        max_bridgeable_gap: float = 0.120,
        corridor_radial_margin: float = 0.035,
        min_mesh_points_for_metal: int = 8,
        min_normal_collinearity: float = 0.707,
    ):
        self.standoff = standoff_distance
        self.spacing = expected_point_spacing
        self.gap_threshold = gap_detection_threshold
        self.max_gap = max_bridgeable_gap
        self.corridor_margin = corridor_radial_margin
        self.min_points = min_mesh_points_for_metal
        self.min_collinearity = min_normal_collinearity

    def scan_for_discontinuities(
        self, waypoints: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        gaps = []
        if len(waypoints) < 2:
            return gaps

        for i in range(len(waypoints) - 1):
            p0 = np.array(waypoints[i]['pos'])
            p1 = np.array(waypoints[i + 1]['pos'])
            dist = float(np.linalg.norm(p1 - p0))

            if dist > self.gap_threshold:
                gaps.append((i, i + 1, dist))

        return gaps

    def audit_gap(
        self,
        p_entry: np.ndarray,
        p_exit: np.ndarray,
        q_entry: np.ndarray,
        q_exit: np.ndarray,
        mesh: o3d.geometry.TriangleMesh,
        kdtree: o3d.geometry.KDTreeFlann,
    ) -> Dict[str, Any]:
        # Defensive check: ensure vertex normals are computed
        if not mesh.has_vertex_normals() or len(mesh.vertex_normals) == 0:
            mesh.compute_vertex_normals()

        gap_vector = p_exit - p_entry
        gap_length = float(np.linalg.norm(gap_vector))

        if gap_length > self.max_gap:
            return {
                "classification": GapClassification.STRUCTURAL_VOID,
                "reason": f"Gap length ({gap_length*1000:.1f}mm) exceeds maximum safety limit ({self.max_gap*1000:.1f}mm)",
                "gap_length": gap_length,
                "metal_points_count": 0,
                "collinearity": 0.0,
            }

        _, idx_entry, _ = kdtree.search_knn_vector_3d(p_entry, 1)
        _, idx_exit, _ = kdtree.search_knn_vector_3d(p_exit, 1)

        normal_entry = np.asarray(mesh.vertex_normals)[idx_entry[0]]
        normal_exit = np.asarray(mesh.vertex_normals)[idx_exit[0]]

        normal_entry = normal_entry / (np.linalg.norm(normal_entry) + 1e-9)
        normal_exit = normal_exit / (np.linalg.norm(normal_exit) + 1e-9)

        collinearity = float(np.dot(normal_entry, normal_exit))

        if collinearity < self.min_collinearity:
            return {
                "classification": GapClassification.GEOMETRIC_ANOMALY,
                "reason": f"Surface normals diverge significantly (collinearity = {collinearity:.3f} < {self.min_collinearity})",
                "gap_length": gap_length,
                "metal_points_count": 0,
                "collinearity": collinearity,
            }

        num_samples = max(5, int(np.ceil(gap_length / 0.010)))
        sample_fractions = np.linspace(0.1, 0.9, num_samples)

        mesh_points_detected = 0
        search_radius = self.standoff + self.corridor_margin

        for t in sample_fractions:
            corridor_point = (1 - t) * p_entry + t * p_exit
            count, _, _ = kdtree.search_radius_vector_3d(corridor_point, search_radius)
            if count > 0:
                mesh_points_detected += count

        if mesh_points_detected >= self.min_points:
            return {
                "classification": GapClassification.SENSOR_DROPOUT,
                "reason": "Physical surface confirmed underneath gap corridor; normals collinear.",
                "gap_length": gap_length,
                "metal_points_count": mesh_points_detected,
                "collinearity": collinearity,
            }
        else:
            return {
                "classification": GapClassification.STRUCTURAL_VOID,
                "reason": f"Insufficient mesh points inside corridor ({mesh_points_detected} < {self.min_points}). True physical void.",
                "gap_length": gap_length,
                "metal_points_count": mesh_points_detected,
                "collinearity": collinearity,
            }
