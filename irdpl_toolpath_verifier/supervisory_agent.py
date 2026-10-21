"""
Supervisory Cognitive Quality & Safety Gatekeeper
Part of irdpl_toolpath_verifier package.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import open3d as o3d

from irdpl_toolpath_verifier.surface_auditor import SurfaceAuditor, GapClassification
from irdpl_toolpath_verifier.path_synthesizer import PathSynthesizer
from irdpl_toolpath_verifier.zone_masker import ZoneMasker


class SupervisoryAgent:
    def __init__(
        self,
        auditor: SurfaceAuditor,
        synthesizer: PathSynthesizer,
        masker: ZoneMasker
    ):
        self.auditor = auditor
        self.synthesizer = synthesizer
        self.masker = masker

    def run_full_pipeline(
        self,
        raw_waypoints: List[Dict[str, Any]],
        mesh: o3d.geometry.TriangleMesh,
        kdtree: o3d.geometry.KDTreeFlann,
        no_spray_zones: List[Dict[str, Any]]
    ) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes the end-to-end verification, bridging, and masking workflow.
        Returns: (success: bool, final_waypoints: list, audit_report: dict)
        """
        report = {
            "gaps_detected": 0,
            "gaps_bridged": 0,
            "masked_waypoints": 0,
            "aborted": False,
            "reason": ""
        }

        # Step 1: Scan for discontinuities
        gaps = self.auditor.scan_for_discontinuities(raw_waypoints)
        report["gaps_detected"] = len(gaps)

        # Step 2: Audit every detected gap
        confirmed_bridging_gaps = []
        for gap in gaps:
            i_start, i_end, dist = gap
            p0 = np.array(raw_waypoints[i_start]["pos"])
            p1 = np.array(raw_waypoints[i_end]["pos"])
            q0 = np.array(raw_waypoints[i_start]["quat"])
            q1 = np.array(raw_waypoints[i_end]["quat"])

            audit_res = self.auditor.audit_gap(p0, p1, q0, q1, mesh, kdtree)

            if audit_res["classification"] == GapClassification.SENSOR_DROPOUT:
                confirmed_bridging_gaps.append(gap)
            else:
                # If there is a true structural void or anomaly, halt for safety!
                report["aborted"] = True
                report["reason"] = f"Safety Halt: Gap at indices ({i_start}, {i_end}) classified as {audit_res['classification'].value}. {audit_res['reason']}"
                return False, raw_waypoints, report

        # Step 3: Synthesize bridges across confirmed gaps
        healed_waypoints = self.synthesizer.heal_trajectory(raw_waypoints, confirmed_bridging_gaps, mesh, kdtree)
        report["gaps_bridged"] = len(confirmed_bridging_gaps)

        # Step 4: Apply No-Spray Masking & Latency Compensation
        final_waypoints = self.masker.apply_masking_and_lookahead(healed_waypoints, no_spray_zones)
        report["masked_waypoints"] = sum(1 for pt in final_waypoints if not pt.get("spray_trigger", True))

        # Step 5: Final Quality Invariant Gatekeeper
        is_safe, error_msg = self.verify_invariants(final_waypoints)
        if not is_safe:
            report["aborted"] = True
            report["reason"] = f"Invariant Gatekeeper Failed: {error_msg}"
            return False, raw_waypoints, report

        report["reason"] = "Trajectory successfully audited, healed, and masked. Ready for execution."
        return True, final_waypoints, report

    def verify_invariants(self, waypoints: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Final sanity checks before sending waypoints to Tesseract."""
        if len(waypoints) < 2:
            return False, "Trajectory contains fewer than 2 waypoints."

        # Invariant 1: Continuous spacing
        for i in range(len(waypoints) - 1):
            p0 = np.array(waypoints[i]["pos"])
            p1 = np.array(waypoints[i+1]["pos"])
            dist = float(np.linalg.norm(p1 - p0))
            if dist > (self.auditor.spacing * 1.5):
                return False, f"Spatial gap remaining between points {i} and {i+1} ({dist*1000:.1f} mm)."

        # Invariant 2: Quaternion normalization
        for i, pt in enumerate(waypoints):
            norm = float(np.linalg.norm(pt["quat"]))
            if abs(norm - 1.0) > 1e-3:
                return False, f"Waypoint {i} has unnormalized quaternion (norm={norm:.4f})."

        return True, "All invariants passed."
