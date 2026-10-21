"""
Process-Aware No-Spray Masking & Actuation Lookahead Engine
Part of irdpl_toolpath_verifier package.
"""

from typing import List, Dict, Any
import numpy as np
from scipy.spatial.transform import Rotation as R


class ZoneMasker:
    def __init__(
        self,
        valve_actuation_delay_s: float = 0.050,   # 50 ms mechanical delay
        nominal_tcp_speed_mps: float = 0.200,      # 200 mm/s robot travel speed
        overspray_safety_margin_m: float = 0.010,  # 10 mm buffer
        liftoff_height_m: float = 0.030            # 30 mm Z-retract for bypass
    ):
        self.valve_delay = valve_actuation_delay_s
        self.tcp_speed = nominal_tcp_speed_mps
        self.overspray_margin = overspray_safety_margin_m
        self.liftoff_height = liftoff_height_m

        # Total spatial lookahead distance
        self.lookahead_distance = (self.tcp_speed * self.valve_delay) + self.overspray_margin

    def is_point_in_zone(self, point: np.ndarray, zone: Dict[str, Any]) -> bool:
        """Checks if a 3D point falls inside a zone's Axis-Aligned Bounding Box (AABB)."""
        min_b = np.array(zone['min_bound'])
        max_b = np.array(zone['max_bound'])
        return bool(np.all(point >= min_b) and np.all(point <= max_b))

    def apply_masking_and_lookahead(
        self,
        waypoints: List[Dict[str, Any]],
        zones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Applies lookahead time-shift compensation and tags waypoints with spray actuation states.
        """
        if not waypoints or not zones:
            for pt in waypoints:
                pt['spray_trigger'] = True
                pt['liftoff_applied'] = False
            return waypoints

        processed_waypoints = [dict(pt) for pt in waypoints]
        num_pts = len(processed_waypoints)

        # 1. Base spatial tagging
        for i, pt in enumerate(processed_waypoints):
            pos = np.array(pt['pos'])
            in_zone = False
            active_mode = None

            for zone in zones:
                if self.is_point_in_zone(pos, zone):
                    in_zone = True
                    active_mode = zone.get('action_mode', 'DIGITAL_SHUTOFF')
                    break

            pt['in_keepout_zone'] = in_zone
            pt['action_mode'] = active_mode
            pt['spray_trigger'] = not in_zone
            pt['liftoff_applied'] = False

        # 2. Lookahead Time-Shift Compensation
        # The valve must close BEFORE entering the zone due to lag
        # Estimate index lookahead steps
        if num_pts > 1:
            avg_step = float(np.mean([
                np.linalg.norm(np.array(processed_waypoints[j+1]['pos']) - np.array(processed_waypoints[j]['pos']))
                for j in range(num_pts - 1)
            ]))
            lookahead_steps = max(1, int(np.round(self.lookahead_distance / (avg_step + 1e-9))))
        else:
            lookahead_steps = 1

        # Shift the trigger OFF command earlier by lookahead_steps
        shifted_triggers = [pt['spray_trigger'] for pt in processed_waypoints]
        for i in range(num_pts):
            # If a waypoint ahead enters a keep-out zone, shut off gun NOW
            look_forward_idx = min(num_pts - 1, i + lookahead_steps)
            if processed_waypoints[look_forward_idx]['in_keepout_zone']:
                shifted_triggers[i] = False

        # Apply the compensated trigger states
        for i in range(num_pts):
            processed_waypoints[i]['spray_trigger'] = shifted_triggers[i]

        # 3. Apply Liftoff Arc if zone requires 'LIFTOFF_BYPASS'
        for pt in processed_waypoints:
            if pt['in_keepout_zone'] and pt.get('action_mode') == 'LIFTOFF_BYPASS':
                # Extract orientation quaternion to find local tool approach vector
                rot = R.from_quat(pt['quat'])
                # Offset along the tool approach vector (Z-axis) to lift away from workpiece
                approach_vector = rot.apply([0.0, 0.0, 1.0])
                pt['pos'] = (np.array(pt['pos']) + (approach_vector * self.liftoff_height)).tolist()
                pt['liftoff_applied'] = True

        return processed_waypoints
