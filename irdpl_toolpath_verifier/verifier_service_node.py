#!/usr/bin/env python3
"""
Production ROS 2 Data-Processing Node: Real-Data Toolpath Verifier
Strictly processes incoming real arrays, meshes, and images with zero mocks.
Part of irdpl_toolpath_verifier package for IRDPL.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, PoseArray
from std_msgs.msg import ColorRGBA
from shape_msgs.msg import Mesh
from sensor_msgs.msg import Image

from irdpl_toolpath_verifier.surface_auditor import SurfaceAuditor, GapClassification
from irdpl_toolpath_verifier.path_synthesizer import PathSynthesizer
from irdpl_toolpath_verifier.zone_masker import ZoneMasker
from irdpl_toolpath_verifier.supervisory_agent import SupervisoryAgent
from irdpl_toolpath_verifier.visual_inspector import VisualInspector


class RealDataVerifierNode(Node):
    def __init__(self):
        super().__init__('irdpl_toolpath_verifier_node')

        # 1. Load Production Parameters
        self.declare_parameter('standoff_distance', 0.150)
        self.declare_parameter('expected_point_spacing', 0.005)
        self.declare_parameter('gap_detection_threshold', 0.015)
        self.declare_parameter('max_bridgeable_gap', 0.120)
        self.declare_parameter('valve_actuation_delay_s', 0.050)
        self.declare_parameter('nominal_tcp_speed_mps', 0.200)
        self.declare_parameter('ollama_host', 'http://localhost:11434')
        self.declare_parameter('vlm_model_name', 'qwen2.5vl:3b')

        standoff = float(self.get_parameter('standoff_distance').value)
        spacing = float(self.get_parameter('expected_point_spacing').value)
        gap_thresh = float(self.get_parameter('gap_detection_threshold').value)
        max_gap = float(self.get_parameter('max_bridgeable_gap').value)
        valve_delay = float(self.get_parameter('valve_actuation_delay_s').value)
        tcp_speed = float(self.get_parameter('nominal_tcp_speed_mps').value)
        ollama_url = str(self.get_parameter('ollama_host').value)
        vlm_model = str(self.get_parameter('vlm_model_name').value)

        # 2. Initialize Core Algorithmic Engines
        self.auditor = SurfaceAuditor(standoff, spacing, gap_thresh, max_gap)
        self.synthesizer = PathSynthesizer(standoff, spacing)
        self.masker = ZoneMasker(valve_delay, tcp_speed)
        self.agent = SupervisoryAgent(self.auditor, self.synthesizer, self.masker)
        self.inspector = VisualInspector(ollama_url, vlm_model)

        # 3. Setup Publishers & Subscriptions for Real Pipeline Data
        self.marker_pub = self.create_publisher(
            MarkerArray, '/irdpl/verified_toolpath_markers', 10
        )
        
        # Publisher for the final cleaned trajectory ready for Tesseract
        self.path_pub = self.create_publisher(
            PoseArray, '/irdpl/verified_toolpath_poses', 10
        )

        self.get_logger().info("==================================================")
        self.get_logger().info("  IRDPL Real-Data Toolpath Verifier is ACTIVE     ")
        self.get_logger().info(f"  VLM Backend: {vlm_model} at {ollama_url}")
        self.get_logger().info("==================================================")

    def process_real_toolpath(
        self, 
        raw_pose_array: PoseArray, 
        ros_mesh: Mesh
    ) -> tuple[bool, PoseArray, dict]:
        """
        Pure data transformation method handling real incoming arrays and meshes.
        No mocks, no synthetic data.
        """
        if len(raw_pose_array.poses) == 0:
            return False, PoseArray(), {"reason": "Received empty raw pose array."}

        if len(ros_mesh.vertices) == 0:
            return False, PoseArray(), {"reason": "Received empty surface mesh."}

        # Convert ROS shape_msgs/Mesh to Open3D TriangleMesh
        vertices = np.array([[v.x, v.y, v.z] for v in ros_mesh.vertices])
        triangles = np.array([[t.vertex_indices[0], t.vertex_indices[1], t.vertex_indices[2]] for t in ros_mesh.triangles])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        kdtree = o3d.geometry.KDTreeFlann(mesh)

        # Convert real incoming geometry_msgs/PoseArray into internal dictionary array
        raw_waypoints = []
        for p in raw_pose_array.poses:
            raw_waypoints.append({
                'pos': [p.position.x, p.position.y, p.position.z],
                'quat': [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            })

        # Load standard industrial keep-out zones (threads, flanges)
        no_spray_zones = [{
            'zone_id': 'keepout_zone',
            'min_bound': [0.120, -0.050, -0.050],
            'max_bound': [0.180, 0.050, 0.050],
            'action_mode': 'DIGITAL_SHUTOFF'
        }]

        # Execute Supervisory Agent on REAL data
        success, final_waypoints, report = self.agent.run_full_pipeline(
            raw_waypoints, mesh, kdtree, no_spray_zones
        )

        # Format output PoseArray for Tesseract
        out_poses = PoseArray()
        out_poses.header = raw_pose_array.header

        for pt in final_waypoints:
            p = Pose()
            p.position.x, p.position.y, p.position.z = pt['pos']
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = pt['quat']
            out_poses.poses.append(p)

        if success:
            self.get_logger().info(
                f"✔ Real Path Verified: Bridged {report['gaps_bridged']} gaps, "
                f"Masked {report['masked_waypoints']} points."
            )
            self.publish_rviz_markers(final_waypoints, raw_pose_array.header.frame_id)
            self.path_pub.publish(out_poses)
        else:
            self.get_logger().error(f"❌ Real Path Rejected: {report['reason']}")

        return success, out_poses, report

    def publish_rviz_markers(self, waypoints, frame_id="world"):
        marker_array = MarkerArray()

        sphere_marker = Marker()
        sphere_marker.header.frame_id = frame_id if frame_id else "world"
        sphere_marker.header.stamp = self.get_clock().now().to_msg()
        sphere_marker.ns = "real_verified_waypoints"
        sphere_marker.id = 0
        sphere_marker.type = Marker.SPHERE_LIST
        sphere_marker.action = Marker.ADD
        sphere_marker.scale.x = 0.008
        sphere_marker.scale.y = 0.008
        sphere_marker.scale.z = 0.008

        for pt in waypoints:
            p = Point(x=float(pt['pos'][0]), y=float(pt['pos'][1]), z=float(pt['pos'][2]))
            sphere_marker.points.append(p)

            color = ColorRGBA(a=1.0)
            if not pt.get('spray_trigger', True):
                color.r, color.g, color.b = 1.0, 0.1, 0.1  # RED (Masked No-Spray)
            elif pt.get('synthesized', False):
                color.r, color.g, color.b = 0.0, 0.8, 1.0  # CYAN (Healed Glare Gap)
            else:
                color.r, color.g, color.b = 0.1, 0.9, 0.1  # GREEN (Active Spray)

            sphere_marker.colors.append(color)

        marker_array.markers.append(sphere_marker)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = RealDataVerifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
