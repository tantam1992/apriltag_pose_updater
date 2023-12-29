#!/usr/bin/env python3

import rospy
import tf
import numpy as np 
# import yaml

from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Pose


def poseCallback(data):
    tag_locations = rospy.get_param("/tags")    # get tag locations from yaml 
    # tag_locations = read_tag_poses("../tag_poses.yaml")
    # listener = tf.TransformListener()
    # listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(3.0))
    # (trans, rot) = listener.lookupTransform("map", "base_link", rospy.Time(0))
    
    tf_ros = tf.TransformerROS()

    # if not data.detections:
    #     map_base_link_pub.publish(map_base_link_data)
    # else:
    if data.detections:
        map_base_link_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=10)

        map_base_link_data = PoseWithCovarianceStamped()
        map_base_link_data.header.seq = data.header.seq
        map_base_link_data.header.stamp = data.header.stamp
        map_base_link_data.header.frame_id = "map"

        # tag w.r.t. map
        tag_detected = []
        for i in range(len(data.detections)):
            if data.detections[i].pose.pose.pose.position.z <= 10.0:
                tag_detected.append(data.detections[i])
        
        if tag_detected:
            tag_sorted = sorted(tag_detected, key=lambda tag_detected: tag_detected.pose.pose.pose.position.z)

            for tag in tag_locations:
                if tag["id"] == int(''.join(map(str, tag_sorted[0].id))): 
                    break
            map_tag_g = tf_ros.fromTranslationRotation([tag["x"], tag["y"], tag["z"]], [tag["qx"], tag["qy"], tag["qz"], tag["qw"]]) 
            
            # tag w.r.t. usb_cam_link
            usb_cam_link_tag_t = [tag_sorted[0].pose.pose.pose.position.x, tag_sorted[0].pose.pose.pose.position.y, tag_sorted[0].pose.pose.pose.position.z]
            usb_cam_link_tag_R = [tag_sorted[0].pose.pose.pose.orientation.x, tag_sorted[0].pose.pose.pose.orientation.y, tag_sorted[0].pose.pose.pose.orientation.z, tag_sorted[0].pose.pose.pose.orientation.w]
            usb_cam_link_tag_g = tf_ros.fromTranslationRotation(usb_cam_link_tag_t, usb_cam_link_tag_R)

            # usb_cam w.r.t. base_link
            base_link_usb_cam_link_g = tf_ros.fromTranslationRotation([0.230, 0.000, 0.104], [-0.500, 0.500, -0.500, 0.500])

            # calculation of base_link w.r.t. map
            tag_usb_cam_link_g = np.linalg.inv(usb_cam_link_tag_g)                  # usb_cam_link w.r.t. tag
            usb_cam_link_base_link_g = np.linalg.inv(base_link_usb_cam_link_g)      # base_link w.r.t. usb_cam_link

            map_base_link_g = np.matmul(map_tag_g, np.matmul(tag_usb_cam_link_g, usb_cam_link_base_link_g))

            # get rotation and translation matrix from transformation matrix 
            map_base_link_t = map_base_link_g[:3, 3]
            # map_base_link_R = map_base_link_g[:3, :3]
            map_base_link_q = tf.transformations.quaternion_from_matrix(map_base_link_g)

            map_base_link_data.pose.pose.position.x = map_base_link_t[0]
            map_base_link_data.pose.pose.position.y = map_base_link_t[1]
            map_base_link_data.pose.pose.position.z = map_base_link_t[2]
            map_base_link_data.pose.pose.orientation.x = map_base_link_q[0]
            map_base_link_data.pose.pose.orientation.y = map_base_link_q[1]
            map_base_link_data.pose.pose.orientation.z = map_base_link_q[2]
            map_base_link_data.pose.pose.orientation.w = map_base_link_q[3]

            map_base_link_pub.publish(map_base_link_data)

def read_tag_poses(file_path):
        with open(file_path, 'r') as file:
            tag_data = yaml.safe_load(file)
            tag_poses = {}
            for tag_info in tag_data.get('tags', []):
                tag_id = tag_info.get('id')
                if tag_id is not None:
                    tag_poses[tag_id] = {
                        'position': [
                            tag_info.get('x', 0.0),
                            tag_info.get('y', 0.0),
                            tag_info.get('z', 0.0),
                        ],
                        'orientation': [
                            tag_info.get('qx', 0.0),
                            tag_info.get('qy', 0.0),
                            tag_info.get('qz', 0.0),
                            tag_info.get('qw', 1.0),
                        ],
                    }
            return tag_poses
        
def posePublisher():
    rospy.init_node('apriltag_pose_updater', anonymous=True)

    rospy.Subscriber("tag_detections", AprilTagDetectionArray, poseCallback)

    rospy.loginfo("Getting transformation from base_link to map.")
    
    rospy.spin()
    

if __name__ == '__main__':
    try:
        posePublisher()
    except rospy.ROSInterruptException:
        pass
