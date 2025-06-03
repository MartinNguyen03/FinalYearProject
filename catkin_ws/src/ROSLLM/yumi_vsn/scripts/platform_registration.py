import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import distance

import rospy
from sensor_msgs.msg import Image, PointCloud2

from utils.images import msg_to_img
from utils.point_clouds import xy_to_yx, read_point_from_region
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, euler_from_matrix, quaternion_matrix

class PlatformRegistration:

    def __init__(self):
        # Load the ArUco dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_81 = None
        self.marker_82 = None
        self.marker_83 = None
        self.marker_84 = None
        
    def print_markers(self):
        ''' Print required Aruco tags onto imgs '''
        img = np.zeros((300, 300, 1), dtype="uint8")
        for id in range(81,85):
            self.aruco_detector.drawMarker(self.aruco_dict, id, 300, img, 1)
            cv2.imwrite("aruco_{}.png".format(id), img)
            cv2.imshow("ArUCo Tag", img)
            cv2.waitKey(0)

    

    def register_platform(self, img, depth, camera_intrinsics, offset=None):
        try:
            (corners, ids, _) = self.aruco_detector.detectMarkers(img)

            if corners is None or len(corners) == 0 or ids is None:
                print("NO Aruco detected!")
                return None

            ids = ids.flatten()
            ids = np.array(ids)
            ids_sorted_idx = ids.argsort()
            ids = ids[ids_sorted_idx]
            corners = np.array(corners)[ids_sorted_idx]

            for marker_id, corner in zip(ids, corners):
                centre_2d = corner.reshape((4, 2)).mean(axis=0).astype(int)
                if offset is not None:
                    centre_2d += np.array(offset)
                centre_3d = read_point_from_region(xy_to_yx(centre_2d), depth, 3, camera_intrinsics=camera_intrinsics)

                if centre_3d is not None and not np.isnan(centre_3d).any():
                    if marker_id == 81:
                        self.marker_81 = centre_3d
                    elif marker_id == 82:
                        self.marker_82 = centre_3d
                    elif marker_id == 83:
                        self.marker_83 = centre_3d
                    elif marker_id == 84:
                        self.marker_84 = centre_3d

           
            op_vertices_3d = []
            missing_ids = []
            for marker_id in [81, 82, 83, 84]:
                vertex = getattr(self, f"marker_{marker_id}")
                if vertex is not None:
                    op_vertices_3d.append(vertex)
                else:
                    missing_ids.append(marker_id)

            if len(op_vertices_3d) != 4:
                print("Detection incomplete! Missing markers: {}".format(missing_ids))
                return None

            return np.array(op_vertices_3d)

        except cv2.error as e:
            print("Error in marker detection: {}".format(e))
            return None



    def check_markers(self, frame):        
        # Detect ArUco markers in the video frame
        (corners, ids, _) = self.aruco_detector.detectMarkers(frame)
        
        # Check that at least one ArUco marker was detected
        if len(corners) > 0:
        # Flatten the ArUco IDs list
            ids = ids.flatten()
        else:
            rospy.sleep(1)
            return
        
        # Loop over the detected ArUco corners
        for (marker_corner, marker_id) in zip(corners, ids):
            
            # Extract the marker corners
            corners = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners
            
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            
            # Draw the bounding box of the ArUco detection
            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
            
            # Calculate and draw the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)
            
            
            # Draw the ArUco marker ID on the video frame
            # The ID is always located at the top_left of the ArUco marker
            cv2.putText(frame, str(marker_id), (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    
    from tf.transformations import quaternion_matrix

    

    
if __name__ == '__main__':
    print(cv2.__version__)
    rospy.init_node('platform_registration', anonymous=True)
    pr = PlatformRegistration()
    while not rospy.is_shutdown():
        pr.check_markers()
        # Close down the video stream
        cv2.destroyAllWindows()