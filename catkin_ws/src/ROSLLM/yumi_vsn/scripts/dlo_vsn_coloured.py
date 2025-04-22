#!/usr/bin/python3
import os, sys
import cv2
import time
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm
from math import isnan, pi, atan2, sqrt

import rospy
from tf import TransformListener
from tf.transformations import euler_from_quaternion, compose_matrix, quaternion_from_euler
np.float = np.float64
from ros_numpy import msgify
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, PoseArray, Quaternion
from rosllm_srvs.srv import DetectTarget, DetectTargetRequest, DetectTargetResponse
from rosllm_srvs.srv import DetectObject, DetectObjectRequest, DetectObjectResponse
from utils.yumi_camera import RealCamera
from utils.point_clouds import depth_pixel_to_metric_coordinate, read_point_from_region, read_points_from_region, xy_to_yx, euclidian_distance
from utils.colour_segmentation import ColourSegmentation

MODEL_DIR = os.path.expanduser('~/Documents/FinalYearProject/FinalYearProject/dlo_perceiver')  # <-- update this
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "dlo_perceiver.pt")
from model_contrastive import DLOPerceiver
from text_encoder import TextEncoder
from transformers import DistilBertTokenizer

# SCENE:
#                x            x       x
# Target L_1 -->  | ----------- | <-- Target R_1        Ropes (--------) can criss-cross over each other, 
# Target L_2 -->  | ----------- | <-- Target R_2        Rope Definition: --(marker_A)-------(marker_B)--
# Target L_3 -->  | ----------- | <-- Target R_3
# Target L_4 -->  | ----------- | <-- Target R_4        Each Target will have it's own unique colour
# Target L_5 -->  | ----------- | <-- Target R_5        Each Marker will have it's own unique colour
# Target L_6 -->  | ----------- | <-- Target R_6        Each Rope will have it's own unique colour
# etc.             x        x     x                     X: Free space / intermediate checkpoints (Hardcoded/Or Not) to place rope avoid collision

class dloVision:
    debug = False
    processed_frame_topic = '/dlo_vsn/frame_processed'
    find_target_srv = 'detect_targets'
    find_object_srv = 'detect_objects'
    robot_frame = "yumi_base_link"
    l515_roi = [0, 180, 1280, 720] # 128[0, 0, 125], [110, 110, 0]0*720
    HSV_THRESHOLDS = {
    'red_1':     {'lower': rospy.get_param("~marker_thresh_red_low", [0, 100, 100]),    'upper': rospy.get_param("~marker_thresh_red_high", [10, 255, 255])},
    'red_2':     {'lower': rospy.get_param("~marker_thresh_red2_low", [160, 100, 100]),  'upper': rospy.get_param("~marker_thresh_red2_high", [179, 255, 255])},
    'green':     {'lower': rospy.get_param("~target_thresh_green_low", [40, 40, 40]),     'upper': rospy.get_param("~target_thresh_green_high", [80, 255, 255])},
    'blue':      {'lower': rospy.get_param("~marker_thresh_blue_low", [100, 150, 0]),    'upper': rospy.get_param("~marker_thresh_blue_high", [140, 255, 255])},
    'yellow':    {'lower': rospy.get_param("~target_thresh_yellow_low", [20, 100, 100]),   'upper': rospy.get_param("~target_thresh_yellow_high", [30, 255, 255])},
    'orange':    {'lower': rospy.get_param("~target_thresh_orange_low", [10, 100, 100]),   'upper': rospy.get_param("~target_thresh_orange_high", [20, 255, 255])},
    'purple':    {'lower': rospy.get_param("~target_thresh_purple_low", [130, 100, 100]),  'upper': rospy.get_param("~target_thresh_purple_high", [160, 255, 255])},
    'pink':      {'lower': rospy.get_param("~target_thresh_pink_low", [145, 100, 100]),  'upper': rospy.get_param("~target_thresh_pink_high", [170, 255, 255])},
    'brown':     {'lower': rospy.get_param("~target_thresh_brown_low", [10, 100, 20]),    'upper': rospy.get_param("~target_thresh_brown_high", [20, 255, 200])},
    'cyan':      {'lower': rospy.get_param("~target_thresh_cyan_low", [80, 100, 100]),   'upper': rospy.get_param("~target_thresh_cyan_high", [100, 255, 255])},
    'gray':      {'lower': rospy.get_param("~target_thresh_gray_low", [0, 0, 40]),       'upper': rospy.get_param("~target_thresh_gray_high", [179, 30, 200])},
    
    }
    
    class Rope:
        def __init__(self, cam, rope_name, auto_execution, marker_a_colour, marker_b_colour):
            self.name = rope_name
            self.cam = cam
            self.marker_a_colour = marker_a_colour
            self.marker_b_colour = marker_b_colour
            self.threshold_a, self.threshold_b = self.updateThreshold(marker_a_colour, marker_b_colour, auto_execution)
            self.priority = 0
            target_l_colour = None
            target_r_colour = None
            curr_target_l_colour = None
            curr_target_r_colour = None
         
        def updateTarget(self, target_l_colour, target_r_colour):
            self.target_l_colour = target_l_colour
            self.target_r_colour = target_r_colour
            self.threshold_l, self.threshold_r = self.updateThreshold(self.cam, self.target_l_colour, self.target_r_colour, auto_execution=False)
                      
        def updateCurrTarget(self, curr_target_l_colour, curr_target_r_colour):
            self.curr_target_l_colour = curr_target_l_colour
            self.curr_target_r_colour = curr_target_r_colour
            self.threshold_curr_l, self.threshold_curr_r = self.updateThreshold(self.cam, curr_target_l_colour, curr_target_r_colour, auto_execution=True)
            
        def updateThreshold(self, colour_1, colour_2, auto_execution):
            thresh_low_1 = self.HSV_THRESHOLDS[colour_1]['lower']
            thresh_high_1 = self.HSV_THRESHOLDS[colour_1]['upper']
            thresh_low_2 = self.HSV_THRESHOLDS[colour_2]['lower']
            thresh_high_2 = self.HSV_THRESHOLDS[colour_2]['upper']
            return ColourSegmentation(thresh_low_1, thresh_high_1, self.cam.read_image, live_adjust=not auto_execution), ColourSegmentation(thresh_low_2, thresh_high_2, self.cam.read_image, live_adjust=not auto_execution)
             
    class RopePerceiver:
        def __init__(self, cam):
            """
            Initialise model, tokenizer, and colour/view prompts.
            """
            self.cam = cam
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_dict = torch.load(MODEL_PATH)
            self.model_config = self.model_dict["config"]
            self.model_weights = self.model_dict["model"]
            self.model = DLOPerceiver(
                iterations=self.model_config["iterations"],
                n_latents=self.model_config["n_latents"],
                latent_dim=self.model_config["latent_dim"],
                depth=self.model_config["depth"],
                dropout=self.model_config["dropout"],
                img_encoder_type=self.model_config["img_encoder_type"],
            )
            self.model.load_state_dict(self.model_weights)
            self.model.to(device=self.device)
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.textEncoder = TextEncoder(model_name="distilbert-base-uncased")

            self.colours = ["red", "green", "blue", "gray", "yellow", "orange", "purple", "brown"]
            self.views = ["top", "bottom"]

        def _generatePrompt(self, colour, view):
            """
            Generate a text prompt using object name, colour, and view.
            """
            return f"{self.object_name} {colour} {view}"

        def _getTextEmbedding(self, text):
            """
            Convert text prompt into embedding using DistilBERT.
            """
            tokens = self.tokenizer([text], return_tensors="pt", padding=True)["input_ids"]
            textEmb = self.textEncoder(tokens).to(self.device)
            return textEmb

        def _prepareImage(self, img):
            """
            Resize and normalise image for model input.
            """
            img = cv2.resize(img, (self.model_config["img_w"], self.model_config["img_h"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            return img

        def _predictMask(self, imgTensor, textEmb):
            """
            Predict binary mask from image and text embedding.
            """
            with torch.no_grad():
                pred, (_, _, _) = self.model(imgTensor, textEmb)
            mask = pred.sigmoid().squeeze().detach().cpu().numpy()
            return mask

        def _isDuplicate(self, newMask, existingMasks, threshold=0.9):
            """
            Check if new mask is a duplicate based on IoU.
            """
            for mask in existingMasks:
                iou = self._maskIoU(newMask, mask)
                if iou > threshold:
                    return True
            return False

        def _maskIoU(self, mask1, mask2):
            """
            Compute Intersection over Union (IoU) between two masks.
            """
            mask1_bin = (mask1 > 0.5).astype(np.uint8)
            mask2_bin = (mask2 > 0.5).astype(np.uint8)
            intersection = np.logical_and(mask1_bin, mask2_bin).sum()
            union = np.logical_or(mask1_bin, mask2_bin).sum()
            return intersection / union if union != 0 else 0

        def segmentRope(self, imgTensor, colour, view):
            """
            Segment a rope by given colour and view. Returns binary mask.
            """
            prompt = self._generatePrompt(colour, view)
            textEmb = self._getTextEmbedding(prompt)
            mask = self._predictMask(imgTensor, textEmb)
            return (mask > 0.5).astype(np.uint8)

        def _rankRopesByHierarchy(self, ropeList):
            """
            Stub for hierarchical ranking. Compares overlap between masks.
            """
            # Example: sort ropes by average pixel intensity
            # You can improve this using occlusion detection logic later
            ropeList.sort(key=lambda r: r["mask"].sum(), reverse=True)
            return ropeList

        def countRopes(self, frame):
            """
            Detect and count unique ropes using mask similarity.
            Returns list of unique rope dicts with colour, view, prompt, and mask.
            """
            imgTensor = self._prepareImage(frame)
            uniqueRopes = []
            existingMasks = []

            for colour in self.colours:
                for view in self.views:
                    mask = self.segmentRope(imgTensor, colour, view)

                    if mask.max() < 0.1:
                        continue  # No activation

                    if not self._isDuplicate(mask, existingMasks):
                        existingMasks.append(mask)
                        uniqueRopes.append({
                            "colour": colour,
                            "view": view,
                            "prompt": self._generatePrompt(colour, view),
                            "mask": mask
                        })

            rankedRopes = self._rankRopesByHierarchy(uniqueRopes)
            return rankedRopes

        
    def __init__(self, auto_execution):
        
        # read ROS parameters
        self.robot_name = rospy.get_param("~robot_name", "yumi")
        
        self.auto_execution = auto_execution
        
        
        
        self.l515 = RealCamera(self.robot_name+'_l515', roi=self.l515_roi, jetson=True)
        self.d435_l = RealCamera(self.robot_name+'_d435_l')
        self.d435_r = RealCamera(self.robot_name+'_d435_r')

        self.dloPerciever = self.RopePerceiver(self.l515)
        
        self.rope_config = {
            'rope_r': self.Rope(self.l515, 'rope_r', self.auto_execution, 'blue', 'red_1'),
            'rope_g': self.Rope(self.l515, 'rope_g', self.auto_execution, 'blue', 'green'),
            'rope_b': self.Rope(self.l515, 'rope_b', self.auto_execution, 'blue', 'blue'),
        }
        
        self.ropes = []
        
        # initialise the tf listener
        self.listener = TransformListener()
        
        # define the services
        rospy.Service(self.find_target_srv, DetectTarget, self.srvPubScene)
        rospy.loginfo('Find targets service ready')
        rospy.Service(self.find_object_srv, DetectObject, self.srvPubScene)
        rospy.loginfo('Find objects service ready')
        
        # define the publishers
        # Crrently Unused
        self.frame_pub = rospy.Publisher(self.processed_frame_topic, Image, queue_size=10)
        if self.debug:
            for rope_name, _ in self.rope_config.items():       
                self.edge_pub = rospy.Publisher(f'dlo_vsn/{rope_name}/edge', Image, queue_size=10)
                self.mask_pub = rospy.Publisher(f'dlo_vsn/{rope_name}/masks', Image, queue_size=10)
                self.marker_pub_a = rospy.Publisher(f"dlo_vsn/{rope_name}/marker_a", PoseArray, queue_size=1)
                self.marker_pub_b = rospy.Publisher(f"dlo_vsn/{rope_name}/marker_b", PoseArray, queue_size=1)
                self.target_pub = rospy.Publisher(f"dlo_vsn/{rope_name}/a", PoseArray, queue_size=1)
                self.target_pub_r = rospy.Publisher(f"dlo_vsn/{rope_name}/b", PoseArray, queue_size=1)


        if self.debug:
            rospy.sleep(1)
            self.find_targets = rospy.ServiceProxy(self.find_target_srv, DetectTarget)
            self.find_objects = rospy.ServiceProxy('find_objects', DetectObject)
            rospy.wait_for_service(self.find_target_srv)
            request = DetectTargetRequest()
            request.target_name = 'marker_b' # targets_b, targets_r, marker_a, marker_b
            request.camera_name = 'l515' # d435_l, d435_r, l515
            while not rospy.is_shutdown():
                start = time.perf_counter()
                response = self.find_targets(request)
                stop = time.perf_counter()
                print(stop-start)
                rospy.sleep(0.5)
    
    
    ''' Callback of find target service '''
    def srvPubScene(self, request):
        # generate an empty response
        response = DetectTargetResponse()
        # turn the img msg to numpy array
        if request.camera_name == 'l515':
            img = self.l515.read_image()
            depth = self.l515.read_depth()
            camera_intrinsics = self.l515.read_camera_info()
            (trans,quat) = self.listener.lookupTransform(self.robot_frame, self.l515.frame, rospy.Time(0))
            cam_to_rob = compose_matrix(angles=euler_from_quaternion(quat), translate=trans) # transform from camera to robot
        elif request.camera_name == 'd435_l':
            camera = self.d435_l
            seg = self.seg_target_r
        elif request.camera_name == 'd435_r':
            camera = self.d435_r
            seg = self.seg_target_l
        else:
            rospy.logerr("Unknown type of camera!")
            return response

        
        

          

        self.frame_pub.publish(msgify(Image, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'rgb8'))
        return response


    def pubMarkers(self, request):
        
            
            
        # start processing the request
        if request.target_name == "marker_a":
            # localise the two markers from the masks
            pose_list = []
            img_list = []
            for _ in range(5):
                img = self.l515.read_image()
                depth = self.l515.read_depth()
                # get masks for two markers
                mask = self.seg_marker_a.predict_img(img)
                pose, img = self.marker_detection(img, mask, depth, cam_to_rob, camera_intrinsics)
                if pose is not None:
                    pose_list.append(pose)
                    img_list.append(img)
            if len(pose_list)>0:
                pose_list = np.array(pose_list)
                id = pose_list[:,2].argsort()[pose_list.shape[0]//2]
                pose = pose_list[id]
                img = img_list[id]

                # generate the response message
                target = PoseArray()
                target.header.stamp = rospy.Time.now()
                target.header.frame_id = self.robot_frame
                marker_pose = Pose()
                marker_pose.position = Point(*pose[:3])
                marker_pose.orientation = Quaternion(*pose[3:])
                target.poses.append(marker_pose)
                response.target = target

                if self.debug:
                    self.marker_pub_a.publish(target) 

        if request.target_name == "marker_b":
            # localise the two markers from the masks
            pose_list = []
            img_list = []
            for _ in range(5):
                img = self.l515.read_image()
                depth = self.l515.read_depth()
                # get masks for two markers
                mask = self.seg_marker_b.predict_img(img)
                pose, img = self.marker_detection(img, mask, depth, cam_to_rob, camera_intrinsics)
                if pose is not None:
                    pose_list.append(pose)
                    img_list.append(img)

            if len(pose_list)>0:
                pose_list = np.array(pose_list)
                id = pose_list[:,2].argsort()[pose_list.shape[0]//2]
                pose = pose_list[id]
                img = img_list[id]

                # generate the response message
                target = PoseArray()
                target.header.stamp = rospy.Time.now()
                target.header.frame_id = self.robot_frame
                marker_pose = Pose()
                marker_pose.position = Point(*pose[:3])
                marker_pose.orientation = Quaternion(*pose[3:])
                target.poses.append(marker_pose)
                response.target = target

                if self.debug:
                    self.marker_pub_b.publish(target)
                    
    def pubTargets(self, request):
        if request.target_name == "targets_r":
            targets, img, confidence = self.target_detection(camera, seg)
            # generate the target point messages
            target = PoseArray()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = self.robot_frame
            for target in targets:
                target_pose = Pose()
                target_pose.position = Point(*target[:3])
                normal = target[3:]
                if any(np.isnan(normal)):
                    quaternion = [0,0,0,1]
                else:
                    if normal[1]<0: normal = [ -v for v in normal] # force to face the shoe centre

                    roll = 0 # marker has 2 degree of rotational freedom
                    yaw = atan2(normal[1], normal[0]) 
                    # if yaw<0: yaw += pi # only defined to the left side 
                    pitch = -atan2(normal[2], sqrt(normal[0]**2+normal[1]**2))
                    # pitch = 0 # set pitch to 0 to avoid bumping into the sheo tongue during pulling after the insertion
                    # quaternion = [roll, pitch, yaw, 0]
                    quaternion = quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
                target_pose.orientation = Quaternion(*quaternion)
                target.poses.append(target_pose)
            response.target = target

            conf_msg = Float64MultiArray()
            conf_msg.data = confidence
            response.confidence = conf_msg

            if self.debug:
                self.target_pub_r.publish(target)   

        if request.target_name == "targets_b":
            targets, img, confidence = self.target_detection(camera, seg)
            # generate the target point messages
            target = PoseArray()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = self.robot_frame
            for target in targets:
                target_pose = Pose()
                target_pose.position = Point(*target[:3])
                normal = target[3:]
                if any(np.isnan(normal)):
                    quaternion = [0,0,0,1]
                else:
                    if normal[1]>0: normal = [ -v for v in normal]

                    roll = 0 # marker has 2 degree of rotational freedom
                    yaw = atan2(normal[1], normal[0]) 
                    # if yaw>0: yaw -= pi # only defined to the left side 
                    pitch = atan2(-normal[2], sqrt(normal[0]**2+normal[1]**2))
                    # pitch = 0 # set pitch to 0 to avoid bumping into the shoe tongue during pulling after the insertion
                    quaternion = quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
                    # quaternion = [roll, pitch, yaw, 0]
                target_pose.orientation = Quaternion(*quaternion)
                target.poses.append(target_pose)
            response.target = target

            conf_msg = Float64MultiArray()
            conf_msg.data = confidence
            response.confidence = conf_msg

            if self.debug:
                self.target_pub.publish(target) 
                     
    def ropeDetection(self, request):
        
        return None
    
    def marker_detection(self, img, mask, depth, transform, camera_intrinsics):
        '''
        [Main recognition function to detect the markers]
        Input: image
        output: pose of the marker (7), result image
        '''
        # remove the shoe region
        _, width = np.shape(mask)
        mask[:, width//5*2:width//5*3] = 0
        # mask[530:600, 405:870] = 0 # remove the central region

        # smooth the mask
        # mask = cv2.GaussianBlur(mask,(5,5),0)
        
        if self.debug:
            self.mask_pub.publish(msgify(Image, mask, 'mono8'))

        # hierarchy: [Next, Previous, First_Child, Parent]
        contours_temp, hierachy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours_temp is None or hierachy is None:
            return None, img
        
        # filter out all the non first layer contours
        contours = []
        for id, h in enumerate(hierachy[0]):
            if h[3] == -1:
                contours.append(contours_temp[id])
        # img = cv2.drawContours(img, contours, -1, (255,255,0), 3)

        # filter out the big contours
        boxes = []
        sizes = []
        for i in range(len(contours)):
            box = cv2.minAreaRect(contours[i]) #(center(x, y), (width, height), angle of rotation)
            size = box[1][0]*box[1][1]
            if size>=1000: # ignore the big contours
                continue
            if size<=5: # ignore the small contours
                continue
            boxes.append(box)
            sizes.append(size)
            
        # find the two boxes whose distance fits requirement
        dist_mat = np.eye(len(boxes))*1000
        candidates = []
        for i in range(len(boxes)):
            for j in range(i):
                distance = euclidian_distance(boxes[i][0], boxes[j][0])
                dist_mat[i,j] = dist_mat[j,i] = distance
                if 10<=distance<=70:
                    dist_mat[i,j] = dist_mat[j,i] = 1000 # not needed anymore
                    candidates.append([i, j])
        ids = None
        for c in candidates:
            if all(dist_mat[c[0], :]>20) and all(dist_mat[c[1], :]>20):
                ids = [c[0], c[1]] if sizes[c[0]]<sizes[c[1]] else [c[1],c[0]]
                break

        # generate the output
        if ids is not None:
            arrow_pt = np.array(self.list_add(boxes[ids[0]][0], self.l515_roi[:2]))
            arrow_pt_3d = read_point_from_region(xy_to_yx(arrow_pt), depth, region=3, camera_intrinsics=camera_intrinsics)
            arrow_pt_3d = self.transform_point(arrow_pt_3d, transform)

            # anchor_pt = self.list_add(boxes[ids[1]][0], self.l515_roi[:2])
            anchor_box = boxes[ids[1]]
            anchor_half_length = max(anchor_box[1])/2
            anchor_pt = np.array(anchor_box[0])
            anchor_pt = self.list_add(anchor_pt, self.l515_roi[:2])
            aa_distance = euclidian_distance(anchor_pt, arrow_pt)
            anchor_pt = anchor_pt+(anchor_pt-arrow_pt)*anchor_half_length/aa_distance/4*3
            anchor_pt_3d = read_point_from_region(xy_to_yx(anchor_pt), depth, region=3, camera_intrinsics=camera_intrinsics)
            anchor_pt_3d = self.transform_point(anchor_pt_3d, transform)

            yaw = atan2(arrow_pt_3d[1]-anchor_pt_3d[1],arrow_pt_3d[0]-anchor_pt_3d[0])
            pose = np.concatenate([anchor_pt_3d, quaternion_from_euler(0, 0, yaw)])

            img = self.generate_frame(img, [boxes[ids[0]], boxes[ids[1]], (tuple(anchor_pt-self.l515_roi[:2]), min(anchor_box[1]))])
            return pose, img
        else:
            return None, img

    def target_detection(self, camera, seg):
        '''
        [Main recognition function to detect the targets]
        Input: image
        output: list of poses (n*7, n for number of targets), result image
        camera is RealCamera instance
        seg is ColourSegmentation instance
        '''

        img = camera.read_image()
        depth = camera.read_depth()
        camera_intrinsics = camera.read_camera_info()
        # if True:
        #     pose = camera.read_camera_pose()
        #     (trans,quat) = pose[:3], pose[3:]
        # else:
        (trans,quat) = self.listener.lookupTransform(self.robot_frame, camera.frame, rospy.Time(0))
        transform = compose_matrix(angles=euler_from_quaternion(quat), translate=trans)
        
        mask = seg.predict_img(img)

        # cap with depth
        mask = mask*(depth<400) # remove everything beyond 0.5m
        
        if self.debug:
            self.mask_pub.publish(msgify(Image, mask, 'mono8'))
        
        h,w = mask.shape
        mask[int(350/480*h):, int(620/848*w):int(700/848*w)] = 0 # remove markers

        # smooth the mask
        # mask = cv2.GaussianBlur(mask,(5,5),0)

        # hierarchy: [Next, Previous, First_Child, Parent]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or hierarchy is None:
            return [], img, []

        hierarchy = hierarchy[0]
        # approximate the contours with circles
        circles = []
        outer_circles = []
        outer_contours = []
        inner_circle_boxes = []
        outer_circle_boxes = []
        for i in range(len(contours)):
            if hierarchy[i,3] == -1:
                continue
            inner_circle = cv2.minEnclosingCircle(contours[i]) # (x,y),radius inner_circle
            outer_circle = cv2.minEnclosingCircle(contours[hierarchy[i, 3]]) # (x,y),radius outer_circle
            if outer_circle[1] >= 30 or inner_circle[1] <= 5: # remove unreasonable circles
                continue
            circles.append(inner_circle)
            outer_circles.append(outer_circle)
            outer_contours.append(contours[hierarchy[i, 3]])
            # radius = inner_circle[1]*2
            radius = (outer_circle[1]/sqrt(2)+inner_circle[1])
            inner_circle_box = (inner_circle[0], (radius, radius), 0) #(center(x, y), (width, height), angle of rotation)
            outer_circle_box = (outer_circle[0], (outer_circle[1]/sqrt(2), outer_circle[1]/sqrt(2)), 0) #(center(x, y), (width, height), angle of rotation)
            outer_circle_boxes.append(outer_circle_box)
            inner_circle_boxes.append(inner_circle_box)
        n_circles = len(circles)
        n_obs = 5
            
        # get poses of the circles
        poses_observations = []
        circles_points_observations = []
        for _ in range(n_circles):
            circles_points_observations.append([])
        for i in range(n_obs):
            depth = camera.read_depth()
            poses = []
            for id in range(n_circles):
                inner_circle_box = cv2.boxPoints(inner_circle_boxes[id])
                inner_circle_box = np.int0(inner_circle_box)

                # get the centre of the boxes
                box_3d = read_points_from_region(xy_to_yx(inner_circle_box), depth, region=5, camera_intrinsics=camera_intrinsics)
                # position_cam = np.mean(box_3d, axis=0)
                box_3d = [self.transform_point(v, transform) for v in box_3d]
                position = np.mean(box_3d, axis=0)

                # get all points on the outer contour
                for p in outer_contours[id]:
                    p_3d = depth_pixel_to_metric_coordinate(xy_to_yx(p[0]), depth, camera_intrinsics)
                    if not isnan(p_3d[0]):
                        p_3d = self.transform_point(p_3d, transform)
                        circles_points_observations[id].append(p_3d)

                poses.append(np.concatenate((position, [0]*3)))

            poses_observations.append(poses)
        poses = np.mean(poses_observations, axis=0) # average out

        for id in range(n_circles):
            open3d_cloud = o3d.geometry.PointCloud()
            open3d_cloud.points = o3d.utility.Vector3dVector(circles_points_observations[id])
            plane_model, inliers = open3d_cloud.segment_plane(distance_threshold=0.005,
                                                    ransac_n=3,
                                                    num_iterations=1500)
            normal = plane_model[:3]/np.linalg.norm(plane_model[:3]) # renormalise
            if camera == self.d435_l:
                if normal[1]>0: normal = [ -v for v in normal] # force to face the shoe centre
            else:
                if normal[1]<0: normal = [ -v for v in normal] # force to face the shoe centre
            poses[id, 3:] = normal
        
        confidences = []
        for pose in poses:
            # normal = pose[3:]/np.linalg.norm(pose[3:]) # renormalise
            normal = pose[3:]
            # compute confidence
            ae_vec = np.subtract(pose[:3], trans)
            # if normal[1]>0: normal = [ -v for v in normal] # force to face the shoe centre
            c = np.dot(ae_vec,normal)/np.linalg.norm(ae_vec)/np.linalg.norm(normal) # -> cosine of the angle
            angle = np.arccos(np.clip(c, -1, 1)) # the angle
            confidence = 1-angle/pi
            confidences.append(confidence if not np.isnan(confidence) else 0)

        result_img = self.generate_frame(img, outer_circles+circles, list(confidences)*2)
        result_img = self.generate_frame(result_img, inner_circle_boxes, confidences)
        return poses, result_img, confidences

    
    @staticmethod
    def transform_point(point, transfer):
        return np.dot(transfer, np.array([point[0], point[1], point[2], 1.0]))[:3]

    @staticmethod
    def list_add(list_a,list_b):
        return [a+b for a,b in zip(list_a, list_b)]

    def generate_frame(self, img, shapes, confidence=None):
        '''
        [Generate the new frame for display]
        Input: image, shapes [(centre, radius) or (x, y, w, h) (numpy)]
        output: resulting image
        '''
        if confidence is None or np.any(np.isnan(confidence)):
            confidence = len(shapes)*[1]
        confidence = [max((c-0.7)/0.3, 0) for c in confidence]
        for id, shape in enumerate(shapes):
            color = int(255*confidence[id])
            if len(shape)==2: # if the shape is a circle
                img = cv2.circle(img, tuple(np.int0(shape[0])), int(shape[1]), (0,color,0), 2)
                # if id < len(shapes)//2:
                #     cv2.putText(img, str(id), tuple(np.int0(shape[0])), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,color,0), 2, 8)
            elif len(shape)==3: # if the shape is a box
                box = cv2.boxPoints(shape)
                box = np.int0(box)
                img = cv2.drawContours(img,[box],0,(0,0,color),2)            
        return img

if __name__ == "__main__":
    dlo_vsn_node = dloVision()
