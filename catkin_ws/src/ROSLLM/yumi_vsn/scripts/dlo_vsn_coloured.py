#!/usr/bin/python3
import os, sys
import cv2
from cv_bridge import CvBridge
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
from rosllm_srvs.srv import DetectRope, DetectRopeRequest, DetectRopeResponse
from rosllm_srvs.srv import ObserveScene, ObserveSceneRequest, ObserveSceneResponse
from utils.yumi_camera import RealCamera
from utils.point_clouds import depth_pixel_to_metric_coordinate, read_point_from_region, read_points_from_region, xy_to_yx, euclidian_distance
from utils.colour_segmentation import ColourSegmentation
from platform_registration import PlatformRegistration


MODEL_DIR = os.path.expanduser(rospy.get_param("dlo_perceiver", '/catkin_ws/src/ROSLLM/yumi_vsn/scripts/utils/dlo_perceiver/'))  # <-- update this
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)
MODEL_PATH = os.path.join(MODEL_DIR, "dlo_perceiver.pt")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, "../results")
TEST_DIR = os.path.join(MODEL_DIR, 'images')
from utils.dlo_perceiver.dlo_perceiver.model_contrastive import DLOPerceiver
from utils.dlo_perceiver.dlo_perceiver.text_encoder import TextEncoder
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



RGB_THRESHOLDS = {
    'purple':   {'lower': rospy.get_param("~purple_lower", [203, 134, 142]),   'upper':    rospy.get_param("~purple_upper", [255, 198, 255])},
    'magenta':  {'lower': rospy.get_param("~magenta_lower", [124, 64, 124] ),     'upper':    rospy.get_param("~magenta_upper", [196, 153, 255])},
    'red':      {'lower': rospy.get_param("~red_lower", [0, 77, 221]),      'upper':    rospy.get_param("~red_upper", [130, 147, 255])},
    'pink':     {'lower': rospy.get_param("~pink_lower", [181, 194, 234]),    'upper':    rospy.get_param("~pink_upper", [244, 252, 255])},
    'cyan':     {'lower': rospy.get_param("~cyan_lower", [203, 0, 0]),    'upper':    rospy.get_param("~cyan_upper", [255, 211, 19])},
    'grey':     {'lower': rospy.get_param("~grey_lower", [177, 191, 118]),   'upper':    rospy.get_param("~grey_upper", [237, 250, 228])},
    'yellow':   {'lower': rospy.get_param("~yellow_lower", [78, 173, 220]),    'upper':    rospy.get_param("~yellow_upper", [197, 255, 249])}
}

class RopePerceiver:
        def __init__(self):
            """
            Initialise model, tokenizer, and colour/view prompts.
            """
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            rospy.loginfo(f"Using device: {self.device}")
            self.model_dict = torch.load(MODEL_PATH)
            rospy.loginfo(f"Loaded model from {MODEL_PATH}")
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
            self.model.eval()  # Set model to evaluation mode
            total_val_loss = 0
            self.colours = {"orange": "rope_o",
                            "green": "rope_g",
                            "blue": "rope_b"}
            self.views = ["top", "bottom"]


        def _prepareImage(self, img) :
            """
            Resize and normalise image for model input.
            """

            
            img = cv2.resize(img, (self.model_config["img_w"], self.model_config["img_h"]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = img.transpose(2, 0, 1)
            img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).to(device=self.device)
            
            return img

        def _predictMask(self, imgTensor, textEmb):
            """
            Predict binary mask from image and text embedding.
            """
            with torch.no_grad():
                pred, (_, _, _) = self.model(imgTensor, textEmb)
            mask = pred.sigmoid().squeeze().detach().cpu().numpy()
            return mask


        def _segmentRope(self, imgTensor, colour):
            """
            Segment a rope by given colour and view. Returns binary mask.
            """
            
            x = f"rope,{colour},top"
            x = x.split(",")
            rospy.loginfo(f"Segmenting rope with prompt: {x}")
            prompt = {"object": x[0], "color": x[1], "top_down": x[2]}
            text = f"{prompt['object']} {prompt['color']} {prompt['top_down']}"
            texts = [text]
            tokens = self.tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
            textEmb = self.textEncoder(tokens).to(self.device)       
            mask = self._predictMask(imgTensor, textEmb)
            mask = ((mask * 255)).astype("uint8") 
            
            return mask  # Convert to binary mask

        def _rankRopesByHierarchy(self, ropeList):
            """
            Stub for hierarchical ranking. Compares overlap between masks.
            """
            # Example: sort ropes by average pixel intensity
            # You can improve this using occlusion detection logic later
            rope_visibility = []
            for rope in ropeList:
                if "score" in rope:
                    rope_visibility.append((rope["score"], rope))

            # Sort by descending area → more visible on top
            rope_visibility.sort(reverse=True, key=lambda x: x[0])

            return [r for _, r in rope_visibility]
        def _cleanMask(self, uniqueRopes):
            """
            Clean masks by removing duplicates based on IoU.
            """
            binary_masks = [((r["mask"] > 127).astype(np.uint8)) for r in uniqueRopes]
            # Step 2: Sum all masks to find overlaps
            sum_mask = np.sum(binary_masks, axis=0)

            # Step 3: Build exclusive mask by subtracting overlapping pixels
            cleaned_masks = []
            for i, m in enumerate(uniqueRopes):
                binary = binary_masks[i]
                # Keep only pixels where this mask is on and no other
                exclusive = np.logical_and(binary == 1, sum_mask == 1).astype(np.uint8)
                cleaned = (exclusive * 255).astype(np.uint8)
                cleaned_masks.append({
                    "colour": m["colour"],
                    "mask": cleaned,
                    "score": cleaned.sum()
                })

            return cleaned_masks
        def countRopes(self, frame) -> list:
            """
            Detect and count unique ropes using mask similarity.
            Returns list of unique rope dicts with colour, view, prompt, and mask.
            """
            rospy.loginfo("Counting ropes in the frame")
            imgJpg = self._prepareImage(frame)
            rospy.loginfo("Image prepared for model input")
            uniqueRopes = []
            existingMasks = []
            ropeList = []
            for colour in self.colours.keys():
                rospy.loginfo(f"Segmenting rope of colour: {colour}")
                mask = self._segmentRope(imgJpg, colour)
                if mask.max() < 0.1:
                    continue  # No activation
                # if not self._isDuplicate(mask, existingMasks):
                #     existingMasks.append(mask)
                uniqueRopes.append({
                    "colour": colour,
                    "mask": mask,
                    "score": mask.sum()
                })
                
            # uniqueRopes = self._cleanMask(uniqueRopes)
            return self._rankRopesByHierarchy(uniqueRopes)
        
class Rope:
        def __init__(self, cam_img, rope_name, auto_execution, marker_a_colour, marker_b_colour):
            self.cam_img = cam_img
            self.name = rope_name
            self.marker_a_colour = marker_a_colour
            self.marker_b_colour = marker_b_colour
            self.auto_execution = auto_execution
            self.threshold_a, self.threshold_b = self.updateThreshold(marker_a_colour, marker_b_colour)
            self.priority = 0
            self.mask = None
            self.target_l_colour = None
            self.target_r_colour = None
            self.curr_target_l_colour = None       
            # self.threshold_l, self.threshold_r = self.updateThreshold(self.target_l_colour, self.target_r_colour)
                      
        def updateCurrMarker(self, curr_target_l_colour, curr_target_r_colour):
            self.curr_target_l_colour = curr_target_l_colour
            self.curr_target_r_colour = curr_target_r_colour
            self.threshold_curr_l, self.threshold_curr_r = self.updateThreshold(curr_target_l_colour, curr_target_r_colour)
            
        def updateThreshold(self, colour_1, colour_2):
            thresh_low_1 = RGB_THRESHOLDS[colour_1]['lower']
            thresh_high_1 = RGB_THRESHOLDS[colour_1]['upper']
            thresh_low_2 = RGB_THRESHOLDS[colour_2]['lower']
            thresh_high_2 = RGB_THRESHOLDS[colour_2]['upper']
            rospy.loginfo(f"Updating thresholds for {self.name} with colours {colour_1}")
            marker_a= ColourSegmentation(thresh_low_1, thresh_high_1, self.cam_img, live_adjust=not self.auto_execution) 
            rospy.loginfo(f"Marker A thresholds: {thresh_low_1}, {thresh_high_1}")
            rospy.loginfo(f"Updating thresholds for {self.name} with colours {colour_2}")
            marker_b = ColourSegmentation(thresh_low_2, thresh_high_2, self.cam_img, live_adjust=not self.auto_execution)
            rospy.loginfo(f"Marker B thresholds: {thresh_low_2}, {thresh_high_2}")
            return marker_a, marker_b
        
class dloVision:
    debug = True
    debug_name = None
    processed_frame_topic = '/dlo_vsn/frame_processed'
    scene_pub_topic = '/dlo_vsn/scene'
    observe_scene_srv = 'observe_scene'
    find_rope_srv = 'detect_ropes'
    robot_frame = "yumi_base_link"
    l515_roi = [0, 180, 1280, 720] # 128[0, 0, 125], [110, 110, 0]0*720
    
    
        
    def __init__(self, auto_execution):
        
        # read ROS parameters
        self.robot_name = rospy.get_param("~robot_name", "yumi")
        self.bridge = CvBridge()
        self.auto_execution = auto_execution
        
        
        
        self.l515 = RealCamera(self.robot_name+'_l515', roi=self.l515_roi, jetson=False)
        rospy.loginfo(f"Using camera {self.l515.frame} for rope perception")
        # self.d435_l = RealCamera(self.robot_name+'_d435_l')
        # self.d435_r = RealCamera(self.robot_name+'_d435_r')

        self.dloPerciever = RopePerceiver()
        rospy.loginfo("Loaded DLO Perceiver model and tokenizer")
        
        self.rope_config = {
            'rope_o': Rope(self.l515.read_image, 'rope_o', self.auto_execution, 'grey', 'cyan'),
            'rope_g': Rope(self.l515.read_image, 'rope_g', self.auto_execution, 'purple', 'red'),
            'rope_b': Rope(self.l515.read_image, 'rope_b', self.auto_execution, 'pink', 'yellow')
        }

        
        # initialise the tf listener
        self.listener = TransformListener()
        
        # define the services
        rospy.Service(self.observe_scene_srv, ObserveScene, self.srvPubScene)
        rospy.loginfo('Find Scene service ready')
        rospy.Service(self.find_rope_srv, DetectRope, self.pubMarkers)
        rospy.loginfo('Find objects service ready')
       
        
        # define the publishers
        # Crrently Unused
        self.frame_pub = rospy.Publisher(self.processed_frame_topic, Image, queue_size=10)
        if self.debug:
            for rope_name, _ in self.rope_config.items():       
                self.edge_pub = rospy.Publisher(f'dlo_vsn/{rope_name}/edge', Image, queue_size=10)
                self.mask_pub = rospy.Publisher(f'dlo_vsn/{rope_name}/masks', Image, queue_size=10)
                self.rope_o_pub = rospy.Publisher(f"dlo_vsn/{rope_name}", Image, queue_size=10)
                self.rope_g_pub = rospy.Publisher(f"dlo_vsn/{rope_name}", Image, queue_size=10)
                self.rope_b_pub = rospy.Publisher(f"dlo_vsn/{rope_name}", Image, queue_size=10)
                self.marker_pub_a = rospy.Publisher(f"dlo_vsn/{rope_name}/marker_a", PoseArray, queue_size=1)
                self.marker_pub_b = rospy.Publisher(f"dlo_vsn/{rope_name}/marker_b", PoseArray, queue_size=1)
                self.target_pub = rospy.Publisher(f"dlo_vsn/{rope_name}/a", PoseArray, queue_size=1)
                self.target_pub_r = rospy.Publisher(f"dlo_vsn/{rope_name}/b", PoseArray, queue_size=1)
                

        if self.debug:
            rospy.sleep(1)
            # self.find_targets = rospy.ServiceProxy(self.find_target_srv, DetectTarget)
            self.find_rope = rospy.ServiceProxy(self.find_rope_srv, DetectRope)
            self.observe_scene = rospy.ServiceProxy(self.observe_scene_srv, ObserveScene)
            if self.debug_name == 'marker':
                rospy.wait_for_service(self.find_rope_srv)
                request = DetectRopeRequest()
                request.rope = 'rope_o' # rope_o, rope_g, rope_b
                request.target_name = 'marker_a' # targets_b, targets_r, marker_a, marker_b
                request.camera_name = 'l515' # d435_l, d435_r, l515
                while not rospy.is_shutdown():
                    start = time.perf_counter()
                    response = self.find_rope(request)
                    stop = time.perf_counter()
                    print(stop-start)
                    rospy.sleep(0.5)
            else:
                rospy.loginfo("Starting to observe scene")
                rospy.wait_for_service(self.observe_scene_srv)
                rospy.loginfo("Observe scene service is ready")
                request = ObserveSceneRequest()
                rospy.loginfo("Requesting scene observation")
                start = time.perf_counter()
                rospy.loginfo("Calling observe_scene service")
                response = self.observe_scene(request)
                rospy.loginfo("Received response from observe_scene service")
                while response.success is False and not rospy.is_shutdown():
                    rospy.loginfo(f"Detected ropes: {response.ropes}")
                    rospy.loginfo(f"Detected centre: {response.centre}")
                    stop = time.perf_counter()
                    rospy.loginfo(f"Time Taken: {stop-start}")
                    rospy.sleep(0.5)
            
                
            
    
    
    ''' Callback of find overall scene service '''
    def srvPubScene(self, request):
        # generate an empty response
        response = ObserveSceneResponse()
        # turn the img msg to numpy array
        rospy.loginfo("Observing scene")
        l515_img = self.l515.read_image()
        depth = self.l515.read_depth()
        camera_intrinsics = self.l515.read_camera_info()
        (trans, quat) = self.listener.lookupTransform(self.robot_frame, self.l515.frame, rospy.Time(0))
        cam_to_rob = compose_matrix(angles=euler_from_quaternion(quat), translate=trans)
        
        
      
        rospy.loginfo("Counting ropes in the scene")
        detected_ropes = self.dloPerciever.countRopes(l515_img)
        rgb_img = cv2.cvtColor(l515_img, cv2.COLOR_BGR2RGB)
        raw_img_out_path = os.path.join(RESULT_PATH, "img_raw.png")
        raw_img_bgr_path = os.path.join(RESULT_PATH, "img_raw_bgr.png")
        cv2.imwrite(raw_img_out_path, rgb_img)
        cv2.imwrite(raw_img_bgr_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        rope_list = []
        for rope in detected_ropes:
            colour = rope["colour"]
            ropeName = self.dloPerciever.colours[colour]
            if ropeName not in self.rope_config:
                rospy.logerr(f"Unknown rope name: {ropeName}")
                continue
            rope_list.append(ropeName)
            rospy.loginfo(f"Detected rope: {ropeName}, Score: {rope['score']}")
            ropeObj = self.rope_config[ropeName]
            ropeObj.mask = rope["mask"]
            
            if colour == 'orange':
                rospy.loginfo("Saving mask for orange rope")
                cv2.imwrite(os.path.join(RESULT_PATH, "rope_o_mask.png"), rope['mask'])
            elif colour == 'green':
                rospy.loginfo("Saving mask for green rope")
                cv2.imwrite(os.path.join(RESULT_PATH, "rope_g_mask.png"), rope['mask'])
            else:
                rospy.loginfo("Saving mask for blue rope")
                cv2.imwrite(os.path.join(RESULT_PATH, "rope_b_mask.png"), rope['mask'])
            
        response.centre, aruco_img, new_img = self.scene_detection(l515_img, depth, cam_to_rob, camera_intrinsics)
            
        cv2.imwrite(os.path.join(RESULT_PATH, "aruco_img_new.png"), new_img)
        cv2.imwrite(os.path.join(RESULT_PATH, "aruco_img.png"), aruco_img)
        response.img = self.bridge.cv2_to_imgmsg(cv2.cvtColor(l515_img, cv2.COLOR_RGB2BGR), encoding="rgb8")
        response.success = True
        response.ropes = rope_list 
        
        cv2.imwrite(os.path.join(RESULT_PATH, "aruco_img.png"), aruco_img)
        rospy.loginfo(f"Detected ropes Heirarchy: {rope_list}")
        
        
        self.frame_pub.publish(msgify(Image, cv2.cvtColor(l515_img, cv2.COLOR_BGR2RGB), 'rgb8'))
        rospy.loginfo("Published processed frame")
        return response

    def scene_detection(self, img, depth, transform, camera_intrinsics):
        ''' Fake scene detection with Aruco markers '''
        pr = PlatformRegistration()
        target = None
        while target is None and not rospy.is_shutdown():
            makers_3d = pr.register_platform(img, depth, camera_intrinsics, offset=self.l515_roi[:2])
            if makers_3d is not None:
                if len(makers_3d) == 4:
                    makers_3d = [self.transform_point(m, transform) for m in makers_3d]
                    # estimate the centre
                    centre = np.mean(makers_3d, axis=0)
                    # estimate the orientation
                    top_centre = np.mean(makers_3d[2:], axis=0)
                    bot_centre = np.mean(makers_3d[:2], axis=0)
                    top_centre_prime = top_centre-bot_centre
                    yaw = atan2(top_centre_prime[1], top_centre_prime[0])
                    euler = [0, 0, yaw]
                    target = np.concatenate((centre, quaternion_from_euler(*euler)))
                    break
                else:
                    rospy.logwarn("Not enough Aruco markers detected! Expected 4, got {}".format(len(makers_3d)))
            else:
                rospy.logwarn("No Aruco markers detected!")
            rospy.sleep(1)  
            img = self.l515.read_image()
            depth = self.l515.read_depth()
            rospy.sleep(0.5)
        # draw the markers
        img = pr.check_markers(img, target)
       
        # new_img = pr.new_check_markers(img, target, camera_intrinsics)
        rospy.loginfo("Centre of the platform: {}".format(target))
        return target, img

    def pubMarkers(self, request):
        """
        Localize the two markers (marker_a and marker_b) from the masks.
        """
        response = DetectRopeResponse()
        rope = self.rope_config[request.rope_colour]
        camera_intrinsics = self.l515.read_camera_info()
        (trans,quat) = self.listener.lookupTransform(self.robot_frame, self.l515.frame, rospy.Time(0))
        transform = compose_matrix(angles=euler_from_quaternion(quat), translate=trans) # transform from camera to robot
            
        pose_list = {"marker_a": [], "marker_b": []}
        img_list = {"marker_a": [], "marker_b": []}

        for _ in range(5):  # Perform multiple observations for robustness
            img = self.l515.read_image()
            depth = self.l515.read_depth()

            # Get masks for both markers
            mask_a = rope.threshold_a.predict_img(img)
            mask_b = rope.threshold_b.predict_img(img)

            # Detect poses for both markers
            pose_a, img_a = self.marker_detection(img, mask_a, depth, transform, camera_intrinsics)
            pose_b, img_b = self.marker_detection(img, mask_b, depth, transform, camera_intrinsics)

            if pose_a is not None:
                pose_list["marker_a"].append(pose_a)
                img_list["marker_a"].append(img_a)
            else:
                rospy.logwarn(f"{rope.name}: No pose detected for marker_a")
            if pose_b is not None:
                pose_list["marker_b"].append(pose_b)
                img_list["marker_b"].append(img_b)
            else:
                rospy.logwarn(f"{rope.name}: No pose detected for marker_b")

        # Process poses for marker_a and marker_b
        marker_poses = {}
        for marker in ["marker_a", "marker_b"]:
            if len(pose_list[marker]) > 0:
                poses = np.array(pose_list[marker])
                id = poses[:, 2].argsort()[poses.shape[0] // 2]  # Median pose based on Z-axis
                pose = poses[id]
                img = img_list[marker][id]

                # Generate the response message
                target = PoseArray()
                rospy.loginfo(f"MARKERS")
                target.header.stamp = rospy.Time.now()
                target.header.frame_id = self.robot_frame
                marker_pose = Pose()
                marker_pose.position = Point(*pose[:3])
                marker_pose.orientation = Quaternion(*pose[3:])
                target.poses.append(marker_pose)

                if self.debug:
                    if marker == "marker_a":
                        self.marker_pub_a.publish(target)
                    elif marker == "marker_b":
                        self.marker_pub_b.publish(target)
                marker_poses[marker] = target
                
        response.marker_a_pose = marker_poses["marker_a"]
        response.marker_b_pose = marker_poses["marker_b"]
        if marker_poses["marker_a"] is None:
            rospy.logerr("No marker_a detected!")
            response.success = False
        if marker_poses["marker_b"] is None:
            rospy.logerr("No marker_b detected!")
            response.success = False
        
        self.frame_pub.publish(msgify(Image, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'rgb8'))
        return response

                    
    def marker_detection(self, img, mask, depth, transform, camera_intrinsics):
        '''
        [Main recognition function to detect the markers]
        Input: image
        output: pose of the marker (7), result image
        '''
            
        # remove the shoe region
        # _, width = np.shape(mask)
        # mask[:, width//5*2:width//5*3] = 0
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
