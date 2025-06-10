import threading
import numpy as np
from os import path
from scipy.spatial import distance
from math import cos, sin, pi, sqrt

from scene_params import SceneParameters
from utils import list_to_pose_msg, ls_concat, ls_add, tf_ls2mat, tf_mat2ls, pose_msg_to_list, is_sorted
from yumi_wrapper import YumiWrapper
import tf
import rospy
import rospkg
from rosllm_srvs.srv import DetectRope, DetectRopeRequest, ObserveScene, ObserveSceneRequest, VLM, VLMResponse, ExecuteBehavior, ExecuteBehavior, ExecuteBehaviorResponse
from std_msgs.msg import String, Int32MultiArray
from geometry_msgs.msg import Pose, PoseArray
from tf.transformations import euler_from_quaternion, compose_matrix




class ScenePrimitives:
    package_name = 'yumi_ctrl'
    marker_topic = '/visualization_marker'
    # robot_frame = "yumi_base_link"
    rope_srv = 'detect_ropes'
    scene_srv = 'observe_scene'
    bt_srv = 'execute_behaviour'
    log_topic = 'scene_logs'
    rope_names_topic = ["rope_o", "rope_b", "rope_g"]
    marker_ids_topic = ['marker_a', 'marker_b']
    hand_pose_topic = 'hand_pose'
    target_pose_topic = 'target_pose'
    

    def __init__(self, auto_execution, reset=True, start_id=0):
        # ros related initialisation
        self.debug = True
        self.logs_pub = rospy.Publisher(self.log_topic, String, queue_size=1)
        
        # Initialise nested dictionary for rope → marker → topic publishers
        self.marker_owner_pubs = {}
        self.marker_pose_pubs = {}

        for rope in self.rope_names_topic:
            self.marker_owner_pubs[rope] = {}
            self.marker_pose_pubs[rope] = {}

            for marker in self.marker_ids_topic:
                base_topic = f"{rope}/{marker}"

                owner_topic = f"{base_topic}/marker_owner"
                pose_topic = f"{base_topic}/marker_pose"

                self.marker_owner_pubs[rope][marker] = rospy.Publisher(owner_topic, Int32MultiArray, queue_size=1)
                self.marker_pose_pubs[rope][marker] = rospy.Publisher(pose_topic, PoseArray, queue_size=1)
                
                
        
        self.target_pose_pub = rospy.Publisher(self.target_pose_topic, PoseArray, queue_size=1)
        self.hand_pose_pub = rospy.Publisher(self.hand_pose_topic, PoseArray, queue_size=1)
        # self.cursor_pub = rospy.Publisher(self.cursor_topic, Int32MultiArray, queue_size=1)
        self.tf_listener = tf.TransformListener()
        
        
        rospy.loginfo("Waiting for services...")
        self.find_rope = rospy.ServiceProxy(self.rope_srv, DetectRope)
        rospy.wait_for_service(self.rope_srv)

        self.observe_scene = rospy.ServiceProxy(self.scene_srv, ObserveScene)
        rospy.wait_for_service(self.scene_srv)
        rospy.Service(self.bt_srv, ExecuteBehavior, self.execute)
        rospy.loginfo("Service for executing behaviour is ready.")
        # load params
       
        self.pkg_path = rospkg.RosPack().get_path(self.package_name)
        rospy.loginfo("Setting Scene Parameters from: " + path.join(self.pkg_path, 'params'))
        self.pm = SceneParameters(reset=True, start_id=start_id,
                                           config_path=path.join(self.pkg_path, 'params'),
                                           result_path=path.join(self.pkg_path, 'results'),
                                           log_handle=self.add_to_log)
        # self.update_cursor('left', self.pm.left_cursor)
        # self.update_cursor('right', self.pm.right_cursor)


        self.yumi = YumiWrapper(auto_execution,
            workspace=self.pm.workspace,
            vel_scale=self.pm.vel_scale,
            gp_opening=self.pm.marker_thickness*1000*2, # to mm
            table_offset=self.pm.table_offset,
            grasp_states=self.pm.grasp_states,
            grasp_states2=self.pm.grasp_states2,
            observe_states=self.pm.observe_states)
        # scan the shoe
        self.pm.update_yumi_constriants = self.yumi.update_rope_constriants
        self.init_target_poses()
        
        rospy.loginfo('Execution module ready.')

        # create and start the tf listener thread
        # self.side_queue = []
        # self.side_thread = threading.Thread(target=self.tf_thread_func, daemon=True)
        # self.side_thread.start()

    def tf_thread_func(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                
                if self.get_marker_at('left_gripper') is not None:
                    (trans,rot) = self.tf_listener.lookupTransform(self.yumi.robot_frame, self.yumi.ee_frame_l, rospy.Time(0))
                    self.pub_hand_poses('left_gripper', trans+rot)
                    pose = tf_mat2ls(tf_ls2mat(trans+rot)@self.yumi.gripper_l_to_marker)
                    rope, marker = self.get_marker_at('left_gripper')
                    self.pub_marker_pose(marker, pose)
                elif self.get_marker_at('right_gripper') is not None:
                    (trans,rot) = self.tf_listener.lookupTransform(self.yumi.robot_frame, self.yumi.ee_frame_r, rospy.Time(0))
                    self.pub_hand_poses('right_gripper', trans+rot)
                    pose = tf_mat2ls(tf_ls2mat(trans+rot)@self.yumi.gripper_r_to_marker)
                    rope, marker = self.get_marker_at('right_gripper')
                    self.pub_marker_pose(rope, marker, pose)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            rate.sleep()                
                
                
    def execute(self, req):
        '''
        pick: gripper aglet site
        ''' 
        
        res = ExecuteBehaviorResponse()
        action = req.action
        rope = req.rope
        marker = req.marker
        site = req.site
        
        self.add_to_log(f'Executing action: {action} on {rope} {marker} to {site}')
         
        if action == 'left_place':
            if self.left_place(rope, marker, site) == True:
                res.success = True
                res.description = f'Left place of {marker} on {rope} to {site} successful.'
            else:
                res.success = False
                res.description = f'Left place of {marker} on {rope} to {site} failed.'
        elif action == 'right_place':
            if self.right_place(rope, marker, site) == True:
                res.success = True
                res.description = f'Right place of {marker} on {rope} to {site} successful.'
            else:
                res.success = False
                res.description = f'Right place of {marker} on {rope} to {site} failed.'
        else:
            self.add_to_log(f'Action {action} not recognised.')
            res.success = False
        return res
        
        
        

    def right_pick(self, rope, marker, fine_ori=False):
        """ pick up the marker with the right gripper """
        # calc pick poses
        if self.pm.check_marker_location(rope, marker)[:4] != 'site':
            if self.pm.check_marker_location(rope, marker)[:6] == 'target':
                self.right_remove(rope, marker)
                return
            else:
                self.add_to_log("Marker is not available for pick up!")
                return
            
        if self.pm.check_marker_location(rope, marker)[:6] != 'site_u':
            vert_offset = self.pm.table_offset + -0.0073
        else:
            vert_offset = self.pm.table_offset + 0.08
            
        marker_pos, yaw = self.get_rope_poses(rope, marker)
        pick_pos = [marker_pos[0], marker_pos[1], vert_offset+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os])
        pick_rot = self.pm.grasp_rot_r  
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])

        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.right_go_thro(waypoints,"Pick Approach")
        self.yumi.open_right_gripper()

        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Rotate")
        # pick the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick")
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = 'right_gripper'
        self.yumi.close_right_gripper()

        # self.update_marker_ownership(rope, marker, 'right_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos_approach, [0,0,self.pm.app_os]),pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Retract")
        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupancy(rope, marker, current_site, False)
        
        self.pm.update_site_occupancy(rope, marker, 'right_gripper')

        if fine_ori:
            self.right_refine_orientation()

    def left_pick(self, rope, marker, fine_ori=False):
        """ pick up the aglet with the left gripper """
        
        if self.pm.check_marker_location(rope, marker)[:4] != 'site':
            if self.pm.check_marker_location(rope, marker)[:6] == 'target':
                self.left_remove(rope, marker)
                return
            else:
                self.add_to_log("Marker is not available for pick up!")
                return
        
        if self.pm.check_marker_location(rope, marker)[:6] != 'site_u':
            vert_offset = self.pm.table_offset + -0.0073
        else:
            vert_offset = self.pm.table_offset + 0.08
            
        # calc pick poses
        marker_pos, yaw = self.get_rope_poses(rope, marker)
        pick_pos = [marker_pos[0], marker_pos[1], vert_offset+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os])
        pick_rot = self.pm.grasp_rot_l
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])
        
        self.yumi.remove_table()
        # approach the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.left_go_thro(waypoints,"Pick Approach")
        self.yumi.open_left_gripper()
        
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Rotate")
        # pick the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick")
        self.yumi.close_left_gripper()
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = 'left_gripper'
        # self.update_marker_ownership(rope, marker, 'left_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos_approach, [0,0,self.pm.app_os]),pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Retract")

        self.yumi.add_table()
        
        
        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupancy(rope, marker, current_site, False)
        
        self.pm.update_site_occupancy(rope, marker, 'left_gripper')

        if fine_ori:
            self.left_refine_orientation()
    def right_remove(self, rope, marker):
        
        
        _, _ = self.get_rope_poses(rope, marker)
        target = self.pm.check_marker_location(rope, marker)
        target_pos = self.pm.site_poses[target] # Choose target position
        marker_pos, [_,_,yaw] = target_pos[:3], euler_from_quaternion(target_pos[3:])
        
        pick_pos = [marker_pos[0], marker_pos[1], marker_pos[2]+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os]) # Slightly above the marker
        pick_rot = self.pm.grasp_rot_r
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])

        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.right_go_thro(waypoints,"Pick Approach")
        self.yumi.open_right_gripper()

        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Rotate")
        # pick the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick")
        self.yumi.close_right_gripper()
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = 'right_gripper'
        # self.update_marker_ownership(rope, marker, 'right_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos, [0, 2*self.pm.app_os, 0]),pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Retract")
        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos_approach, [0, 2*self.pm.app_os, self.pm.app_os]), pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Raise")
        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupancy(rope, marker, current_site, False)
        self.pm.update_site_occupancy(rope, marker, 'right_gripper')

    def left_remove(self, rope, marker):
         # calc pick poses
        _, _ = self.get_rope_poses(rope, marker) # Get updated positions
        target = self.pm.check_marker_location(rope, marker)
        target_pos = self.pm.site_poses[target] # Choose target position
        marker_pos, [_,_,yaw] = target_pos[:3], euler_from_quaternion(target_pos[3:])
        
        pick_pos = [marker_pos[0], marker_pos[1], marker_pos[2]+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os])
        pick_rot = self.pm.grasp_rot_l
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])
        
        self.yumi.remove_table()
        # approach the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.left_go_thro(waypoints,"Pick Approach")
        self.yumi.open_left_gripper()

        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Rotate")
        # pick the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick")
        self.yumi.close_left_gripper()
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = 'left_gripper'
        # self.update_marker_ownership(rope, marker, 'left_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos, [0, -2*self.pm.app_os, 0]),pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Retract")

        waypoints = []
        waypoints.append(ls_concat(ls_add(pick_pos_approach, [0, -2*self.pm.app_os, self.pm.app_os]), pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Raise")
        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupancy(rope, marker, current_site, False)
        
        self.pm.update_site_occupancy(rope, marker, 'left_gripper')
        
        
    def right_place(self, rope, marker, site='site_dr'):
        """ place down the aglet with the right gripper"""

        if site[:6] == 'target':
            return self.right_insert(rope, marker, site)
            
        # stretch the shoelace
        self.add_to_log("Placing to "+site)
        self.right_pick(rope, marker)
        if site == 'site_dr':
            section = self.pm.site_poses['site_dr']
        elif site == 'site_dd':
            section = self.pm.site_poses['site_dd']
        elif site == 'site_ur':
            section = self.pm.site_poses['site_ur']
        elif site == 'site_uu':
            section = self.pm.site_poses['site_uu']
        else:
            rospy.logerr("No Section is available for placing!")
            return False

        self.pm.update_site_occupancy(rope, marker, 'right_gripper', False)
       
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = site
        self.pm.update_site_occupancy(rope, marker, site)

        # calc place poses
        place_pos = [section[0], section[1], section[2]+self.pm.gp_os]
        place_pos_approach = ls_add(place_pos, [0, -self.pm.app_os, self.pm.app_os])
        [_,_,yaw] = euler_from_quaternion([0, 0, -0.6934, 0.7206])
        place_rot = ls_add(self.pm.grasp_rot_r, [0, 0, yaw])
        self.yumi.remove_table()
        # approach the aglet
        waypoints = []
        waypoints.append(place_pos_approach+place_rot)
        self.yumi.right_go_thro(waypoints,"Place Approach", eef_step=0.05)
        # place the aglet
        waypoints = []
        waypoints.append(place_pos+place_rot)
        self.yumi.right_go_thro(waypoints,"Place")
        self.yumi.open_right_gripper(full=True)
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = site
        # self.update_marker_ownership(rope, marker, site)
        # retreat after place
        waypoints = []
        waypoints.append(ls_add(place_pos_approach, [0,0,self.pm.app_os])+place_rot)
        self.yumi.right_go_thro(waypoints,"Place Retract")
        self.yumi.add_table()
        self.yumi.right_go_grasp()
        self.yumi.close_right_gripper()
        return True

    def left_place(self, rope, marker, site='site_dd'):
        """ place down the marker with the right gripper"""

        if site[:6] == 'target':
            return self.left_insert(rope, marker, site)
            
        # stretch the shoelace
        self.add_to_log("Placing to "+site)
        
        self.left_pick(rope, marker)
        
        if site == 'site_dl':
            section = self.pm.site_poses['site_dl']
            # self.left_stretch_backward(rope, marker)
        elif site == 'site_dd':
            section = self.pm.site_poses['site_dd']
            # self.left_stretch_backward(rope, marker)
        elif site == 'site_ul':
            section = self.pm.site_poses['site_ul']
            # self.left_stretch_forward(rope, marker)
        elif site == 'site_uu':
            section = self.pm.site_poses['site_uu']
            # self.left_stretch_forward(rope, marker)
        else:
            rospy.logerr("No Section is available for placing!")
            return False

        self.pm.update_site_occupancy(rope, marker, 'left_gripper', False)
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = site
        self.pm.update_site_occupancy(rope, marker, site)

        # calc place poses
        place_pos = [section[0], section[1], section[2]+self.pm.gp_os]
        place_pos_approach = ls_add(place_pos, [0, self.pm.app_os, self.pm.app_os])
        [_,_,yaw] = euler_from_quaternion([0, 0, 0.6934, 0.7205])
        place_rot = ls_add(self.pm.grasp_rot_l, [0, 0, yaw])
        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(place_pos_approach+place_rot)
        self.yumi.left_go_thro(waypoints,"Place Approach", eef_step=0.05)
        # place the marker
        waypoints = []
        waypoints.append(place_pos+place_rot)
        self.yumi.change_speed(0.5)
        self.yumi.left_go_thro(waypoints,"Place")
        self.yumi.open_left_gripper(full=True)
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = site
        # self.update_marker_ownership(rope, marker, site)
        # retreat after place
        waypoints = []
        waypoints.append(ls_add(place_pos_approach, [0,0,self.pm.app_os])+place_rot)
        self.yumi.left_go_thro(waypoints,"Place Retract")
        self.yumi.add_table()
        self.yumi.left_go_observe()
        self.yumi.close_left_gripper()
        return True

    def right_insert(self, rope, marker, target, reset=False):
        """
        This primitive inserts the marker (aglet) from inside the object toward the outside
        using only the right gripper.
        """
        

        # 2. Pick up the aglet
        self.right_pick(rope, marker, True)
        
        
        insert_pos = self.pm.site_poses[target][:3]
        [_,_,yaw] = euler_from_quaternion(self.pm.target_poses[target][3:])
        
        insert_app_pos = ls_add(insert_pos, [0, self.pm.app_os/2, 0])  # Slightly above the target
        insert_rot = self.pm.grasp_rot_r
        insert_rot_fine = ls_add(insert_rot, [0, 0, yaw])
        # 7. Compute poses (relative to hole frame, inserting from behind)
        # Note: we insert in the X direction, so we back off in -X to approach
        
        waypoints = []
        waypoints.append(ls_concat(insert_app_pos, insert_rot_fine))
        self.yumi.right_go_thro(waypoints, "Insert Approach", eef_step=0.05)
        
        waypoints = []
        waypoints.append(ls_concat(insert_pos, insert_rot_fine))
        self.yumi.right_go_thro(waypoints, "Insert Marker", eef_step=0.05, velocity_scaling=0.5)

        self.yumi.open_right_gripper()  # release the marker
        
        waypoints = []
        waypoints.append(ls_add(insert_app_pos, [0, self.pm.app_os, 2*self.pm.app_os]) + insert_rot_fine)
        self.yumi.right_go_thro(waypoints, "Insert Retract", eef_step=0.05)
        
        self.yumi.right_go_observe()
        self.yumi.close_right_gripper()  # reset gripper
        self.yumi.wait_for_side_thread()
        return True


    def left_insert(self, rope, marker, target, reset=False):
        """
        Insert aglet from inside to outside of an target using the left gripper only.
        Simplified version of left_lace with no grasp or pull, and one-arm control.
        """

        ''' 1. Pick the aglet with the left gripper '''
        
        self.left_pick(rope, marker, True)

        insert_pos = self.pm.site_poses[target][:3]
        [_,_,yaw] = euler_from_quaternion(self.pm.target_poses[target][3:])
        
        insert_app_pos = ls_add(insert_pos, [0, -self.pm.app_os/2, 0])  # Slightly above the target
        insert_rot = self.pm.grasp_rot_r
        insert_rot_fine = ls_add(insert_rot, [0, 0, yaw])
        # 7. Compute poses (relative to hole frame, inserting from behind)
        # Note: we insert in the X direction, so we back off in -X to approach
        
        waypoints = []
        waypoints.append(ls_concat(insert_app_pos, insert_rot_fine))
        self.yumi.left_go_thro(waypoints, "Insert Approach", eef_step=0.05)
        
        waypoints = []
        waypoints.append(ls_concat(insert_pos, insert_rot_fine))
        self.yumi.left_go_thro(waypoints, "Insert Marker", eef_step=0.05, velocity_scaling=0.5)

        self.yumi.open_left_gripper()  # release the marker
        
        waypoints = []
        waypoints.append(ls_add(insert_app_pos, [0, -self.pm.app_os, 2* self.pm.app_os]) + insert_rot_fine)
        self.yumi.left_go_thro(waypoints, "Insert Retract", eef_step=0.05)
        ''' 7. Reset left arm '''
      
        self.yumi.left_go_grasp()
        self.yumi.close_left_gripper()  # reset gripper
        self.yumi.wait_for_side_thread()
        return True
    
    
    
    
    def right_refine_orientation(self):
        # calc transfer poses
        transfer_point_l = ls_add(self.pm.hand_over_centre_2, 
                                  [self.pm.da_os_x,self.pm.marker_length/2+self.pm.gp_tip_w/2,-0.003])
        transfer_pose_l = list(transfer_point_l)+[-pi/2, 0, 0]
        transfer_pose_l2 = list(transfer_point_l)+[-pi/2, -pi/2, 0]
        transfer_pose_r = list(self.pm.hand_over_centre_2)+[0, 0, pi/2]

        # move left arm
        self.yumi.left_go_observe(main=False)
        self.yumi.open_left_gripper(main=False, full=1)

        self.yumi.right_go_transfer()
        # calc target ee_pose
        self.yumi.wait_for_side_thread()
        left_target = compose_matrix(translate=transfer_point_l[:3], angles=transfer_pose_l[3:])
        waypoints = [tf_mat2ls(left_target@tf_ls2mat([0, 0, 0, 0, 0, 0]))]
        self.yumi.left_tip_go_thro(waypoints, "Refine pinch 1")
        
        self.yumi.close_left_gripper(full_force=False)
        self.yumi.open_left_gripper()
        
        left_target = compose_matrix(translate=transfer_pose_l2[:3], angles=transfer_pose_l2[3:])
        waypoints = [tf_mat2ls(left_target@tf_ls2mat([0, 0, 0, 0, 0, 0]))]
        self.yumi.left_tip_go_thro(waypoints, "Refine pinch 2")

        self.yumi.close_left_gripper(full_force=False)
        self.yumi.open_left_gripper()

        # retreat left arm
        waypoints = [tf_mat2ls(left_target@tf_ls2mat([0, 0, 0.1, 0, 0, 0]))]
        self.yumi.left_tip_go_thro(waypoints, "Refine retract")

        self.yumi.left_go_observe(main=False)
        self.yumi.close_left_gripper(full_force=False)

        # retreat after action
        right_target = compose_matrix(translate=transfer_pose_r[:3], angles=transfer_pose_r[3:])
        waypoints = [tf_mat2ls(right_target@tf_ls2mat([0, 0, 0.05, 0, 0, 0]))]
        self.yumi.right_tip_go_thro(waypoints, "Place Retraction")
        
        self.yumi.wait_for_side_thread()

    def left_refine_orientation(self):
        #### change orientation
        transfer_point_r = ls_add(self.pm.hand_over_centre_2, 
                                  [-self.pm.da_os_x,-self.pm.marker_length/2,-self.pm.da_os_z])
        transfer_pose_r = list(transfer_point_r)+[pi/2, 0, 0]
        transfer_pose_r2 = list(transfer_point_r)+[pi/2, -pi/2, 0]
        transfer_pose_l = list(self.params.hand_over_centre_2)+[0, 0, -pi/2]

        # move right arm
        self.yumi.right_go_observe(main=False)
        self.yumi.open_right_gripper(main=False, full=1)

        self.yumi.left_go_transfer()
        # calc target ee_pose
        self.yumi.wait_for_side_thread()
        right_target = compose_matrix(translate=transfer_point_r[:3], angles=transfer_pose_r[3:])
        waypoints = [tf_mat2ls(right_target@tf_ls2mat([0, 0, 0, 0, 0, 0]))]
        self.yumi.right_tip_go_thro(waypoints, "Refine pinch 1")
        
        self.yumi.close_right_gripper(full_force=False)
        self.yumi.open_right_gripper()
        
        right_target = compose_matrix(translate=transfer_pose_r2[:3], angles=transfer_pose_r2[3:])
        waypoints = [tf_mat2ls(right_target@tf_ls2mat([0, 0, 0, 0, 0, 0]))]
        self.yumi.right_tip_go_thro(waypoints, "Refine pinch 2")

        self.yumi.close_right_gripper(full_force=False)
        self.yumi.open_right_gripper()

        # retreat left arm
        waypoints = [tf_mat2ls(right_target@tf_ls2mat([0, 0, 0.1, 0, 0, 0]))]
        self.yumi.right_tip_go_thro(waypoints, "Refine retract")

        self.yumi.right_go_observe(main=False)
        self.yumi.close_right_gripper(full_force=False)

        # retreat after action
        left_target = compose_matrix(translate=transfer_pose_l[:3], angles=transfer_pose_l[3:])
        waypoints = [tf_mat2ls(left_target@tf_ls2mat([0, 0, 0.05, 0, 0, 0]))]
        self.yumi.left_tip_go_thro(waypoints, "Place Retraction")
        
        self.yumi.wait_for_side_thread()

    def right_stretch_backward(self, rope, marker):
        self.yumi.change_speed(0.25)
        rope_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_z = 0.2
        stretch_y = -sqrt(rope_length**2-(stretch_z-self.pm.gp_os-root[2])**2)*cos(pi/4)+root[1]
        stretch_y = max(stretch_y, -0.3)
        stretch_x = -sqrt(rope_length**2-(stretch_y-root[1])**2-(stretch_z-self.pm.gp_os-root[2])**2)
        stretch_point = [root[0]+stretch_x, stretch_y, stretch_z]
        stretch_point_retract = [root[0]+stretch_x/2, stretch_y, stretch_z]
        stretch_rot = [0, pi, 0]
        self.add_to_log("[Right stretch backward to] "+str(stretch_point))

        # stretch the shoelace backward
        waypoints = []
        waypoints.append(ls_concat(stretch_point, stretch_rot))
        self.yumi.right_go_thro(waypoints,"Stretch Backward") 
        rospy.sleep(1)
        waypoints = []
        waypoints.append(ls_concat(stretch_point_retract, stretch_rot))
        self.yumi.right_go_thro(waypoints,"Stretch Backward Retract")
        self.yumi.change_speed(1)

    def right_stretch_forward(self, rope, marker):
        self.yumi.change_speed(0.5)
        rope_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_x = max(root[0], self.pm.site_r1[0]+self.pm.app_os)-self.pm.gp_os
        stretch_z = 0.25
        stretch_y = root[1]-sqrt(rope_length**2-(stretch_z-root[2])**2-(stretch_x+self.pm.gp_os-root[0])**2)
        stretch_point = [stretch_x, stretch_y, stretch_z]
        stretch_point_retract = [stretch_x, stretch_y/2, stretch_z]
        stretch_rot = [0, pi/2, 0]
        self.add_to_log("[Right stretch forward to] "+str(stretch_point))

        # stretch the shoelace forward
        waypoints = []
        waypoints.append(ls_concat(stretch_point, stretch_rot))
        self.yumi.right_go_thro(waypoints,"Stretch Forward")
        rospy.sleep(1)
        waypoints = []
        waypoints.append(ls_concat(stretch_point_retract, stretch_rot))
        self.yumi.right_go_thro(waypoints,"Stretch Forward Retract")
        self.yumi.change_speed(1)

    def left_stretch_backward(self, rope, marker):
        self.yumi.change_speed(0.25)
        rope_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_z = 0.2
        stretch_y = sqrt(rope_length**2-(stretch_z-self.pm.gp_os-root[2])**2)*cos(pi/4)+root[1]
        stretch_y = min(stretch_y, 0.3)
        stretch_x = -sqrt(rope_length**2-(stretch_y-root[1])**2-(stretch_z-self.pm.gp_os-root[2])**2)
        stretch_point = [root[0]+stretch_x, stretch_y, stretch_z]
        stretch_point_retract = [root[0]+stretch_x/2, stretch_y, stretch_z]
        stretch_rot = [0, pi, 0]
        self.add_to_log("[Left stretch backward to] "+str(stretch_point))

        # stretch the shoelace backward
        waypoints = []
        waypoints.append(ls_concat(stretch_point, stretch_rot))
        self.yumi.left_go_thro(waypoints,"Stretch Backward")
        rospy.sleep(1)
        waypoints = []
        waypoints.append(ls_concat(stretch_point_retract, stretch_rot))
        self.yumi.left_go_thro(waypoints,"Stretch Backward Retract")
        self.yumi.change_speed(1)
    
    def left_stretch_forward(self, rope, marker):
        self.yumi.change_speed(0.5)
        rope_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_x = max(root[0], self.pm.site_l1[0]+self.pm.app_os)-self.pm.gp_os
        stretch_z = 0.25
        stretch_y = root[1]+sqrt(rope_length**2-(stretch_z-root[2])**2-(stretch_x+self.pm.gp_os-root[0])**2)
        stretch_point = [stretch_x, stretch_y, stretch_z]
        stretch_point_retract = [stretch_x, stretch_y/2, stretch_z]
        stretch_rot = [0, pi/2, 0]
        self.add_to_log("[Left stretch forward to] "+str([stretch_x, stretch_y, stretch_z]))

        # stretch the shoelace forward
        waypoints = []
        waypoints.append(ls_concat(stretch_point, stretch_rot))
        self.yumi.left_go_thro(waypoints,"Stretch Forward")
        rospy.sleep(1)
        waypoints = []
        waypoints.append(ls_concat(stretch_point_retract, stretch_rot))
        self.yumi.left_go_thro(waypoints,"Stretch Forward Retract")
        self.yumi.change_speed(1)

    

    def right_to_left_handover(self):
        # calc the transfer points
        centre = self.pm.hand_over_centre_2
        transfer_point_r = ls_add(centre, [0, -self.pm.gp_os, 0])
        transfer_point_r_approach = ls_add(transfer_point_r, [0, -self.pm.app_os, 0])
        transfer_rot_r = [-pi/2, pi, 0]
        transfer_point_l = ls_add(centre, [self.pm.da_os_x, self.pm.gp_os, self.pm.gp_tip_w])
        transfer_point_l_approach = ls_add(transfer_point_l, [0, self.pm.app_os, 0])
        transfer_rot_l = [pi/2, pi, 0]
        
        # right to preparation position
        waypoints = []
        waypoints.append(ls_concat(transfer_point_r_approach, transfer_rot_r))
        waypoints.append(ls_concat(transfer_point_r, transfer_rot_r))
        self.yumi.right_go_thro(waypoints, "Handover Right", main=False)
        # reset left arm
        self.yumi.left_go_observe()
        self.yumi.open_left_gripper()
        # grasp with left gripper
        waypoints = []
        waypoints.append(ls_concat(transfer_point_l_approach, transfer_rot_l))
        waypoints.append(ls_concat(transfer_point_l, transfer_rot_l))
        self.yumi.left_go_thro(waypoints, "Handover Left")
        self.yumi.wait_for_side_thread()
        self.yumi.close_left_gripper()
        self.yumi.open_right_gripper()
        rope, marker = self.get_marker_at('right_gripper')
        # self.update_marker_ownership(rope, marker, 'left_gripper')
        # retract right arm
        waypoints = []
        waypoints.append(ls_concat(transfer_point_r_approach, transfer_rot_r))
        self.yumi.right_go_thro(waypoints, "Handover Right Retract", main=True)
        # reset right arm
        self.yumi.close_right_gripper(main=False)
        self.yumi.right_go_observe(main=False)

    def left_to_right_handover(self):
        # calc the transfer points
        centre = self.pm.hand_over_centre_2
        transfer_point_l = ls_add(centre, [0, self.pm.gp_os, 0])
        transfer_point_l_approach = ls_add(transfer_point_l, [0, self.pm.app_os, 0])
        transfer_rot_l = [pi/2, pi, 0]
        transfer_point_r = ls_add(centre, [-self.pm.da_os_x, -self.pm.gp_os, self.pm.gp_tip_w])
        transfer_point_r_approach = ls_add(transfer_point_r, [0, -self.pm.app_os, 0])
        transfer_rot_r = [-pi/2, pi, 0]

        # reset left arm
        self.yumi.left_go_observe(main=False)
        # left to preparation position
        waypoints = []
        waypoints.append(ls_concat(transfer_point_l_approach, transfer_rot_l))
        waypoints.append(ls_concat(transfer_point_l, transfer_rot_l))
        self.yumi.left_go_thro(waypoints, "Handover Left", main=False)
        # reset right arm
        self.yumi.right_go_observe()
        self.yumi.open_right_gripper()
        # grasp with right gripper
        waypoints = []
        waypoints.append(ls_concat(transfer_point_r_approach, transfer_rot_r))
        waypoints.append(ls_concat(transfer_point_r, transfer_rot_r))
        self.yumi.right_go_thro(waypoints, "Handover Right")
        self.yumi.wait_for_side_thread()
        self.yumi.close_right_gripper()
        self.yumi.open_left_gripper()
        rope, marker = self.get_marker_at('left_gripper')
        # self.update_marker_ownership(rope, marker, 'right_gripper')
        # retract left arm
        waypoints = []
        # waypoints.append([centre[0]+extra_os_x, centre[1]+self.pm.gp_os+0.20, centre[2], pi/2, pi, 0])
        waypoints.append(ls_concat(transfer_point_l_approach, transfer_rot_l))
        self.yumi.left_go_thro(waypoints, "Handover Left Retract", main=False)
        # reset left arm
        self.yumi.close_left_gripper(main=False)
        self.yumi.left_go_observe(main=False)

    def right_replace(self, rope, marker, site='site_r1'):
        self.right_pick(rope, marker, fine_ori=False)
        self.right_place(rope, marker, site=site)

    def left_replace(self, rope, marker, site='site_l1'):
        self.left_pick(rope, marker, fine_ori=False)
        self.left_place(rope, marker, site=site)

    def right_to_left(self, rope, marker, reset=True, site='site_l1'):
        self.right_pick(rope, marker, fine_ori=False)
        self.right_to_left_handover()
        self.left_place(rope, marker, site=site)

        # reset to initial position
        self.yumi.close_left_gripper(main=False)
        if reset:
            self.yumi.left_go_observe()
        self.yumi.wait_for_side_thread()

    def left_to_right(self, rope, marker, reset=True, site='site_r1'):
        self.left_pick(rope, marker, fine_ori=False)
        self.left_to_right_handover()
        self.right_place(rope, marker, site=site)

        # reset to initial position
        self.yumi.close_right_gripper(main=False)
        if reset:
            self.yumi.right_go_observe()
        self.yumi.wait_for_side_thread()


    def call_rope_srv(self, rope):
        '''
        input: rope (colour e.g. red, blue)
        output: marker_a_pose, marker_b_pose
        '''
        request = DetectRopeRequest()
        request.rope = rope
        while not rospy.is_shutdown() :
            # get the target pose
            response = self.find_rope(request)
            if response.success == True:
                # if self.yumi.check_command('Satisfied with the result?'):
                    break
            #     # else:
            #     #     rospy.logwarn('Retrying to find the rope...')
        return pose_msg_to_list(response.marker_a_pose), pose_msg_to_list(response.marker_b_pose)
       
    
    
    
    def call_scene_srv(self):
        '''
        input: None
        output: list of ropes from top to bottom
        '''
        request = ObserveSceneRequest()
        while not rospy.is_shutdown():
            # get the target pose
            response = self.observe_scene(request)
            if response.success == False:
                if not self.yumi.check_command('Got empty reply. Try again?'):
                    print('Cancelled action. Exiting.')
                    exit()
            else:
                # if self.yumi.check_command('Satisfied with the result?'):
                break
        
        return response.ropes, response.img, 

    def pub_hand_poses(self, hand, pose):
        self.pm.hand_poses[self.pm.gripper_dict[hand]] = pose
        poses = PoseArray()
        for e in self.pm.hand_poses:
            poses.poses.append(list_to_pose_msg(e))
        poses.header.frame_id = self.yumi.robot_frame
        self.hand_pose_pub.publish(poses)

    def pub_marker_pose(self, rope, marker):
        if self.pm.rope_dict[rope]:
            rope_ = self.pm.rope_dict[rope]
        else:
            print('Unknown rope name!')
            return
        if rope_.marker_dict[marker]:
            marker_ = rope_.marker_dict[marker]
        else:
            print('Unknown marker name!')
            return 
        
        poses = PoseArray()
        poses.poses.append(list_to_pose_msg(marker_['pose']))
        poses.header.frame_id = self.yumi.robot_frame
        self.marker_pose_pubs[rope][marker].publish(poses)

    def update_target_poses(self, pose, id):
        if len(self.pm.target_poses)>id:
            self.pm.target_poses[id] = pose

    def pub_target_poses(self):
        poses = PoseArray()
        for e in self.pm.target_poses:
            poses.poses.append(list_to_pose_msg(e))
        poses.header.frame_id = self.yumi.robot_frame
        self.target_pose_pub.publish(poses)

    def get_rope_poses(self, rope, marker, initial=False):
        '''
        input: rope (colour e.g. rope_o, rope_b), marker (marker_a, marker_b)
        output: marker_pose, yaw
        '''
        if marker == 'marker_a':
            marker_poses, _ = self.call_rope_srv(rope)
            self.pm.rope_dict[rope].marker_dict[marker]['pose'] = marker_poses
        elif marker == 'marker_b':
            _, marker_poses = self.call_rope_srv(rope)
            self.pm.rope_dict[rope].marker_dict[marker]['pose'] = marker_poses
        else:
            print("Unknown marker name!")
            return None, None
        site = self.pm.find_closest_site(rope=rope, marker=marker)
        if site is None:
            self.get_rope_poses(rope, marker, initial=initial) 
            
        self.yumi.rope_dict[rope]['marker_dict'][marker]['marker_at'] = site
        # self.pm.update_site_occupancy(rope, marker, site)
        
        site = self.pm.check_marker_location(rope, marker) 
        
        side = self.pm.sites_dict[site] # -1 for left, 1 for right, 0 for centre
        if initial==False:
            if side==-1:
                self.yumi.left_go_grasp()
            elif side==1:
                self.yumi.right_go_grasp()
            else:
                self.yumi.left_go_grasp()
                self.yumi.right_go_grasp()
        # get the target pose
        
        # post processing
        marker_pose = ls_add(marker_poses, (self.pm.l_l_offset if side==-1 else self.pm.l_r_offset)+[0,0,0,0])
        [_, _, yaw] = euler_from_quaternion(marker_pose[3:]) # RPY
        self.pm.rope_dict[rope].marker_dict[marker]['position'] = marker_pose
        # publish aglet pose
        self.pub_marker_pose(rope, marker)
        return marker_pose, yaw

        
    
    
    def get_scene_poses(self):
        ropes, img = self.call_scene_srv()
        for rope in ropes:
            self.pm.heirarchy.append(rope)
        self.pm.img_frame = img
        
    # def update_marker_ownership(self, rope, marker, site):
    #     self.pm.rope_dict[rope].marker_dict['marker_a']['marker_at'] = site
        
    #     owner_msg = Int32MultiArray()
    #     owner_msg.data = [site-6, 
    #                     self.pm.sites_dict[self.pm.aglet_at['aglet_b']]-6] # left 0, right 1
    #     self.aglet_owner_pub.publish(owner_msg)

    def get_marker_at(self, site):
        '''
        input:
        output: rope_name, marker 
        '''
        for rope_name, rope in self.pm.rope_dict.items():  
            for marker, s in rope.marker_dict.items():
                if s['marker_at'] == site: return rope_name, marker
        return None

    def add_to_log(self, content):
        self.logs_pub.publish(String(content))
        rospy.sleep(0.5)

    def init_target_poses(self):
        

        self.pub_target_poses() # publish target poses
        
    
    
        for rope in ['rope_o', 'rope_g', 'rope_b']:
            self.get_rope_poses(rope, 'marker_a', initial=True)
            self.get_rope_poses(rope, 'marker_b', initial=True)
            
        self.yumi.both_go_grasp()
        self.get_scene_poses()
        
        # self.pm.update_yumi_constriants('marker_b', self.pm.rope_length_l, self.pm.get_root_position('marker_b'))
        # self.pm.update_yumi_constriants('marker_a', self.pm.rope_length_r, self.pm.get_root_position('marker_a'))
        
        
    def stop(self):
        self.yumi.stop()
        self.pm.save_params()


if __name__ == "__main__":
    rospy.init_node('scene_ctrl', anonymous=True)
    
  
  
    # def update_cursor(self, name, id):
    #     if name=='left':
    #         self.pm.left_cursor = id
    #         self.cursor_pub.publish(Int32MultiArray(data=[id, self.pm.right_cursor]))
    #     elif name=='right':
    #         self.pm.right_cursor = id
    #         self.cursor_pub.publish(Int32MultiArray(data=[self.pm.left_cursor, id]))
    #     else:
    #         print('Unknown cursor name!')