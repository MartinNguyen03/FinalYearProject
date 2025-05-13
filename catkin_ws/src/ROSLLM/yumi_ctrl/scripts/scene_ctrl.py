import threading
import numpy as np
from os import path
from scipy.spatial import distance
from math import cos, sin, pi, sqrt

from scene_params import SceneParameters
from utils import list_to_pose_msg, ls_concat, ls_add, tf_ls2mat, tf_mat2ls, pose_msg_to_list, is_sorted
from catkin_ws.src.ROSLLM.yumi_ctrl.scripts.yumi_wrapper import YumiWrapper
import tf
import rospy
import rospkg
from rosllm_srvs.srv import DetectTarget, DetectTargetRequest, DetectRope, DetectRopeRequest, ObserveScene, ObserveSceneRequest
from std_msgs.msg import String, Int32MultiArray
from geometry_msgs.msg import PoseArray
from tf.transformations import euler_from_quaternion, compose_matrix




class ScenePrimitives:
    package_name = 'scene_ctrl'
    marker_topic = '/visualization_marker'
    # robot_frame = "yumi_base_link"
    target_srv = 'find_targets'
    rope_srv = 'find_ropes'
    scene_srv = 'observe_scene'
    log_topic = 'scene_logs'
    rope_names_topic = ['rope_r', 'rope_b', 'rope_g']
    marker_ids_topic = ['marker_a', 'marker_b']
    hand_pose_topic = 'hand_pose'
    target_pose_topic = 'target_pose'
    # cursor_topic = 'cursor'

    def __init__(self, auto_execution, reset=True, start_id=0):
        # ros related initialisation
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
        self.find_rope = rospy.ServiceProxy(self.rope_srv, DetectRope)
        rospy.wait_for_service(self.rope_srv)
        self.find_targets = rospy.ServiceProxy(self.target_srv, DetectTarget)
        rospy.wait_for_service(self.target_srv)
        self.observe_scene = rospy.ServiceProxy(self.scene_srv, ObserveScene)
        rospy.wait_for_service(self.scene_srv)

        # load params
        self.debug = False
        self.pkg_path = rospkg.RosPack().get_path(self.package_name)
        self.pm = SceneParameters(reset=True, start_id=start_id,
                                           config_path=path.join(self.pkg_path, 'params'),
                                           result_path=path.join(self.pkg_path, 'results'),
                                           log_handle=self.add_to_log)
        # self.update_cursor('left', self.pm.left_cursor)
        # self.update_cursor('right', self.pm.right_cursor)


       
        self.yumi = YumiWrapper(auto_execution, rope_dict=self.pm.rope_dict,
            workspace=self.pm.workspace,
            vel_scale=self.pm.vel_scale,
            gp_opening=self.pm.marker_thickness*1000*2, # to mm
            table_offset=self.pm.table_offset,
            grasp_states=self.pm.grasp_states,
            grasp_states2=self.pm.grasp_states2,
            observe_states=self.pm.observe_states)

        # scan the shoe
        self.pm.update_yumi_constriants = self.yumi.update_sl_constriants
        self.init_target_poses()
        
        rospy.loginfo('Execution module ready.')

        # create and start the tf listener thread
        self.side_queue = []
        self.side_thread = threading.Thread(target=self.tf_thread_func, daemon=True)
        self.side_thread.start()

    def tf_thread_func(self):
        rate = rospy.Rate(10.0)
        while not rospy.is_shutdown():
            try:
                
                if self.get_marker_at('left_gripper') is not None:
                    (trans,rot) = self.tf_listener.lookupTransform(self.yumi.robot_frame, self.yumi.ee_frame_l, rospy.Time(0))
                    self.pub_hand_poses('left_gripper', trans+rot)
                    pose = tf_mat2ls(tf_ls2mat(trans+rot)@self.yumi.gripper_l_to_aglet)
                    rope, marker = self.get_marker_at('left_gripper')
                    self.pub_marker_pose(marker, pose)
                elif self.get_marker_at('right_gripper') is not None:
                    (trans,rot) = self.tf_listener.lookupTransform(self.yumi.robot_frame, self.yumi.ee_frame_r, rospy.Time(0))
                    self.pub_hand_poses('right_gripper', trans+rot)
                    pose = tf_mat2ls(tf_ls2mat(trans+rot)@self.yumi.gripper_r_to_aglet)
                    rope, marker = self.get_marker_at('right_gripper')
                    self.pub_marker_pose(rope, marker, pose)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            rate.sleep()                
                
                
    # def execute(self, action, next_action, sl_cost):
    #     '''
    #     pick: gripper aglet site
    #     ''' 
    #     self.add_to_log(f'{action} Cost: {sl_cost}')
    #     # interpret the action
    #     if 'left_insert' in action[0]:
    #         reset = False if next_action is not None and next_action[0][:4]=='left' else True
    #         self.left_lace('targets_l', int(action[1][6:]), 'aglet{}'.format(action[0][-2:]), 
    #                                  sl_cost=sl_cost+self.pm.target_to_edge, reset=reset, site=action[-1])
    #     elif 'right_insert' in action[0]:
    #         reset = False if next_action is not None and next_action[0][:5]=='right' else True
    #         self.right_lace('targets_r', int(action[1][6:]), 'aglet{}'.format(action[0][-2:]), 
    #                                    sl_cost=sl_cost+self.pm.target_to_edge, reset=reset, site=action[-1])
    #     elif action[0] == 'right_to_left_transfer':
    #         reset = False if next_action is not None and next_action[0][:4]=='left' else True
    #         self.right_to_left(action[1], reset=reset, site=action[-1])
    #     elif action[0] == 'left_to_right_transfer':
    #         reset = False if next_action is not None and next_action[0][:5]=='right' else True
    #         self.left_to_right(action[1], reset=reset, site=action[-1])
    #     elif action[0] == 'left_replace':
    #         self.left_replace(action[1], site=action[-1])
    #     elif action[0] == 'right_replace':
    #         self.right_replace(action[1], site=action[-1])
    #     else:
    #         print('Unrecogised primitive name ({})!'.format(action[0]))

    def right_pick(self, rope, marker, fine_ori=True):
        """ pick up the marker with the right gripper """
        # calc pick poses
        if self.pm.check_marker_location(rope, marker)[:4] != 'site':
            if self.pm.check_marker_location(rope, marker)[:6] == 'target':
                self.right_remove(rope, marker)
                return
            else:
                self.add_to_log("Marker is not available for pick up!")
                return
            
        marker_pos, yaw = self.get_rope_poses(rope, marker)
        pick_pos = [marker_pos[0], marker_pos[1], marker_pos[2]+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os])
        pick_rot = self.pm.grasp_rot_r
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])

        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.right_go_thro(waypoints,"Pick Approach")
        self.yumi.open_right_gripper()

        # pick the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick")
        self.yumi.close_right_gripper()

        self.update_marker_ownership(rope, marker, 'right_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Retract")
        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupency(rope, marker, current_site, False)
        self.pm.update_site_occupency(rope, marker, 'right_gripper')

        if fine_ori:
            self.right_refine_orientation()

    def left_pick(self, rope, marker, fine_ori=True):
        """ pick up the aglet with the left gripper """
        
        if self.pm.check_marker_location(rope, marker)[:4] != 'site':
            if self.pm.check_marker_location(rope, marker)[:6] == 'target':
                self.left_remove(rope, marker)
                return
            else:
                self.add_to_log("Marker is not available for pick up!")
                return
        # calc pick poses
        marker_pos, yaw = self.get_rope_poses(rope, marker)
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

        # pick the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick")
        self.yumi.close_left_gripper()
        self.update_marker_ownership(rope, marker, 'left_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Retract")

        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupency(rope, marker, current_site, False)
        self.pm.update_site_occupency(rope, marker, 'left_gripper')

        if fine_ori:
            self.left_refine_orientation()

    def right_place(self, rope, marker, site='site_dr'):
        """ place down the aglet with the right gripper"""

        # stretch the shoelace
        self.add_to_log("Placing to "+site)
        if site == 'site_dr':
            self.yumi.right_go_grasp()
            section = self.pm.site_poses['site_dr']
            self.right_stretch_backward(rope, marker)
            self.yumi.right_go_grasp()
        elif site == 'site_dd':
            self.yumi.right_go_grasp()
            section = self.pm.site_poses['site_dd']
            self.right_stretch_backward(rope, marker)
            self.yumi.right_go_grasp()
        elif site == 'site_ur':
            self.yumi.right_go_grasp()
            section = self.pm.site_poses['site_ur']
            self.right_stretch_forward(rope, marker)
            self.yumi.right_go_grasp()
        elif site == 'site_uu':
            self.yumi.right_go_grasp()
            section = self.pm.site_poses['site_uu']
            self.right_stretch_forward(rope, marker)
            self.yumi.right_go_grasp()
        else:
            rospy.logerr("No Section is available for placing!")
            return

        self.pm.update_site_occupency(rope, marker, 'right_gripper', False)
        self.pm.update_site_occupency(rope, marker, site)

        # calc place poses
        place_pos = [section[0], section[1], section[2]+self.pm.gp_os]
        place_pos_approach = ls_add(place_pos, [0, -self.pm.app_os, self.pm.app_os])
        place_rot = self.pm.grasp_rot_r
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
        self.update_marker_ownership(rope, marker, site)
        # retreat after place
        waypoints = []
        waypoints.append(place_pos_approach+place_rot)
        self.yumi.right_go_thro(waypoints,"Place Retract")
        self.yumi.add_table()

    def left_place(self, rope, marker, site='site_dd'):
        """ place down the marker with the right gripper"""

        # stretch the shoelace
        self.add_to_log("Placing to "+site)
        
        if site == 'site_dl':
            self.yumi.left_go_grasp()
            section = self.pm.site_poses['site_dl']
            self.left_stretch_backward(marker)
            self.yumi.left_go_grasp()
        elif site == 'site_dd':
            self.yumi.left_go_grasp()
            section = self.pm.site_poses['site_dd']
            self.left_stretch_backward(marker)
            self.yumi.left_go_grasp()
        elif site == 'site_ul':
            self.yumi.left_go_grasp()
            section = self.pm.site_poses['site_ul']
            self.left_stretch_forward(marker)
            self.yumi.left_go_grasp()
        elif site == 'site_uu':
            self.yumi.left_go_grasp()
            section = self.pm.site_poses['site_uu']
            self.left_stretch_forward(marker)
            self.yumi.left_go_grasp()
        else:
            rospy.logerr("No Section is available for placing!")
            return

        self.pm.update_site_occupency(rope, marker, 'left_gripper', False)
        self.pm.update_site_occupency(rope, marker, site)

        # calc place poses
        place_pos = [section[0], section[1], section[2]+self.pm.gp_os]
        place_pos_approach = ls_add(place_pos, [0, self.pm.app_os, self.pm.app_os])
        place_rot = self.pm.grasp_rot_l
        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(place_pos_approach+place_rot)
        self.yumi.left_go_thro(waypoints,"Place Approach", eef_step=0.05)
        # place the marker
        waypoints = []
        waypoints.append(place_pos+place_rot)
        self.yumi.left_go_thro(waypoints,"Place")
        self.yumi.open_left_gripper(full=True)
        self.update_marker_ownership(marker, site)
        # retreat after place
        waypoints = []
        waypoints.append(place_pos_approach+place_rot)
        self.yumi.left_go_thro(waypoints,"Place Retract")
        self.yumi.add_table()

    def right_insert(self, rope, marker, target, reset=False):
        """
        This primitive inserts the marker (aglet) from inside the object toward the outside
        using only the right gripper.
        """
        # 1. Parse target info
        target_group, target_id = self.pm.get_target_id(target)

        # 2. Pick up the aglet
        self.right_pick(rope, marker)

        # 3. Get initial target position
        prev_target_pos = self.pm.target_poses[target_id][:3]

        # 4. Refine target pose using vision
        self.get_target_poses(target_group, target_id=target_id//2, init=False, fine=True,
                              compare_with=prev_target_pos)
        target_pos = self.pm.target_poses[target_id][:3]

        # 5. Define insertion parameters
        insert_pitch = self.pm.insert_pitch2  # usually ~pi/2 if inserting downwards

        # 6. Compose matrix at hole location
        target_pose = compose_matrix(translate=target_pos, angles=[0, 0, pi/2])

        # 7. Compute poses (relative to hole frame, inserting from behind)
        # Note: we insert in the X direction, so we back off in -X to approach
        insert_pose = tf_mat2ls(target_pose @ tf_ls2mat([self.pm.target_radius, 0, 0,
                                                         insert_pitch, 0, 0]))
        insert_pose_approach = tf_mat2ls(target_pose @ tf_ls2mat([self.pm.target_radius + self.pm.app_os*2, 0, 0,
                                                                  insert_pitch, 0, 0]))
        retract_pose = tf_mat2ls(target_pose @ tf_ls2mat([self.pm.target_radius + self.pm.app_os*4, 0, 0,
                                                          insert_pitch, 0, 0]))

        # 8. Move to pre-insert posture
        self.yumi.right_go_thro([self.pm.pre_insert_r + [0, pi - insert_pitch, 0]], "Insert Prepare")

        # 9. Execute insertion
        self.yumi.right_tip_go_thro([insert_pose_approach, insert_pose], "Insert Marker", velocity_scaling=1.5)
        self.yumi.open_right_gripper(full=1)  # release the marker

        # 10. Retract
        self.yumi.right_tip_go_thro([insert_pose_approach, retract_pose], "Retract Marker")

        # 11. Final reset and state update
        self.update_marker_ownership(rope, marker, target)  # no longer held
        if reset:
            self.yumi.right_go_observe()
        self.yumi.wait_for_side_thread()


    def left_insert(self, rope, marker, target, reset=False):
        """
        Insert aglet from inside to outside of an target using the left gripper only.
        Simplified version of left_lace with no grasp or pull, and one-arm control.
        """

        ''' 1. Pick the aglet with the left gripper '''
        target_group, target_id = self.pm.get_target_id(target)
        self.left_pick(marker)
        
        ''' 2. Locate the target target (coarse) '''
        prev_target_pos = self.pm.target_poses[target_id][:3]
        
        self.get_target_poses(target_group, target_id=target_id//2, fine=False,
                              compare_with=prev_target_pos)
        
        target_pos = self.pm.target_poses[target_id][:3]

        ''' 4. Compute insertion poses (inside-out logic) '''
        insert_pitch = self.pm.insert_pitch2  # downward pitch (~π/2)
        e_pose = compose_matrix(translate=target_pos, angles=[0, 0, -pi/2])  # Fix orientation of hole
        insert_pose = tf_mat2ls(e_pose @ tf_ls2mat([self.pm.target_radius, 0, 0,
                                                    insert_pitch, 0, 0]))
        insert_pose_approach = tf_mat2ls(e_pose @ tf_ls2mat([self.pm.target_radius + self.pm.app_os * 2, 0, 0,
                                                             insert_pitch, 0, 0]))
        insert_pose_retract = tf_mat2ls(e_pose @ tf_ls2mat([self.pm.target_radius + self.pm.app_os * 4, 0, 0,
                                                            insert_pitch, 0, 0]))

        ''' 5. Prepare and execute insertion '''
        self.yumi.left_go_thro([ls_concat(self.pm.pre_insert_l, [0, pi - insert_pitch, 0])], "Insert Prepare")
        self.yumi.left_tip_go_thro([insert_pose_approach, insert_pose], "Insert Marker", velocity_scaling=1.5)
        self.yumi.open_left_gripper(full=1)  # release aglet
        self.update_marker_ownership(rope, marker, target)  # no longer held

        ''' 6. Retract the gripper '''
        self.yumi.left_tip_go_thro([insert_pose_approach, insert_pose_retract], "Insert Retract")

        ''' 7. Reset left arm '''
        if reset:
            self.yumi.left_go_observe()
        self.yumi.wait_for_side_thread()

    
    def right_remove(self, rope, marker):
        
        marker_pos, yaw = self.get_rope_poses(rope, marker)
        pick_pos = [marker_pos[0], marker_pos[1], marker_pos[2]+self.pm.gp_os]
        pick_pos_approach = ls_add(pick_pos, [0, 0, self.pm.app_os])
        pick_rot = self.pm.grasp_rot_r
        pick_rot_fine = ls_add(pick_rot, [0, 0, yaw])

        self.yumi.remove_table()
        # approach the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot))
        self.yumi.right_go_thro(waypoints,"Pick Approach")
        self.yumi.open_right_gripper()

        # pick the marker
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick")
        self.yumi.close_right_gripper()

        self.update_marker_ownership(rope, marker, 'right_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.right_go_thro(waypoints,"Pick Retract")
        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupency(rope, marker, current_site, False)
        self.pm.update_site_occupency(rope, marker, 'right_gripper')

    
    
    def left_remove(self, rope, marker):
         # calc pick poses
        marker_pos, yaw = self.get_rope_poses(rope, marker)
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

        # pick the aglet
        waypoints = []
        waypoints.append(ls_concat(pick_pos,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick")
        self.yumi.close_left_gripper()
        self.update_marker_ownership(rope, marker, 'left_gripper')

        # retreat after pick
        waypoints = []
        waypoints.append(ls_concat(pick_pos_approach,pick_rot_fine))
        self.yumi.left_go_thro(waypoints,"Pick Retract")

        self.yumi.add_table()

        # set section flags
        current_site = self.pm.check_marker_location(rope, marker)
        if current_site is not None:
            self.pm.update_site_occupency(rope, marker, current_site, False)
        self.pm.update_site_occupency(rope, marker, 'left_gripper')
    
    
    
    
    def right_refine_orientation(self):
        # calc transfer poses
        transfer_point_l = ls_add(self.pm.hand_over_centre_2, 
                                  [self.pm.da_os_x,self.pm.aglet_length/2,0])
        pinch_rot_l = [-pi/2, 0, 0]
        pinch_rot_l2 = [-pi/2, -pi/2, 0]
        pinch_rot_r = [0, 0, pi/2]
        transfer_point_l_retreat = ls_add(transfer_point_l, [0, self.pm.app_os*2, 0])
        transfer_point_r_retreat = ls_add(self.pm.hand_over_centre_2, [0, 0, self.pm.app_os])

        # reset left arm
        self.yumi.left_go_observe(main=False)
        self.yumi.open_left_gripper(main=False, full=1)
        # reset right arm
        self.yumi.right_go_transfer()
        self.yumi.wait_for_side_thread()
        # pinch the aglet
        waypoints = [ls_concat(transfer_point_l, pinch_rot_l)]
        self.yumi.left_tip_go_thro(waypoints, "Reorient Pinch 1")
        self.yumi.close_left_gripper(full_force=False)
        self.yumi.open_left_gripper()
        # pinch the aglet again
        waypoints = [ls_concat(transfer_point_l, pinch_rot_l2)]
        self.yumi.left_tip_go_thro(waypoints, "Reorient Pinch 2")
        self.yumi.close_left_gripper(full_force=False)
        self.yumi.open_left_gripper()
        # retreat left arm
        waypoints = [ls_concat(transfer_point_l_retreat, pinch_rot_l2)]
        self.yumi.left_tip_go_thro(waypoints, "Reorient Left Retract")
        self.yumi.left_go_observe(main=False)
        self.yumi.close_left_gripper(full_force=False)
        # retreat after action
        waypoints = [ls_concat(transfer_point_r_retreat, pinch_rot_r)]
        self.yumi.right_tip_go_thro(waypoints, "Reorient Right Retract")
        self.yumi.wait_for_side_thread()

    def left_refine_orientation(self):
        #### change orientation
        transfer_point_r = ls_add(self.pm.hand_over_centre_2, 
                                  [-self.pm.da_os_x,-self.pm.aglet_length/2,-self.pm.da_os_z])
        pinch_rot_r = [pi/2, 0, 0]
        pinch_rot_r2 = [pi/2, -pi/2, 0]
        pinch_rot_l = [0, 0, -pi/2]
        transfer_point_r_retreat = ls_add(transfer_point_r, [0, -self.pm.app_os*2, 0])
        transfer_point_l_retreat = ls_add(self.pm.hand_over_centre_2, [0, 0, self.pm.app_os])

        # reset right arm
        self.yumi.right_go_observe(main=False)
        self.yumi.open_right_gripper(main=False, full=1)
        # reset left arm
        self.yumi.left_go_transfer()        
        self.yumi.wait_for_side_thread()
        # pinch the aglet
        waypoints = [ls_concat(transfer_point_r, pinch_rot_r)]
        self.yumi.right_tip_go_thro(waypoints, "Reorient Pinch 1")
        self.yumi.close_right_gripper(full_force=False)
        self.yumi.open_right_gripper()
        # pinch the aglet again
        waypoints = [ls_concat(transfer_point_r, pinch_rot_r2)]
        self.yumi.right_tip_go_thro(waypoints, "Reorient Pinch 2")
        self.yumi.close_right_gripper(full_force=False)
        self.yumi.open_right_gripper()
        # retreat left arm
        waypoints = [ls_concat(transfer_point_r_retreat, pinch_rot_r2)]
        self.yumi.right_tip_go_thro(waypoints, "Reorient Right Retract")
        self.yumi.right_go_observe(main=False)
        self.yumi.close_right_gripper(full_force=False)
        # retreat after action
        waypoints = [ls_concat(transfer_point_l_retreat, pinch_rot_l)]
        self.yumi.left_tip_go_thro(waypoints, "Reorient Left Retract")
        self.yumi.wait_for_side_thread()

    def right_stretch_backward(self, rope, marker):
        self.yumi.change_speed(0.25)
        sl_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_z = 0.2
        stretch_y = -sqrt(sl_length**2-(stretch_z-self.pm.gp_os-root[2])**2)*cos(pi/4)+root[1]
        stretch_y = max(stretch_y, -0.3)
        stretch_x = -sqrt(sl_length**2-(stretch_y-root[1])**2-(stretch_z-self.pm.gp_os-root[2])**2)
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
        sl_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_x = max(root[0], self.pm.site_r1[0]+self.pm.app_os)-self.pm.gp_os
        stretch_z = 0.25
        stretch_y = root[1]-sqrt(sl_length**2-(stretch_z-root[2])**2-(stretch_x+self.pm.gp_os-root[0])**2)
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
        sl_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_z = 0.2
        stretch_y = sqrt(sl_length**2-(stretch_z-self.pm.gp_os-root[2])**2)*cos(pi/4)+root[1]
        stretch_y = min(stretch_y, 0.3)
        stretch_x = -sqrt(sl_length**2-(stretch_y-root[1])**2-(stretch_z-self.pm.gp_os-root[2])**2)
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
        sl_length = self.pm.get_shoelace_length(rope, marker)
        root = self.pm.get_root_position(rope, marker)
        # calc stretch poses
        stretch_x = max(root[0], self.pm.site_l1[0]+self.pm.app_os)-self.pm.gp_os
        stretch_z = 0.25
        stretch_y = root[1]+sqrt(sl_length**2-(stretch_z-root[2])**2-(stretch_x+self.pm.gp_os-root[0])**2)
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
        self.update_marker_ownership(rope, marker, 'left_gripper')
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
        self.update_marker_ownership(rope, marker, 'right_gripper')
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
        request.rope_colour = rope
        while not rospy.is_shutdown():
            # get the target pose
            response = self.observe_scene(request)
            if len(response.marker_a_pose) == 0 or len(response.marker_b_pose) == 0:
                if not self.yumi.check_command('Got empty reply. Try again?'):
                    print('Cancelled action. Exiting.')
                    exit()
            else:
                if self.yumi.check_command('Satisfied with the result?'):
                    break
        return response.marker_a_pose, response.marker_b_pose
       
    
    def call_target_srv(self, target_name, camera_name):
        '''
        input: target_name, camera_name
        output: target_poses, confidence
        '''
        request = DetectTargetRequest()
        request.target_name = target_name
        request.camera_name = camera_name
        while not rospy.is_shutdown():
            # get the target pose
            response = self.find_targets(request)
            if len(response.target.poses) == 0:
                if not self.yumi.check_command('Got empty reply. Try again?'):
                    print('Cancelled action. Exiting.')
                    exit()
            else:
                if self.yumi.check_command('Satisfied with the result?'):
                    break
        target_poses = []
        for pose in response.target.poses:
            target_poses.append(pose_msg_to_list(pose))
        return target_poses, response.confidence.data
    
    def call_scene_srv(self):
        '''
        input: None
        output: list of ropes from top to bottom
        '''
        request = ObserveSceneRequest()
        while not rospy.is_shutdown():
            # get the target pose
            response = self.observe_scene(request)
            if len(response.success) == False:
                if not self.yumi.check_command('Got empty reply. Try again?'):
                    print('Cancelled action. Exiting.')
                    exit()
            else:
                if self.yumi.check_command('Satisfied with the result?'):
                    break
        for rope in response.ropes:
            self.pm.heirarchy.append(rope)
        self.pm.img_frame = response.img_frame
        
        return response.ropes, response.img

    def pub_hand_poses(self, hand, pose):
        self.pm.hand_poses[self.pm.gripper_dict[hand]] = pose
        poses = PoseArray()
        for e in self.pm.hand_poses:
            poses.poses.append(list_to_pose_msg(e))
        poses.header.frame_id = self.yumi.robot_frame
        self.hand_pose_pub.publish(poses)

    def pub_marker_pose(self, rope, marker, pose):
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
        marker_['position'] = pose
        poses = PoseArray()
        for a in marker_['position'].values():
            poses.poses.append(list_to_pose_msg(a))
        poses.header.frame_id = self.yumi.robot_frame
        self.marker_pose_pubs[rope][marker].publish(poses)

    def update_target_poses(self, pose, id):
        if len(self.pm.target_poses)>id:
            self.pm.target_poses[id] = pose

    # def update_cursor(self, name, id):
    #     if name=='left':
    #         self.pm.left_cursor = id
    #         self.cursor_pub.publish(Int32MultiArray(data=[id, self.pm.right_cursor]))
    #     elif name=='right':
    #         self.pm.right_cursor = id
    #         self.cursor_pub.publish(Int32MultiArray(data=[self.pm.left_cursor, id]))
    #     else:
    #         print('Unknown cursor name!')

    def pub_target_poses(self):
        poses = PoseArray()
        for e in self.pm.target_poses:
            poses.poses.append(list_to_pose_msg(e))
        poses.header.frame_id = self.yumi.robot_frame
        self.target_pose_pub.publish(poses)

    def get_rope_poses(self, rope, marker):
        '''
        input: rope (colour e.g. rope_r, rope_b), marker (marker_a, marker_b)
        output: marker_pose, yaw
        '''
        site = self.pm.check_marker_location(rope, marker) 
        side = self.pm.sites_dict[site] # -1 for left, 1 for right, 0 for centre
        if side==-1:
            self.yumi.left_go_grasp()
        elif side==1:
            self.yumi.right_go_grasp()
        else:
            self.yumi.left_go_grasp()
            self.yumi.right_go_grasp()
        # get the target pose
        if marker == 'marker_a':
            marker_poses, _ = self.call_rope_srv(rope)
        elif marker == 'marker_b':
            _, marker_poses = self.call_rope_srv(rope)
        else:
            print("Unknown marker name!")
            return None, None
        # post processing
        marker_pose = ls_add(marker_poses[0], (self.pm.l_l_offset if side==-1 else self.pm.l_r_offset)+[0,0,0,0])
        [_, _, yaw] = euler_from_quaternion(marker_pose[3:]) # RPY
        self.pm.rope_dict[rope].marker_dict[marker]['position'] = marker_pose
        # publish aglet pose
        self.pub_marker_pose(rope, marker, marker_pose)
        return marker, yaw

    def call_vision_target(self, target_name):
        # get the position of the target
        if target_name == 'targets_l':
            target_poses, confidence = self.call_target_srv(target_name, 'd435_l')
        elif target_name == 'targets_r':
            target_poses, confidence = self.call_target_srv(target_name, 'd435_r')
        else:
            print("Unknown target name!")
        # pose processing
        targets = []
        for id, target in enumerate(target_poses):
            target[3:] = list(euler_from_quaternion(target[3:]))
            if target_name == 'targets_r':
                target = ls_add(target, self.pm.e_r_offset+[0,0,0])
            elif target_name == 'targets_l':
                target = ls_add(target, self.pm.e_l_offset+[0,0,0])
            if confidence[id]<0.6: # impossible, depth must be wrong
                target[3:] = [0, pi/24*2, -pi/2] if target_name=='targets_l' else [0, pi/24*2, pi/2]
            targets.append(target)
        if target_name == 'targets_l' and self.pm.left_cursor==0:
            targets[0][3:] = [0, pi/24*2, -pi/2]
        return np.array(targets), confidence

    def sanity_check_targets(self, targets, target_id):
        if any(np.isnan(targets[target_id][:3])):
            return False
        if len(targets) > 1: # do not test when only one is detected
            return is_sorted(targets[:,0])  # x  should be ascending
        else:
            return True
        
    def get_target_poses(self, name, target_id=0, fine=False, init=True, compare_with=None):
        ''' 
            target_id: index in all targets
            target_id: index on one side
            return: list of translations and euelr angles 
        '''
        # go to observe pose if first time
        if name=='targets_l' and init:
            self.yumi.left_go_observe()
        elif name=='targets_r' and init:
            self.yumi.right_go_observe(cartesian=True)
        target_id = target_id*2 if name == 'targets_l' else target_id*2+1

        max_n_tryouts = 10
        rospy.sleep(0.5)
        # sanity check
        cursor = self.pm.left_cursor if name == 'targets_l' else self.pm.right_cursor
        target_id_visible = 0 if target_id==cursor else target_id
        if self.debug: target_id_visible=target_id
        for _ in range(max_n_tryouts):
            targets, confidence = self.call_vision_target(name)
            if not self.sanity_check_targets(targets, target_id_visible): # x and z should be ascending
                print('Result pose list does not seem to be correct. Retrying ...')
                rospy.sleep(0.1)
                continue
            elif compare_with is not None:
                if distance.euclidean(compare_with[:3], targets[target_id_visible][:3])<=0.01:
                    break
                else:
                    print('Camera depth faulty.')
                    continue
            else:
                break

        # check confidence and adjust observing position
        if fine and confidence[target_id_visible]<0.90: # first target confidence low
            print('Start Active Observation for target at {} with confidence {}!\
                  '.format(targets[target_id_visible], confidence[target_id_visible]))
            arm = 'left' if name=='targets_l' else 'right'
            base_to_target = compose_matrix(translate=targets[target_id_visible][:3], angles=targets[target_id_visible][3:])

            image_offset = [-0.17, -0.03, -0.05, 0, -pi/2, 0] if arm=='left' else [-0.17, -0.03, -0.05, 0, -pi/2, 0]
            waypoints = [tf_mat2ls(base_to_target@tf_ls2mat(image_offset))] # align target to image centre
            self.yumi.tip_go_thro(arm, waypoints, "Active Observe")
            rospy.sleep(0.5) # wait for auto white balance
            targets = self.get_target_poses(name,target_id=target_id,init=False)
        else:
            print('Target target: {}, confidence {}'.format(targets[target_id_visible], confidence[target_id_visible]))
        self.update_target_poses(targets[target_id_visible], target_id)
        self.pub_target_poses()
        return targets
    
    def get_scene_poses(self):
        ropes, img = self.call_scene_srv()
        for rope in ropes:
            self.pm.heirarchy.append(rope)
        self.pm.img_frame = img
        
        
    def update_marker_ownership(self, rope, marker, site):
        self.pm.rope_dict[rope].marker_dict['marker_at'] = site
        self.yumi.marker_at[marker] = site
        owner_msg = Int32MultiArray()
        owner_msg.data = [self.pm.sites_dict[self.pm.aglet_at['aglet_a']]-6, 
                        self.pm.sites_dict[self.pm.aglet_at['aglet_b']]-6] # left 0, right 1
        self.aglet_owner_pub.publish(owner_msg)

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
        targets_l = self.get_target_poses('targets_l') # left side
        targets_l = self.get_target_poses('targets_l', target_id=len(targets_l)//2, fine=True) # left side
        self.yumi.left_go_observe()
        targets_r = self.get_target_poses('targets_r') # right side
        targets_r = self.get_target_poses('targets_r', target_id=len(targets_r)//2, fine=True) # right side
        self.yumi.right_go_observe()
        assert len(targets_l) == len(targets_r), 'Found {} targets on the left and {} on the right.\
            '.format(len(targets_l), len(targets_r))
        self.pm.load_target_poses((targets_l, targets_r))

        self.pub_target_poses() # publish target poses
        self.yumi.both_go_grasp()
        
        
        for rope in self.rope_names_topic:
            self.get_rope_poses(rope, 'marker_a')
            self.get_rope_poses(rope, 'marker_b')

        self.get_scene_poses()
        
        
    def stop(self):
        self.yumi.stop()
        self.pm.save_params()


if __name__ == "__main__":
    rospy.init_node('sl_ctrl', anonymous=True)
    
  