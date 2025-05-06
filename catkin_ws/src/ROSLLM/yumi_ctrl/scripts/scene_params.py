#!/usr/bin/python3
import numpy as np
from os import path
from json import load
from yaml import safe_load, dump
from scipy.spatial import distance
from math import cos, sin, pi, sqrt
from tf.transformations import quaternion_from_matrix

from utils import ls_concat, ls_add, eval_list, tf_ls2mat, tf_mat2ls, pose_msg_to_list, is_sorted

class SceneParameters:
    
    class Rope:
        def __init__(self, rope_name, marker_a, marker_b):
            self.name = rope_name
            self.priority = 0
            self.marker_dict = {
                'marker_a': {
                    'colour': marker_a,
                    'pose': [0,0,0,0,0,0],
                    'marker_at': None
                },
                'marker_b': {
                    'colour': marker_b,
                    'pose': [0,0,0,0,0,0],
                    'marker_at': None
                }
                
            }
            self.marker_a_root = []
            self.marker_b_root = []
            self.mask = None
            self.target_l = None
            self.target_r = None
  
                      
                      
        def updateTarget(self, target_l, target_r):
            self.target_l = target_l
            self.target_r = target_r
            self.curr_target_l = target_l
            self.curr_target_r = target_r
            
        
    vel_scale = 2.5

    sites_dict = {"site_ul": 0, "site_ur": 1, "site_ll": 2, "site_lr": 3, "target_l1":4, "target_r1":5, "target_l2":6, "target_r2":7, "target_l3":8, "target_r3":9, 'left_gripper':10, 'right_gripper':11}
    gripper_dict = {'left_gripper':0, 'right_gripper':1}
    rope_dict = {
        'rope_r': Rope('rope_r', 'gray', 'cyan'),
        'rope_g': Rope( 'rope_g','orange', 'brown'),
        'rope_b': Rope( 'rope_b','yellow', 'pink')
    }
    # aglets_dict = {"aglet_a":0, "aglet_b":1}
    

    def __init__(self, reset, start_id, config_path, result_path, log_handle):
        self.config_path = config_path
        self.dynamic_param_path = path.join(config_path, 'dynamic_params.yaml')
        self.static_param_path = path.join(config_path, 'static_params.yaml')
        self.shoe_param_path = path.join(result_path, 'scene_params.yaml')
        self.read_static_params()
        self.read_dynamic_params(reset)
        self.left_cursor = start_id if start_id%2==0 else start_id-1
        self.right_cursor = start_id+1 if start_id%2==0 else start_id
        # self.aglet_poses = {"aglet_a":[0,0,0,0,0,0], "aglet_b":[0,0,0,0,0,0]}
        self.target_poses = []
        self.hand_poses = [[0,0,0,0,0,0], [0,0,0,0,0,0]]
        self.add_to_log = log_handle
        # self.aglet_at = {"aglet_a":"site_l1", "aglet_b":"site_r1"}

    def update_yumi_constriants(self):
        pass

    def load_target_poses(self, target_poses):
        (targets_l, targets_r) = target_poses
        self.n_rows = len(targets_l)
        self.n_targets = self.n_rows*2
        # cross check with shoe model
        targets_temp = []
        for i in range(self.n_rows):
            targets_temp.append(targets_l[i])
            targets_temp.append(targets_r[i])
        self.target_poses = np.array(targets_temp)
        # estimate mathematical shoe properties
        self.horizontal_gap = np.mean([distance.euclidean(b[:3],r[:3]) for b,r in zip(targets_l, targets_r)])
        self.rope_length_l += -self.horizontal_gap/2
        self.rope_length_r += -self.horizontal_gap/2
        self.vertical_gap = np.mean([distance.euclidean(targets_l[i][:3],targets_l[i-1][:3]) for i in range(self.n_rows)])
        self.marker_a_root = np.mean([self.target_poses[0][:3], self.target_poses[1][:3]], axis=0) # root of the red lace tip
        self.marker_b_root = self.marker_a_root # root of the blue lace tip
        # estimate target distances
        self.target_distances = np.zeros((self.n_targets, self.n_targets))
        for id1 in range(self.n_targets):
            for id2 in range(id1+1, self.n_targets):
                self.target_distances[id1, id2] = self.target_distances[id2, id1] = distance.euclidean(self.target_poses[id1][:3],self.target_poses[id2][:3])
        # update robot constraints
        self.update_yumi_constriants('marker_b', self.rope_length_l, self.get_root_position('marker_b'))
        self.update_yumi_constriants('marker_a', self.rope_length_r, self.get_root_position('marker_a'))
        # save the results
        self.save_scene_params()

    

    
    def update_section_availability(self, rope, marker, section):
        if rope == 'rope_r':
            rope_id = 0
        elif rope == 'rope_g':
            rope_id = 1
        elif rope == 'rope_b':
            rope_id = 2
        else:
            print('Unknown rope!')
            return
        if marker == 'marker_a':
            marker_id = 0
        elif marker == 'marker_b':
            marker_id = 1
        else:
            print('Unknown marker!')
            return
        self.sites_availabilty[:, ((rope_id * 2) + marker_id)] = 0
        if section:
            section_id = self.sites_dict[section]
            self.sites_availabilty[section_id, ((rope_id * 2) + marker_id)] = 1

    def check_section_availability(self, section):
        """
        Return true if available, false if not
        """
        section_id = self.sites_dict[section]
        return not np.any(self.sites_availabilty[section_id, :])

    def check_marker_location(self, rope, marker):
        """_summary_

        Args:
            rope (_type_): 'rope_r', 'rope_g', 'rope_b'
            marker (_type_): 'marker_a', 'marker_b'

        Returns:
            _type_: location of the marker or None if it is being held
        """
        if rope == 'rope_r':
            rope_id = 0
        elif rope == 'rope_g':
            rope_id = 1
        elif rope == 'rope_b':
            rope_id = 2
        else:
            print('Unknown rope!')
            return
        if marker == 'marker_a':
            marker_id = 0
        elif marker == 'marker_b':
            marker_id = 1
        else:
            print('Unknown marker!')
            return
        location = np.nonzero(self.sites_availabilty[:, ((rope_id * 2) + marker_id)])[0]
        if location.size == 0:
            return None # aglet is being held
        else:
            return location[0]



    def update_root_position(self, rope, marker, position):
        if self.rope_dict[rope]:
            rope_ = self.rope_dict[rope]
        else:
            print('Unknown rope!')
            return
        if marker == 'marker_b':
            rope_.marker_b_root = position # update the root position of the lace tip
            self.update_yumi_constriants(marker, self.rope_length_r, position)
        elif marker == 'marker_a':
            rope_.marker_a_root = position # update the root position of the lace tip
            self.update_yumi_constriants(marker, self.rope_length_l, position)
        else:
            print('Unknown aglet!')
        self.add_to_log('[Root update] '+marker+', '+str(position))

    def get_shoelace_length(self, marker):
        if marker == 'marker_b':
            return self.rope_length_r
        elif marker == 'marker_a':
            return self.rope_length_l
        else:
            print('Unknown marker!')

    def get_root_position(self, rope, marker):
        if self.rope_dict[rope]:
            rope_ = self.rope_dict[rope]
        else:
            print('Unknown rope!')
            return
        if marker == 'marker_b':
            return rope_.marker_b_root
        elif marker == 'marker_a':
            return rope_.marker_a_root
        else:
            print('Unknown marker!')



    

    def read_static_params(self):
        static_param_file = open(self.static_param_path, 'r')
        params = safe_load(static_param_file)
        static_param_file.close()

        self.workspace = params["workspace"]
        # measured parameters
        self.gp_os = eval(params["gp_os"])
        self.gp_tip_w = params["gp_tip_w"]
        self.app_os = params["app_os"]
        self.da_os_x = params["da_os_x"]
        self.da_os_z = params["da_os_z"]
        self.eyelet_to_edge = params["eyelet_to_edge"]
        self.eyelet_diameter = params["eyelet_diameter"]
        self.eyelet_radius = self.eyelet_diameter/2
        self.eyestay_thickness = params["eyestay_thickness"]
        self.eyestay_opening = params["eyestay_opening"]
        self.table_offset = params["table_offset"]
        self.scene_centre = np.array(params["scene_centre"])
        self.sl_length = params["sl_length"]
        self.aglet_thickness = params["aglet_thickness"]
        self.aglet_length = params["aglet_length"]

        # primitive parameters
        self.grasp_rot_l = eval_list(params["grasp_rot_l"])
        self.grasp_rot_r = eval_list(params["grasp_rot_r"])
        self.insert_pitch2 = eval(params["insert_pitch2"])

        self.hand_over_centre = ls_add(self.scene_centre, [-0.12, 0, -0.07]) # transfer
        self.hand_over_centre_2 = ls_add(self.scene_centre, [-0.05, 0, -0.05]) # adjusting orientation
        table_height = self.table_offset+0.02
        self.site_l1 = [0.38, 0.21, table_height] # centre of the left section a
        self.site_r1 = [0.38, -0.21, table_height] # centre of the right section a
        self.site_l2 = [0.38, 0.16, table_height] # centre of the left section b
        self.site_r2 = [0.38, -0.16, table_height] # centre of the right section b
        self.site_l3 = [0.53, 0.13, table_height] # centre of the left section c
        self.site_r3 = [0.53, -0.13, table_height] # centre of the right section c
        self.pre_grasp = ls_add(self.scene_centre, [0, 0, 0.1])
        self.pre_insert_l = [0.3, 0.2, 0.2]
        self.pre_insert_r = [0.3, -0.2, 0.2]
        self.observe_states = params['observe_states']
        self.grasp_states = params['grasp_states']
        self.grasp_states2 = params['grasp_states2']

    def read_dynamic_params(self, reset=True):
        dynamic_param_file = open(self.dynamic_param_path, 'r')
        self.dynamic_params = safe_load(dynamic_param_file)
        dynamic_param_file.close()
        if reset:
            self.eyelet_id = 0
            self.rope_length_r = self.sl_length/2
            self.rope_length_l = self.sl_length/2
            self.r_root = None # root of the red lace tip
            self.b_root = None # root of the blue lace tip
            self.sites_availabilty = np.zeros((len(self.sites_dict), len(self.rope_dict) * 2))
            self.sites_availabilty[0, 0] = 1 # assume red initially at left A
            self.sites_availabilty[1, 1] = 1 # assume blue initially at right A
        else:
            self.eyelet_id = self.dynamic_params["eyelet_id"]
            self.rope_length_r = self.dynamic_params["rope_length_r"]
            self.rope_length_l = self.dynamic_params["rope_length_l"]
            self.r_root = self.dynamic_params["r_root"]
            self.b_root = self.dynamic_params["b_root"]
            self.sites_availabilty = np.array(self.dynamic_params["sites_availabilty"])

        self.e_l_offset = np.array(self.dynamic_params["e_l_offset"]).tolist()
        self.e_r_offset = np.array(self.dynamic_params["e_r_offset"]).tolist()
        self.l_l_offset = np.array(self.dynamic_params["l_l_offset"]).tolist()
        self.l_r_offset = np.array(self.dynamic_params["l_r_offset"]).tolist()
        print('Parameters read from file.')

    def update_params(self):
        self.dynamic_params["rope_length_l"] = np.array(self.rope_length_l).tolist()
        self.dynamic_params["rope_length_r"] = np.array(self.rope_length_r).tolist()
        self.dynamic_params["r_root"] = np.array(self.r_root).tolist() if self.r_root is not None else [0]*3
        self.dynamic_params["b_root"] = np.array(self.b_root).tolist() if self.b_root is not None else [0]*3
        self.dynamic_params["sites_availabilty"] = np.array(self.sites_availabilty).tolist()

    def save_params(self):
        self.update_params()
        dynamic_param_file = open(self.dynamic_param_path, 'w')
        dump(self.dynamic_params, dynamic_param_file, default_flow_style=None)
        dynamic_param_file.close()
        print("Parameter saved!")

    def save_scene_params(self):
        content = {
            'num_targets': self.n_targets,
            'targets': np.array(self.target_poses).tolist(),
            'H': np.array(self.horizontal_gap).tolist(),
            'V': np.array(self.vertical_gap).tolist()
        }
        shoe_param_file = open(self.shoe_param_path, 'w')
        dump(content, shoe_param_file, default_flow_style=None)
        shoe_param_file.close()
        print('Shoe parameters saved')
        
    # UNUSED FUNCTIONS
    
    # def check_flip(self, start_id):
    #     final_id = (self.n_eyelets-1-start_id)//2*2+start_id
    #     # check if the last same column eyelet is farther than 0.03m
    #     return distance.euclidean(self.eyelet_poses[start_id][:3], self.eyelet_poses[final_id][:3])>0.03  

    # def get_flip_id(self, start_id):
    #     flip_id = start_id+2
    #     while flip_id < self.n_eyelets-1:
    #         if distance.euclidean(self.eyelet_poses[start_id][:3], self.eyelet_poses[flip_id][:3])>0.03:
    #             return flip_id
    #         flip_id+=2
    #     else: 
    #         print('Unable to find proper eyelet to flip.')
    #         return
    
    
    # def get_the_other_aglet(self, aglet):
    #     aglet_list = list(self.aglets_dict.keys())
    #     aglet_list.remove(aglet)
    #     return aglet_list.pop()
    
    # def update_shoelace_length(self, aglet, difference):
    #     if aglet == 'aglet_b':
    #         self.rope_length_l += difference
    #         self.update_yumi_constriants('aglet_b', self.rope_length_l, self.get_root_position('aglet_b'))
    #     elif aglet == 'aglet_a':
    #         self.rope_length_r += difference
    #         self.update_yumi_constriants('aglet_a', self.rope_length_r, self.get_root_position('aglet_a'))
    #     else:
    #         print('Unknown aglet!')
    #     self.save_params()
    #     self.add_to_log('[Length update] '+str(self.rope_length_r)+', '+str(self.rope_length_l))
    
    # @staticmethod
    # def eyelet_fit_to_shoe(src, tar):
    #     '''
    #     src, tar: Nx3
    #     '''
    #     src = np.array(src)
    #     tar = np.array(tar)
    #     src_ctr = np.mean(src, axis=0)
    #     tar_ctr = np.mean(tar, axis=0)

    #     src -= src_ctr
    #     tar -= tar_ctr

    #     a = np.matmul(tar.T, src)
    #     U,S,Vh = np.linalg.svd(a)
    #     R = np.matmul(U,Vh)
    #     t = tar_ctr - np.matmul(R, src_ctr)
        
    #     eyelets_registered = []
    #     error = 0
    #     for i in range(len(src)):
    #         eyelets_registered.append(R @ src[i] + t)
    #         error += tar - eyelets_registered[i]
    #     from scipy.spatial.transform import Rotation
    #     R = Rotation.from_matrix(R)
    #     print("Fit to shoe model error: ", error)
    #     return R.as_quat(), t, np.array(eyelets_registered), error