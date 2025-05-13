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
            self.completion = False
                      
                      
            
        
    vel_scale = 2.5

    sites_dict = {"site_uu": -1, "site_ul": -1, "site_ur": 1, "site_dd": 1, "site_dl": -1, "site_dr": 1, "target_l1":-1, "target_l2":-1, "target_l3":-1, "target_r1": 1, "target_r2":1,"target_r3":1, 'left_gripper':-2, 'right_gripper':2}
    
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
        # self.left_cursor = start_id if start_id%2==0 else start_id-1
        # self.right_cursor = start_id+1 if start_id%2==0 else start_id
        # self.aglet_poses = {"aglet_a":[0,0,0,0,0,0], "aglet_b":[0,0,0,0,0,0]}
        self.target_poses = []
        self.hand_poses = [[0,0,0,0,0,0], [0,0,0,0,0,0]]
        self.add_to_log = log_handle
        # self.aglet_at = {"aglet_a":"site_l1", "aglet_b":"site_r1"}
        self.site_occupency = {}
        self.heirarchy = []
        self.img_frame = None
        self.site_poses = {}
        
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
            self.site_poses['target_l'+str(i+1)] = targets_l[i]
            self.site_poses['target_r'+str(i+1)] = targets_r[i]
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

    
    def calc_optimal_site(self, rope, marker):
        """
        Calculate the optimal section for the given marker
        """
        if marker == 'marker_a':
            marker_alt = 'marker_b'
        elif marker == 'marker_b':
            marker_alt = 'marker_a'
        else:
            print('Unknown marker!')
            return
        marker_alt_location = self.check_marker_location(rope, marker_alt)
        marker_location = self.check_marker_location(rope, marker)
        
    def update_site_occupency(self, rope, marker, site, add_site=True):
        if add_site:
            if site in self.site_occupancy:
                print(f"Site '{site}' already occupied by {self.site_occupancy[site]}")
                return None
            rope_ = self.rope_dict[rope]
            rope_.marker_dict[marker]['marker_at'] = site
            self.site_occupancy[site] = (rope, marker)
        else:
            """Clears the site the marker is located at."""
            rope_ = self.rope_dict[rope]
            site = rope.marker_dict[marker]['marker_at']
            if site and self.site_occupancy.get(site) == (rope, marker):
                del self.site_occupancy[site]
            rope_.marker_dict[marker]['marker_at'] = None
            
    def get_marker_at_site(self, site):
        """Returns (rope_name, marker_name) or None."""
        return self.site_occupancy.get(site)

    def check_marker_location(self, rope, marker):
        """Returns site_name or None if being held."""
        return self.rope_dict[rope].marker_dict[marker]['marker_at']
            

    def update_heirarchy(self):
        return
        
    def check_site_availability(self, site):
        """
        Return true if available, false if not
        """
        if site in self.site_occupency:
            return False
        else:
            return True





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

    def get_shoelace_length(self, rope, marker):
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


    def find_closest_site(self, rope, marker, tol=0.1):
        """
        Find the closest site to the given marker
        """
        marker_location = self.rope_dict[rope].marker_dict[marker]['pose'][:2] # get the 2D position
        closest_site = None
        closest_distance = float('inf')
        for site, site_location in self.site_poses.items():
            if site in self.site_occupency:
                continue
            distance = np.linalg.norm(np.array(marker_location) - np.array(site_location[:2]))
            if distance < closest_distance and distance < tol:
                closest_distance = distance
                closest_site = site
        if closest_site is not None:
            self.site_occupency[closest_site] = (rope, marker)
            self.rope_dict[rope].marker_dict[marker]['marker_at'] = closest_site
            return closest_site
        else:
            print(f"No available site found for {marker} on {rope} within tolerance {tol}.")
            return None
        
    def get_target_id(self, target):
        side = self.sites_dict[target]
        if side == -1:
            side = 0
            group = 'targets_l'
        else:
            side = 1
            group = 'targets_r'
        target_number = int(target[-1]) - 1
        target_id = (target_number * 2) + side
        return group, target_id
        

        
        
    def init_marker_sites(self):
        """
        Initialize the marker sites
        """
        for rope in self.rope_dict:
            for marker in self.rope_dict[rope].marker_dict:
                estimated_site = self.find_closest_site(rope, marker)
                if estimated_site is not None:
                    self.update_site_occupency(rope, marker, estimated_site)
                else:
                    print(f"Failed to find initial site for {marker} on {rope}.")
                
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
        self.target_to_edge = params["target_to_edge"]
        self.target_diameter = params["target_diameter"]
        self.target_radius = self.target_diameter/2
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
       
        self.site_poses['site_dl'] = [0.38, 0.16, table_height] # centre of the left section a
        self.site_poses['site_dd'] = [0.38, 0, table_height] # centre of the left section a
        self.site_poses['site_dr'] = [0.38, -0.16, table_height] # centre of the right section b
        self.site_poses['site_ul'] = [0.53, 0.13, table_height] # centre of the left section c
        self.site_poses['site_ur'] = [0.53, -0.13, table_height] # centre of the right section c
        self.site_poses['site_uu'] = [0.53, 0, table_height] # centre of the left section d
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
            self.target_id = 0
            self.rope_length_r = self.sl_length/2
            self.rope_length_l = self.sl_length/2
            self.r_root = None # root of the red lace tip
            self.b_root = None # root of the blue lace tip
            self.sites_availabilty = np.zeros((len(self.sites_dict), len(self.rope_dict) * 2))
            # self.sites_availabilty[0, 0] = 1 # assume red initially at left A
            # self.sites_availabilty[1, 1] = 1 # assume blue initially at right A
        else:
            self.target_id = self.dynamic_params["target_id"]
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
        
    