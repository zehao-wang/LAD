'''
Instruction and trajectory (view and object features) dataset
'''
import os
import json
import jsonlines
import numpy as np
import h5py
import math

from .common import load_nav_graphs
from .common import get_angle_fts, get_view_rel_angles
from .common import calculate_vp_rel_pos_fts
from .common import softmax
from ipdb import set_trace

MAX_DIST = 30 # normalize
MAX_STEP = 10 # normalize
TRAIN_MAX_STEP = 20 

class ReverieTextPathData(object):
    def __init__(
        self, anno_files, vp2room_file, img_ft_file, obj_ft_file, scanvp_cands_file, connectivity_dir,
        clip_ft_file, ins2img_file=None, image_feat_size=2048, image_prob_size=1000, angle_feat_size=4,
        obj_feat_size=None, obj_prob_size=None, max_objects=20,
        max_txt_len=100, in_memory=True, act_visited_node=False,
        use_real_dist = False, 
    ):
        self.img_ft_file = img_ft_file
        self.obj_ft_file = obj_ft_file
        self.vp2room_file = vp2room_file
        self.clip_ft_file = clip_ft_file
        self.ins2img_file = ins2img_file
        
        self.image_feat_size = image_feat_size
        self.image_prob_size = image_prob_size
        self.angle_feat_size = angle_feat_size
        self.obj_feat_size = obj_feat_size
        self.obj_prob_size = obj_prob_size

        self.obj_image_h = 480
        self.obj_image_w = 640 
        self.obj_image_size = 480 * 640 

        self.max_txt_len = max_txt_len 
        self.max_objects = max_objects 
        self.act_visited_node = act_visited_node
        
        self.use_real_dist = use_real_dist
        self.in_memory = in_memory
        if self.in_memory:
            self._feature_store = {}
            self._ins2img_feat_store = {}

        with open(self.vp2room_file, "r") as f:
            self.vp2room = json.load(f)["pano_info"]
        
        #{scan_vp: {vp: [viweidx, rel_angel_dist, rel_heading, rel_elevation]}}
        self.scanvp_cands = json.load(open(scanvp_cands_file))

        self.graphs, self.shortest_distances, self.shortest_paths = load_nav_graphs(connectivity_dir)
        self.all_point_rel_angles = [get_view_rel_angles(baseViewId=i) for i in range(36)]
        self.all_point_angle_fts = [get_angle_fts(x[:, 0], x[:, 1], self.angle_feat_size) for x in self.all_point_rel_angles]

        self.data = []
        
        self.use_aug_train_data = False 
        for file_name in anno_files:
            if "aug" in file_name:
                self.use_aug_train_data  = True
        
        for anno_file in anno_files:
            with jsonlines.open(anno_file, "r") as f:
                for item in f:
                    self.data.append(item)

    def __len__(self):
        return len(self.data)
    
    def _cal_score(self, curr_vid, goal_vid, gt_path, st_path, power=2):
        gt_len = len(gt_path) - 1
        local_len = len(st_path[curr_vid][goal_vid]) - 1
        if gt_len == 0:
            if goal_vid == curr_vid:
                return 1  
            return 0
        y = max(np.power((gt_len-local_len)/gt_len, power), 0)
        return y 
    
    def _cal_score_real(self, curr_vid, goal_vid, gt_path, st_path, power=2):
        dist2goal = st_path[curr_vid][goal_vid]
        full_len = st_path[gt_path[0]][goal_vid]
        if full_len == 0:
            if goal_vid == curr_vid:
                return 1 
            return 0
        y = max(np.power((full_len-dist2goal) / full_len, power), 0)
        return y 
    
    def get_scanvp_rt(self, scan, gmap_vpids):
        rt_label = []
        for vp in gmap_vpids:
            if vp is not None: # remove the first point
                rt_label.append(self.vp2room[scan][vp]["label_index"])
        return np.array(rt_label)
    
    
    def get_ins2img_feat(self, instr_id):
        key = instr_id 
        if self.in_memory and key in self._ins2img_feat_store:
            ft = self._ins2img_feat_store[key]
        else:
            with h5py.File(self.ins2img_file,"r") as f:
                ft = f[key][...][:, :self.image_feat_size].astype(np.float32)
                self._ins2img_feat_store[key] = ft 
        return ft 
    
    def get_scanvp_feature(self, scan, viewpoint):
        key = '%s_%s' % (scan, viewpoint)
        if self.in_memory and key in self._feature_store:
            view_fts, obj_fts, obj_attrs = self._feature_store[key]
        else:
            with h5py.File(self.img_ft_file, 'r') as f:
                view_fts = f[key][...].astype(np.float32)

            obj_attrs = {}
            obj_fts = np.zeros((0, self.obj_feat_size+self.obj_prob_size), dtype=np.float32)
            if self.obj_ft_file is not None:
                with h5py.File(self.obj_ft_file, 'r') as f:
                    if key in f:
                        obj_fts = f[key][...].astype(np.float32)
                        obj_fts = obj_fts[:self.max_objects]
                        for attr_key, attr_value in f[key].attrs.items():
                            if attr_key in ['directions', 'sizes', 'bboxes', 'obj_ids']:
                                obj_attrs[attr_key] = attr_value[:self.max_objects]
            
            
            if len(self.clip_ft_file) != 0 :
                with h5py.File(self.clip_ft_file[0], 'r') as f:
                    clip_view_fts = f[key][...].astype(np.float32)
                feat_size = clip_view_fts.shape[1]
                view_fts[:,:feat_size] = clip_view_fts

            if self.in_memory:
                self._feature_store[key] = (view_fts, obj_fts, obj_attrs)

        return view_fts, obj_fts, obj_attrs
    
    def get_obj_label(self, item, last_vp_objids):
        gt_obj_id = item['instr_id'].split('_')[1]
        for k, obj_id in enumerate(last_vp_objids):
            if obj_id == gt_obj_id:
                obj_label = k
                break
        else:
            # it occurs when the gt_objid is not in max_objects
            obj_label = -100 # ignore 
        return obj_label
    
    def get_act_labels(self, end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids):
        scan = item['scan']
        pos_vps = item['pos_vps']
        if end_vp in pos_vps:
            global_act_label = local_act_label = 0 # reach destination -> stop
        else:
            global_act_label = local_act_label = -100
            # global: unvisited vp
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(gmap_vpids):
                if (k > 0) and (not gmap_visited_masks[k]):
                    min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                        + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                    if min_dist < cand_min_dist:
                        cand_min_dist = min_dist
                        global_act_label = k # [stop] is 0
            # local: 
            cand_min_dist = float('inf')
            for k, cand_vp in enumerate(traj_cand_vpids[-1]):
                min_dist = min([self.shortest_distances[scan][end_vp][cand_vp] \
                    + self.shortest_distances[scan][cand_vp][pos_vp] for pos_vp in pos_vps])
                if min_dist < cand_min_dist:
                    cand_min_dist = min_dist
                    local_act_label = k + 1 # [stop] is 0
        return global_act_label, local_act_label
    
    def get_gmap_vpid_node_dist(self,gmap_vpids,end_vp, gt_ep_end_vp, gt_path, scan):
        step_score_to_goals = []
        if self.use_real_dist:
            #step_score_to_goals = [self._cal_score_real(end_vp,gt_ep_end_vp,gt_path,self.shortest_distances[scan])]
            # first point is the current observation 
            for vp in gmap_vpids[1:]:
                score = self._cal_score_real(vp, gt_ep_end_vp, gt_path, self.shortest_distances[scan])
                step_score_to_goals.append(score)
        else:
            # the first element is not used during node dist prediction task 
            #step_score_to_goals = [self._cal_score(end_vp,gt_ep_end_vp,gt_path,self.shortest_distances[scan])]
            # first point is the current observation 
            for vp in gmap_vpids[1:]:
                score = self._cal_score(vp, gt_ep_end_vp, gt_path, self.shortest_paths[scan])
                step_score_to_goals.append(score)
        return np.array(step_score_to_goals)


    def get_input(
        self, idx, end_vp_type, return_img_probs=False, return_act_label=False, 
        return_obj_label=False, return_rt_label=False, end_vp=None, return_node_dist=False,
    ):
        item = self.data[idx]
        scan = item['scan']
        start_vp = item['path'][0]
        start_heading = item.get('heading', 0)
        pos_vps = item['pos_vps'] # vp of end postition
        gt_path = item['path']
        
        gt_ep_end_vp = item['path'][-1]
        if end_vp is None:  # get end position vp bsed on tasks  
            # the end vp is the end of current path 
            if end_vp_type == 'pos':   # at end point
                end_vp = pos_vps[np.random.randint(len(pos_vps))]
                gt_ep_end_vp = end_vp
            elif end_vp_type == 'neg_in_gt_path':  # at other position in the path 
                end_vps = [vp for vp in gt_path if vp not in pos_vps]
                if len(end_vps) == 0:
                    end_vps = gt_path
                end_vp = end_vps[np.random.randint(len(end_vps))]
            elif end_vp_type == 'neg_others':    # at other positions not in the path
                noneg_vp_set = set(pos_vps + gt_path)
                end_vps = [vp for vp in self.graphs[scan].nodes.keys() if vp not in noneg_vp_set]
                end_vp = end_vps[np.random.randint(len(end_vps))]
        
        gt_path = self.shortest_paths[scan][start_vp][end_vp]
        cur_heading, cur_elevation = self.get_cur_angle(scan, gt_path, start_heading)
        # the next to last heading and elevation

        if len(gt_path) > TRAIN_MAX_STEP:
            # truncate trajectory
            gt_path = gt_path[:TRAIN_MAX_STEP] + [end_vp]
            
        # traj features of all path
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
            last_vp_angles, last_vp_objids = self.get_traj_pano_fts(scan, gt_path)

        # global: the first token is [stop]
        gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists = \
            self.get_gmap_inputs(scan, gt_path, cur_heading, cur_elevation)
       
        # local: the first token is [stop]
        vp_pos_fts = self.get_vp_pos_fts(scan, start_vp, end_vp,
            traj_cand_vpids[-1], cur_heading, cur_elevation, len(traj_nav_types[-1]))

        if self.use_aug_train_data:
            ins2img_features = None #self.get_ins2img_feat(item['instr_id'])
        else:
            ins2img_features = self.get_ins2img_feat(item['instr_id']).squeeze()
        
        outs = {
            'instr_id': item['instr_id'],
            'instr_encoding': item['instr_encoding'][:self.max_txt_len],
            'traj_view_img_fts': [x[:, :self.image_feat_size] for x in traj_view_img_fts],
            'traj_obj_img_fts': [x[:, :self.obj_feat_size] for x in traj_obj_img_fts],
            'traj_loc_fts': traj_loc_fts,
            'traj_nav_types': traj_nav_types,
            'traj_cand_vpids': traj_cand_vpids,
            'traj_vpids': gt_path,
            'gmap_vpids': gmap_vpids,
            'gmap_step_ids': gmap_step_ids,
            'gmap_visited_masks': gmap_visited_masks,
            'gmap_pos_fts': gmap_pos_fts,
            'gmap_pair_dists': gmap_pair_dists,
            'vp_pos_fts': vp_pos_fts,
            'vp_angles': last_vp_angles,
            'ins2img_feat': ins2img_features,
            'current_vpid': end_vp,  # end point of current path
            'current_vpid_index': gmap_vpids.index(end_vp)
        }

        if return_obj_label:
            outs['obj_labels'] = self.get_obj_label(item, last_vp_objids)

        if return_act_label:
            global_act_label, local_act_label = self.get_act_labels(
                end_vp, item, gmap_vpids, gmap_visited_masks, traj_cand_vpids
            )
            outs['global_act_labels'] = global_act_label
            outs['local_act_labels'] = local_act_label
        
        if return_img_probs:
            # TODO: whether adding gmap img probs  softmax -> wired CHECK 
            outs['vp_view_probs'] = softmax(traj_view_img_fts[-1][:, self.image_feat_size:], dim=1)
            outs['vp_obj_probs'] = softmax(traj_obj_img_fts[-1][:, self.obj_feat_size:], dim=1)

        if return_rt_label:
            outs['vp2room_label'] = self.get_scanvp_rt(scan, gmap_vpids)
            # gmap_vpids -> all view point ids in the map 
            # traj_cand_vpids -> cand view point ids in each location (after set = all view points in traj)
            # gt_path -> view point id in the path 
            # traj_view_img_fts --> 36* 768 at each location 
            # visited node feats = traj_view_img_fts mean  , unvisited node feats traj_view_img_fts[i]
        if return_node_dist:
            outs['step_score_to_goals'] = self.get_gmap_vpid_node_dist(gmap_vpids, end_vp, gt_ep_end_vp, item['path'], scan) 
        return outs


    def get_cur_angle(self, scan, path, start_heading):
        if len(path) < 2:
            heading = start_heading
            elevation = 0
        else:
            prev_vp = path[-2]
            cur_vp = path[-1]
            viewidx = self.scanvp_cands['%s_%s'%(scan, prev_vp)][cur_vp][0]
            heading = (viewidx % 12) * math.radians(30)
            elevation = (viewidx // 12 - 1) * math.radians(30)
            # at the last two posiiton heading to end position 
        return heading, elevation
    

    def get_traj_pano_fts(self, scan, path):
        '''
        Tokens in each pano: [cand_views, noncand_views, objs]
        Each token consists of (img_fts, loc_fts (ang_fts, box_fts), nav_types)
        '''
        traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids = [], [], [], [], []
        for vp in path:
            view_fts, obj_img_fts, obj_attrs = self.get_scanvp_feature(scan, vp)

            view_img_fts, view_angles, cand_vpids = [], [], []
            # cand views

            #{scan_vp: {vp: [viweidx, rel_angel_dist, rel_heading, rel_elevation]}}
            nav_cands = self.scanvp_cands['%s_%s'%(scan, vp)]
            used_viewidxs = set()
            
            for k, v in nav_cands.items():
                used_viewidxs.add(v[0]) # v[0] -> view index 0~36
                view_img_fts.append(view_fts[v[0]])
                # TODO: whether using correct heading at each step
                view_angle = self.all_point_rel_angles[12][v[0]]
                view_angles.append([view_angle[0] + v[2], view_angle[1] + v[3]])
                cand_vpids.append(k)
            # non cand views

            view_img_fts.extend([view_fts[idx] for idx in range(36) if idx not in used_viewidxs])
            view_angles.extend([self.all_point_rel_angles[12][idx] for idx in range(36) if idx not in used_viewidxs])
           
            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)    # (n_views, dim_ft)
            view_angles = np.stack(view_angles, 0)
            view_ang_fts = get_angle_fts(view_angles[:, 0], view_angles[:, 1], self.angle_feat_size)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            
            # object features
            num_objs = obj_img_fts.shape[0]
            obj_angles = np.zeros((num_objs, 2), dtype=np.float32)
            obj_ang_fts = np.zeros((num_objs, self.angle_feat_size), dtype=np.float32)
            obj_box_fts = np.zeros((num_objs, 3), dtype=np.float32)
            if num_objs > 0:
                for k, (w, h) in enumerate(obj_attrs['sizes']):
                    obj_angles[k] = obj_attrs['directions'][k]
                    obj_box_fts[k] = [h/self.obj_image_h, w/self.obj_image_w, (h*w)/self.obj_image_size]           
                obj_ang_fts = get_angle_fts(obj_angles[:, 0], obj_angles[:, 1], self.angle_feat_size)

            # combine pano features
            traj_view_img_fts.append(view_img_fts)
            traj_obj_img_fts.append(obj_img_fts)
          
            traj_loc_fts.append(
                np.concatenate(
                    [np.concatenate([view_ang_fts, view_box_fts], 1),
                     np.concatenate([obj_ang_fts, obj_box_fts], 1)], axis=0
                )
            )
            # 
            traj_nav_types.append(
                [1] * len(cand_vpids) + [0] * (36 - len(used_viewidxs)) + [2] * len(obj_img_fts)
            )
            traj_cand_vpids.append(cand_vpids)

            last_vp_objids = obj_attrs.get('obj_ids', [])
            last_vp_angles = np.concatenate([view_angles, obj_angles], 0)

        return traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, traj_cand_vpids, \
               last_vp_angles, last_vp_objids
    

    def get_gmap_inputs(self, scan, path, cur_heading, cur_elevation):
        scan_graph = self.graphs[scan]
        cur_vp = path[-1]
        
        visited_vpids, unvisited_vpids = {}, {}
        for t, vp in enumerate(path):
            visited_vpids[vp] = t + 1
            if vp in unvisited_vpids:
                del unvisited_vpids[vp]
            for next_vp in self.scanvp_cands['%s_%s'%(scan, vp)].keys():
                if next_vp not in visited_vpids:
                    unvisited_vpids[next_vp] = 0
        # add [stop] token
        gmap_vpids = [None] + list(visited_vpids.keys()) + list(unvisited_vpids.keys())
        gmap_step_ids = [0] + list(visited_vpids.values()) + list(unvisited_vpids.values())
        if self.act_visited_node: # default -> False
            gmap_visited_masks = [0]
            for vp in gmap_vpids[1:]:
                if vp == path[-1]:
                    gmap_visited_masks.append(1)
                else:
                    gmap_visited_masks.append(0)
        else:
            gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)

        # shape=(num_gmap_vpids, 7)
        gmap_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, gmap_vpids, cur_heading, cur_elevation)
        
        gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
        for i in range(1, len(gmap_vpids)):
            for j in range(i+1, len(gmap_vpids)):
                gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                    self.shortest_distances[scan][gmap_vpids[i]][gmap_vpids[j]]

        return gmap_vpids, gmap_step_ids, gmap_visited_masks, gmap_pos_fts, gmap_pair_dists
    

    def get_gmap_pos_fts(self, scan, cur_vp, gmap_vpids, cur_heading, cur_elevation):
        # dim=7 (sin(heading), cos(heading), sin(elevation), cos(elevation),
        #  line_dist, shortest_dist, shortest_step)
        rel_angles, rel_dists = [], []
        for vp in gmap_vpids:
            if vp is None:
                rel_angles.append([0, 0])
                rel_dists.append([0, 0, 0])
            else:
                rel_heading, rel_elevation, rel_dist = calculate_vp_rel_pos_fts(
                    self.graphs[scan].nodes[cur_vp]['position'], 
                    self.graphs[scan].nodes[vp]['position'],
                    base_heading=cur_heading, base_elevation=cur_elevation,
                )
                rel_angles.append([rel_heading, rel_elevation])
                rel_dists.append(
                    [rel_dist / MAX_DIST, self.shortest_distances[scan][cur_vp][vp] / MAX_DIST, \
                    (len(self.shortest_paths[scan][cur_vp][vp]) - 1) / MAX_STEP]
                )
        rel_angles = np.array(rel_angles).astype(np.float32)
        rel_dists = np.array(rel_dists).astype(np.float32)
        rel_ang_fts = get_angle_fts(rel_angles[:, 0], rel_angles[:, 1], self.angle_feat_size)
        return np.concatenate([rel_ang_fts, rel_dists], 1)
    
    
    def get_vp_pos_fts(self, scan, start_vp, cur_vp, cand_vpids, cur_heading, cur_elevation, vp_ft_len):
        cur_cand_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, cand_vpids, cur_heading, cur_elevation)
        cur_start_pos_fts = self.get_gmap_pos_fts(scan, cur_vp, [start_vp], cur_heading, cur_elevation)
                
        # add [stop] token at beginning
        vp_pos_fts = np.zeros((vp_ft_len+1, 14), dtype=np.float32)
        vp_pos_fts[:, :7] = cur_start_pos_fts
        vp_pos_fts[1:len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts

        return vp_pos_fts
