import enum
import imp
from importlib import import_module
import json 
import os 
import numpy as np
import math 
import random 
import networkx as nx 
from collections import defaultdict
import copy 
import MatterSim 

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature
from env_bases.reverie.env_base import EnvBatch

class ReverNavBatchEnv(object):
    '''Rever nav env including room label and pretrained visual feats'''

    def __init__(
        self, view_db, obj_db, instr_data, connectivity_dir, obj2vps,
        vp2room, ins2img_db=None, multi_endpoints = False, multi_startpoints = False,
        batch_size = 64, angle_feat_size = 4, max_objects = None, seed = 0,
        name = None, sel_data_idxs = None, args=None
    ):
        self.args = args
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.obj_db = obj_db 
        self.ins2img_db = ins2img_db
        self.data = instr_data 
        self.scans = set(x['scan'] for x in self.data)
        self.multi_endpoints = multi_endpoints
        self.multi_startpoints = multi_startpoints
        self.obj2vps = obj2vps
    
        self.vp2room = vp2room["vp2room"] #{scan_vp: {room_label, id}}
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.max_objects = max_objects
        self.name = name 

        for item in self.data:
            if 'objId' in item and item['objId'] is not None:
                # end vps -> all view points that can see obj
                item['end_vps'] = self.obj2vps['%s_%s'%(item['scan'], item['objId'])]
        
        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation
        
        # in validate -> we would split the data  
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None 
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx : end_idx]
        
        self.seed = seed 
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self._load_nav_grpahs()

        self.sim = new_simulator(self.connectivity_dir) # one simulator
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def size(self):
        return len(self.data)
    
    def _next_minibatch(self, batch_size = None, **kwargs):
        """
        Store the minibatch in 'self.batch'
        """ 
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix : self.ix + batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch 

        start_vps = [x['path'][0] for x in self.batch]
        end_vps = [x['path'][-1] for x in self.batch]

        if self.multi_startpoints:
            for i, item in enumerate(batch):
                cand_vps = []
                for cvp, cpath in self.shortest_paths[item['scan']][end_vps[i]].items():
                    if len(cpath) > 4 and len(cpath) < 7:
                        cand_vps.append(cvp)
                    if len(cand_vps) > 0:
                        start_vps[i] = cand_vps[np.random.randint(len(cand_vps))]
        
        if self.multi_endpoints:
            for i,item in enumerate(batch):
                end_vp = item['end_vps'][np.random.randint(len(item['end_vps']))]
                end_vps[i] = end_vp 
        
        if self.multi_startpoints or self.multi_endpoints:
            batch = copy.deepcopy(self.batch)
            for i, item in enumerate(batch):
                item['path'] = self.shortest_paths[item['scan']][start_vps[i]][end_vps[i]]
            self.batch = batch 
    

    def reset_epoch(self, shuffle=False):
        '''Reset the data index to begining of epoch. Primarily for testing.
           You must still call reset() for a new episode.
        '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0 
    
    
    def make_candidate(self, features, scanId, viewpointId, viewId, gt_path=None):

        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)

        base_heading  = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 -1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])
                
                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = features[ix]
            
                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visibale from multiple view use the closed
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)
                    # Heading and elevation for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)

                    if self.args.use_real_dist_norm:
                        step_score_to_goal = self._cal_score_real(loc.viewpointId, gt_path[-1],
                                                    gt_path ,self.shortest_distances[state.scanId])
                    else:
                        step_score_to_goal = self._cal_score(loc.viewpointId, gt_path[-1],
                                                    gt_path, self.shortest_paths[state.scanId])
 
                    #print(f"from {loc.viewpointId} to {gt_path[-1]} {step_score_to_goal}")
                    if (loc.viewpointId not in adj_dict or distance < adj_dict[loc.viewpointId]['distance'] ) :
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            'normalized_heading': state.heading + loc.rel_heading,
                            'normalized_elevation': state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId,
                            'pointId': ix, 
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), - 1),
                            'position': (loc.x, loc.y, loc.z),
                            'room_type': self.vp2room[scanId+"_"+loc.viewpointId]["id"],
                            'step_score_to_goal': step_score_to_goal,
                        }

            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key] for key in ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId', 'pointId', 'idx', 'position', 'room_type', 'step_score_to_goal']}
                for c in candidate
            ]
            return candidate 
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = features[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new
    
    
    def _get_obs(self):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            scanId = state.scanId
            viewpointId = state.location.viewpointId
        
            # Full features
            candidate = self.make_candidate(feature, state.scanId, 
                                            state.location.viewpointId, 
                                            state.viewIndex,item['path'])

            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            
            # objects
            obj_img_fts, obj_ang_fts, obj_box_fts, obj_ids = self.obj_db.get_object_feature(
                state.scanId, state.location.viewpointId,
                state.heading, state.elevation, self.angle_feat_size,
                max_objects = self.max_objects
            )
            ins2img_feat = self.ins2img_db.get_ins2image_feature(item['instr_id'])
            #print("Path ", item['path'])
            if self.args.use_real_dist_norm:
                step_score_to_goal = self._cal_score_real(state.location.viewpointId, item['path'][-1],
                                            item['path'],self.shortest_distances[state.scanId])
            else:
                step_score_to_goal = self._cal_score(state.location.viewpointId, item['path'][-1],
                                            item['path'],self.shortest_paths[state.scanId])

            ob = {
                "instr_id": item['instr_id'],
                'scan': state.scanId,
                'viewpoint': state.location.viewpointId,
                'viewIndex': state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading': state.heading,
                'elevation': state.elevation,
                'feature': feature,
                'ins2img_feat':ins2img_feat, 
                'candidate': candidate,
                'obj_img_fts': obj_img_fts,
                'obj_ang_fts': obj_ang_fts,
                'obj_box_fts': obj_box_fts,
                'obj_ids': obj_ids,
                'navigableLocations': state.navigableLocations,
                'instruction': item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path': item['path'],
                'gt_end_vps': item.get('end_vps', []),
                'gt_obj_id': item['objId'],
                'path_id': item['path_id'],
                'room_type': self.vp2room[scanId+"_"+viewpointId]["id"],
                'step_score_to_goal': step_score_to_goal, 
            }
            
            # RL reward. The negative distance between the state and the final state 
            # There are multiple gt end viewpoints on REVERIE.
            if ob['instr_id'] in self.gt_trajs:
                gt_objid = self.gt_trajs[ob['instr_id']][-1]
                min_dist = np.inf
                for vp in self.obj2vps['%s_%s'%(ob['scan'], str(gt_objid))]:
                    try:
                        min_dist = min(min_dist, self.shortest_distances[ob['scan']][ob['viewpoint']][vp])
                    except:
                        print(ob['scan'], ob['viewpoint'], vp)
                        exit(0)
                ob['distance'] = min_dist
            else:
                ob['distance'] = 0
            obs.append(ob)
        return obs 
    
    
    def reset(self, **kwargs):
        '''Load a new minibatch / episodes.'''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()
    

    def step(self, actions):
        '''Take action (same interface as make Actions)'''
        self.env.makeActions(actions)
        return self._get_obs()


    def _load_nav_grpahs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print("Loading navigation graph for %d scans" % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))
         
    def _cal_score(self, curr_vid, goal_vid, gt_path, st_path, power=5):
        gt_len = len(gt_path) - 1
        local_len = len(st_path[curr_vid][goal_vid]) -1
        if gt_len == 0:
            if goal_vid == curr_vid:
                return 1
            return 0
        # print(gt_len, local_len)
        y = max(np.power((gt_len-local_len)/gt_len, power), 0)
        return y

    def _cal_score_real(self, curr_vid, goal_vid, gt_path, st_path, power=5):
        dist2goal = st_path[curr_vid][goal_vid]
        full_len = st_path[gt_path[0]][goal_vid]
        if full_len == 0:
            if goal_vid == curr_vid:
                return 1
            return 0
        y = max(np.power((full_len-dist2goal)/full_len, power), 0)
        return y

    def _get_gt_trajs(self, data):
        gt_trajs = {
            x['instr_id'] : (x['scan'], x['path'], x['objId']) \
                for x in data if 'objId' in x and x['objId'] is not None
        }
        return gt_trajs 
    
    ############## Nav Evaluation ################
    def _eval_item(self, scan, pred_path, pred_objid, gt_path, gt_objid,instr_id=None):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        path = sum(pred_path, [])

        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])
        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        
        # navigation: success is to arrive to a viewpoint where the object is visible
        goal_viewpoints = set(self.obj2vps['%s_%s'%(scan, str(gt_objid))])
        assert len(goal_viewpoints) > 0, '%s_%s'%(scan, str(gt_objid))
        scores['success'] = float(path[-1] in goal_viewpoints)
        scores['oracle_success'] = float(any(x in goal_viewpoints for x in path))
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)

        scores['rgs'] = str(pred_objid) == str(gt_objid)
        scores['rgspl'] = scores['rgs'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        return scores
    
    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]
        '''

        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = item['trajectory']
            pred_objid = item.get('predObjId', None)
            scan, gt_traj, gt_objid = self.gt_trajs[instr_id]
            traj_scores = self._eval_item(scan, traj, pred_objid, gt_traj, gt_objid,instr_id)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'action_steps': np.mean(metrics['action_steps']),
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'rgs': np.mean(metrics['rgs']) * 100,
            'rgspl': np.mean(metrics['rgspl']) * 100,
        }
        return avg_metrics, metrics

    
