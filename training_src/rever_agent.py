import json 
import os 
import sys
import numpy as np
import random 
import math 
import time 
import line_profiler 
import torch 
import torch.nn as nn 
import torch.nn.functional as F

from collections import defaultdict
from torch import optim 
from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
from env_bases.reverie.agent_base import Seq2SeqAgent 
from models.graph_utils import GraphMap, GraphRoomMap
from models.model import VLNBert, Critic 
from warmup_src.model.ops import pad_tensors_wgrad 
from ipdb import set_trace

class ReverieMapAgent(Seq2SeqAgent):

    def _build_model(self):
        self.vln_bert = VLNBert(self.args).cuda()
        self.critic = Critic(self.args).cuda()
        # buffer
        self.scanvp_cands = {}
    
    def _language_variable(self, obs):
        seq_lengths = [len(ob['instr_encoding']) for ob in obs]
        max_length = max(seq_lengths)
 
        if self.args.tokenizer == 'clip':
            seq_lengths = [ np.sum(np.array(ob['instr_encoding'])) !=0 for ob in obs]
            max_length = 77
        

        # if self.args.num_of_ins_img != 0:
        if False:
            assert self.args.num_of_ins_img >= 0 and self.args.num_of_ins_img <= 5
            seq_lengths = [x+self.args.num_of_ins_img for x in seq_lengths]
            max_length += (self.args.num_of_ins_img+1)
            # add 1 for the last [SEP] token
        
            seq_tensor = np.zeros((len(obs), max_length), dtype=np.int64)
            mask = np.zeros((len(obs), max_length), dtype=np.int64)
            ins2img = np.zeros((len(obs), self.args.num_of_ins_img, self.args.image_feat_size),dtype=np.float)
            for i,ob in enumerate(obs):
                seq_tensor[i, :seq_lengths[i]-self.args.num_of_ins_img] = ob['instr_encoding']
                mask[i, : seq_lengths[i]+1] = True 
                seq_tensor[i, seq_lengths[i]] = 102
                ins2img[i, :] = ob['ins2img_feat'][:self.args.num_of_ins_img, :]
        else:
            seq_tensor = np.zeros((len(obs), max_length), dtype=np.int64)
            mask = np.zeros((len(obs), max_length), dtype=np.int64)

            for i,ob in enumerate(obs):
                seq_tensor[i, :seq_lengths[i]] = ob['instr_encoding']
                mask[i, : seq_lengths[i]] = True  
            ins2img = None

        seq_tensor = torch.from_numpy(seq_tensor).long().cuda()
        mask = torch.from_numpy(mask).cuda()
        return {
            'txt_ids': seq_tensor, 'txt_masks': mask, 'ins2img': ins2img
        }


    def _panorama_feature_variable(self, obs):
        '''Extract precomputed features into variable. '''
        batch_view_img_fts, batch_obj_img_fts, batch_loc_fts, batch_nav_types = [], [], [], []
        batch_view_lens, batch_obj_lens = [], []
        batch_cand_vpids, batch_objids = [], []

        for i, ob in enumerate(obs):
            view_img_fts, view_ang_fts, nav_types, cand_vpids = [], [], [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']): # candidate -> not inclued itself
                # candidate feature -> 768 img + angle 4 = 772
                view_img_fts.append(cc['feature'][: self.args.image_feat_size])
                view_ang_fts.append(cc['feature'][self.args.image_feat_size:])
                nav_types.append(1)   # 1 for candidate
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])
            # non cand views
            # ob['feature'] -> 36*772 ->pano feature -> here only add not candidate view point
            # length of view_img_fts not sure, for candidate poitn could have same pointId
            # since we collect all posibile navigable position on all 36 angles 
           
            view_img_fts.extend([x[:self.args.image_feat_size] for k, x 
                in enumerate(ob['feature']) if k not in used_viewidxs])
            view_ang_fts.extend([x[self.args.image_feat_size:] for k, x  
                in enumerate(ob['feature']) if k not in used_viewidxs])
            nav_types.extend([0] * (36 - len(used_viewidxs)))

            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0) 
            view_ang_fts = np.stack(view_ang_fts, 0)
            view_box_fts = np.array([[1, 1, 1]] * len(view_img_fts)).astype(np.float32)
            view_loc_fts = np.concatenate([view_ang_fts, view_box_fts], 1)

            # object
            obj_loc_fts = np.concatenate([ob['obj_ang_fts'], ob['obj_box_fts']], 1)
            nav_types.extend([2] * len(obj_loc_fts))

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_obj_img_fts.append(torch.from_numpy(ob['obj_img_fts']))
            batch_loc_fts.append(torch.from_numpy(np.concatenate([view_loc_fts, obj_loc_fts], 0)))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_cand_vpids.append(cand_vpids)
            batch_objids.append(ob['obj_ids'])
            batch_view_lens.append(len(view_img_fts))
            batch_obj_lens.append(len(ob['obj_img_fts']))
          
        # pad features to max_len
      
        batch_view_img_fts = pad_tensors(batch_view_img_fts).cuda()
        batch_obj_img_fts = pad_tensors(batch_obj_img_fts).cuda()
        batch_loc_fts = pad_tensors(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True, padding_value=0).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()
        batch_obj_lens = torch.LongTensor(batch_obj_lens).cuda()
        
        # obj_ids are the objects in current position 
        return {
            'view_img_fts': batch_view_img_fts, 'obj_img_fts': batch_obj_img_fts,
            'loc_fts': batch_loc_fts, 'nav_types': batch_nav_types,
            'view_lens': batch_view_lens, 'obj_lens': batch_obj_lens,
            'cand_vpids': batch_cand_vpids, 'obj_ids': batch_objids,
        }        
    

    def _nav_gmap_variable(self, obs, gmaps):
        # [STOP] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks, batch_gmap_room_types = [], [], []
        batch_no_vp_left = []
        batch_gmap_node_score = []
        batch_ins2img_feat = []
        batch_curr_vids = []

        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []
            for k in gmap.node_positions.keys():
                if gmap.graph.visited(k):
                    visited_vpids.append(k)
                else:
                    unvisited_vpids.append(k)
            
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1]*len(visited_vpids) + [0]*len(unvisited_vpids)
            else: # only enc unvisited node
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

           
            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids] 
            # get view point steps -> the point is visited in which step 

            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            ) # cuda  [0] --> for stop 
            
            gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )
            curr_obs = obs[i]
            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])
            
            if self.args.use_room_type:
                gmap_node_room_types = [gmap.get_node_room_type(vp) for vp in gmap_vpids[1:]]
                batch_gmap_room_types.append(torch.LongTensor(gmap_node_room_types))
            
            if self.args.use_gd:
                batch_ins2img_feat.append(curr_obs['ins2img_feat'][:self.args.num_of_ins_img, :])
                gmap_node_score = [gmap.get_node_dist(vp) for vp in gmap_vpids[1:]]
                curr_node_idx = gmap_vpids.index(gmap.curr_id)
                # if os.environ.get('DEBUG', False): 
                #     import ipdb;ipdb.set_trace() # breakpoint 190
                # gmap_node_score= [gmap.get_node_dist2goal(vp, curr_obs['gt_path'][-1], curr_obs['gt_path']) for vp in gmap_vpids[1:]]
                batch_gmap_node_score.append(torch.FloatTensor(gmap_node_score))
                batch_curr_vids.append(curr_node_idx)
           
            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))
            
        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()
       
        if self.args.use_room_type:
            batch_gmap_room_types = pad_sequence(batch_gmap_room_types, batch_first=True).cuda()
            nav_gmap = {"gmap_room_types": batch_gmap_room_types}
        else:
            nav_gmap = {}

        if self.args.use_gd:
            batch_ins2img_feat = np.array(batch_ins2img_feat)
            batch_gmap_node_score = pad_sequence(batch_gmap_node_score, batch_first=True).cuda()
            nav_gmap.update({"gmap_node_dist": batch_gmap_node_score, "ins2img": batch_ins2img_feat, "curr_vid_idx": batch_curr_vids})

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()
       
        nav_gmap.update({
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds,
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, # mask unvisited
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks, # padding mask
            'no_vp_left': batch_no_vp_left,
        })
        return nav_gmap


    def _nav_vp_variable(self, obs, gmaps, pano_embeds, cand_vpids, view_lens, obj_lens, nav_types):
        batch_size = len(obs)

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        batch_vp_pos_fts = []

        for i, gmap in enumerate(gmaps):
            cur_cand_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], cand_vpids[i],
                obs[i]['heading'], obs[i]['elevation']
            )

            cur_start_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], [gmap.start_vp],
                obs[i]['heading'], obs[i]['elevation']
            )

            # add [stop] token at begining
            vp_pos_fts = np.zeros((vp_img_embeds.size(1), 14), dtype=np.float32)
            vp_pos_fts[:, :7] = cur_start_pos_fts
            vp_pos_fts[1: len(cur_cand_pos_fts)+1, 7:] = cur_cand_pos_fts
            batch_vp_pos_fts.append(torch.from_numpy(vp_pos_fts))
        

        batch_vp_pos_fts = pad_tensors(batch_vp_pos_fts).cuda()

        vp_nav_masks = torch.cat([torch.ones(batch_size, 1).bool().cuda(), nav_types==1], 1) # candidate node
        vp_obj_masks = torch.cat([torch.zeros(batch_size, 1).bool().cuda(), nav_types == 2], 1) # object
        
        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_pos_fts': batch_vp_pos_fts,
            'vp_masks': gen_seq_masks(view_lens+obj_lens+1),
            'vp_nav_masks': vp_nav_masks,
            'vp_obj_masks': vp_obj_masks,
            'vp_cand_vpids': [[None]+x for x in cand_vpids]
        }


    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']
    

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action pannoramic view action a_t to euivalen egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s' % (ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 -1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    
    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0 # stop if arrived
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')

                    for j, vpid in enumerate(vpids[i]):
                        # 0 for stop           vpoint is not masks -> unvisited 
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                   + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist 
                                min_idx = j 
                            # which is more closer to the goal point 
                    a[i] = min_idx 
                    if min_idx == self.args.ignoreid:
                        print('scan %s : all vps are searched ' % (scan))
        return torch.from_numpy(a).cuda()
    
    
    def _teacher_object(self, obs, ended, view_lens):
        # not at right position or object not observaed -> target = -100
        targets = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:
                targets[i] = self.args.ignoreid
            else:
                i_vp = ob['viewpoint']
                if i_vp not in ob['gt_end_vps']:
                    targets[i] = self.args.ignoreid
                else:
                    i_objids = ob['obj_ids']
                    targets[i] = self.args.ignoreid
                    for j, obj_id in enumerate(i_objids):
                        if str(obj_id) == str(ob['gt_obj_id']):
                            # TODO check 
                            targets[i] = j + view_lens[i] + 1
                            break 
        return torch.from_numpy(targets).cuda()
    

    def _teacher_room_type(self, nav_inputs):
        gmap_room_type = nav_inputs['gmap_room_types']
        gmap_mask = nav_inputs['gmap_masks']
        batch_size =  gmap_room_type.shape[0]
        mask = gmap_mask[:, 1:]
        gt_room_type = mask.logical_not()*(-100) + gmap_room_type
        return gt_room_type

    def _preprocess_room_loss(self, preds, target): 
        pred_shape = preds.shape 
        target_shape = target.shape 
        preds = preds.reshape(-1, pred_shape[2])
        target = target.reshape(-1)
        return preds, target
    
    def _teacher_node_dist(self, nav_inputs):
        gmap_node_dist = nav_inputs['gmap_node_dist']
        gmap_mask = nav_inputs['gmap_masks']
        mask = gmap_mask[:, 1:]
        return gmap_node_dist, mask
    
    # def _node_dist_loss(self, preds, target, mask):
    #     """ Masked mse loss"""
    #     loss = self.criterion_mse(preds, target)
    #     loss = (loss * mask.float()).sum()

    #     non_zero_elements = mask.sum()
    #     mse_loss = loss / non_zero_elements
    #     return mse_loss

    def _node_dist_loss(self, preds, gmap_node_score, mask):
        """ KL divergence (no mask)"""
        pred = preds.float().masked_fill(mask==0, -float('inf'))
        target = gmap_node_score.float().masked_fill(mask==0, -float('inf'))
        loss = self.criterion_kl(pred, target)
        batch_sum_kl_loss = loss.sum(dim=1)
        kl_loss = batch_sum_kl_loss.sum()
        return kl_loss
              
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset: # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)
        batch_size = len(obs)
        # build graph : keep the start viewpoint
        if not self.args.use_room_type:
            gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        else:
            # if self.args.use_gd:
            #     gmaps = [GraphRoomMapScore(ob['viewpoint'], ob['scan'], use_real_dist=self.args.use_real_dist_norm) for ob in obs ]
            # else:
            gmaps = [GraphRoomMap(ob['viewpoint']) for ob in obs ]


        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob) 
        
        # Record the navigation path
        traj = [{
            'instr_id' : ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'rt_records': [],
            'pred_objid': None,
            'details': {},
        } for ob in obs ]
        
        # Language input: txt_ids , txt_masks
        language_inputs = self._language_variable(obs)
        txt_embeds = self.vln_bert('language', language_inputs)
        
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        ml_loss = 0.
        og_loss = 0.
        room_type_loss = 0.
        node_dist_loss = 0.

        for t in range(self.args.max_action_len):
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
            
            # graph representation
            pano_inputs = self._panorama_feature_variable(obs)
            #  view_img_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens, cand_vpids, obj_ids


            # model get node embedding 
            pano_embeds, pano_masks = self.vln_bert('panorama', pano_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim = True)
 
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(pano_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])
                                                              # contains obj info 
            # navigation policy 
            nav_inputs = self._nav_gmap_variable(obs, gmaps) 
            # -> get embed for global branch   -> node embeding, visited->avg_pano  unvisited->view feat
        
            nav_inputs.update(
                self._nav_vp_variable(
                    obs, gmaps, pano_embeds, pano_inputs['cand_vpids'],
                    pano_inputs['view_lens'], pano_inputs['obj_lens'],
                    pano_inputs['nav_types'],
                )
            )   # -> get embeding for local branch 
    
            nav_inputs.update({
                'txt_embeds': txt_embeds,
                'txt_masks': language_inputs['txt_masks'],
                # 'ins2img': language_inputs['ins2img']
            }
            )
            if not self.args.use_room_type:
                nav_outs = self.vln_bert('navigation', nav_inputs)
            elif self.args.h_graph:
                nav_outs = self.vln_bert('navigation_with_layout_graph', nav_inputs)
            elif self.args.use_gd:
                if self.args.stable_gd:
                    nav_outs = self.vln_bert('navigation_with_rt_gd', nav_inputs)
                else: # dynamic weight
                    nav_outs = self.vln_bert('navigation_with_room_type_node_dist', nav_inputs)
            else:
                nav_outs = self.vln_bert('navigation_with_room_type', nav_inputs) 
        
            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            elif self.args.fusion == 'fuse_ins_img':
                nav_logits = nav_outs['fused_logits'] 
                dist_logits = nav_outs['dist_logits']
                logit_mask = dist_logits == -float('inf')
                nav_logits = torch.softmax(nav_logits, 1) + self.args.fuse_dist_score_to_global * torch.softmax(dist_logits, 1)
                nav_logits[logit_mask] = -float('inf')
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits'] 
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            obj_logits = nav_outs['obj_logits']
          
            # update graph 
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    # update i_vp: stop and object grounding scores
                    i_objids = obs[i]['obj_ids']
                    i_obj_logits = obj_logits[i, pano_inputs['view_lens'][i]+1: ]
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                        'og': i_objids[torch.argmax(i_obj_logits)] if len(i_objids) >0 else None,
                        'og_details': {'objids': i_objids, 'logits': i_obj_logits[:len(i_objids)]},
                    }

            # records room type
            if self.args.record_rt:
                target_room_type = self._teacher_room_type(nav_inputs)
                pred_room, target_room = self._preprocess_room_loss(nav_outs['room_type_pred'][:,1:], target_room_type)
                pred_room_label = torch.argmax(pred_room, dim=1)
                pred_room_label = pred_room_label.reshape(target_room_type.size())
                rt_records = (target_room_type.detach().cpu().numpy().astype(int).tolist(), pred_room_label.detach().cpu().numpy().astype(int).tolist(), nav_vpids, [gmap.curr_id for gmap in gmaps])


            if train_ml is not None:
                # supervised training
                nav_targets = self._teacher_action(
                    obs, nav_vpids, ended,
                    visited_masks = nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None 
                )

                ml_loss += self.criterion(nav_logits, nav_targets)
                if self.args.fusion in ['avg', 'dynamic', 'fuse_ins_img'] and self.args.loss_nav_3:
                    ml_loss += self.criterion(nav_outs['global_logits'], nav_targets)
                    local_nav_targets = self._teacher_action(
                        obs, nav_inputs['vp_cand_vpids'], ended, visited_masks=None 
                    )
                    ml_loss += self.criterion(nav_outs['local_logits'], local_nav_targets)

                obj_targets = self._teacher_object(obs, ended, pano_inputs['view_lens'])
                og_loss += self.criterion(obj_logits, obj_targets)
  
                if self.args.use_room_type or self.args.h_graph:
                    target_room_type = self._teacher_room_type(nav_inputs)
                    pred_room, target_room = self._preprocess_room_loss(nav_outs['room_type_pred'][:,1:], target_room_type)
                    room_type_loss += self.criterion(pred_room, target_room)
                
                # if self.args.use_gd:
                #     target_node_score, mask = self._teacher_node_dist(nav_inputs)
                #     node_dist_loss += self._node_dist_loss(nav_outs['node_dist_pred'][:,1:].squeeze(2), target_node_score, mask)
                if self.args.use_dist_logits_prediction and not self.args.stable_gd:
                    ml_loss += self.criterion(nav_outs['dist_logits'], nav_targets)

            # Determinate next navigation viewpoint 
            if self.feedback == 'teacher':
                a_t = nav_targets
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)
                a_t = a_t.detach()
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())
                entropys.append(c.entropy())
                a_t = c.sample().detach()
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')
            

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample':
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0 

            # Prepare environment action
            cpu_a_t = []
            for i in range(batch_size):
                if self.args.record_rt:
                    # currid, neighborvids, target_rt, pred_rt
                    traj[i]['rt_records'].append((rt_records[3][i], list(zip(rt_records[2][i][1:], rt_records[0][i][:len(nav_vpids[i][1:])], rt_records[1][i][:len(nav_vpids[i][1:])]))))

                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_len - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True 
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])
            
            # Make action and got the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i] :
                    stop_node, stop_score = None, {'stop': -float('inf'), 'og': None}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v 
                            stop_node = k 
                    
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    traj[i]['pred_objid'] = stop_score['og']

                    if self.args.detailed_output:
                        for k,v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                                'obj_ids': [str(x) for x in v['og_details']['objids']],
                                'obj_logits': v['og_details']['logits'].tolist()
                            }
            
            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)
            
            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break 
        
      
        if train_ml is not None:
            ml_loss = ml_loss * train_ml / batch_size
            og_loss = og_loss * train_ml / batch_size 
            if self.args.use_room_type:
                rt_loss = room_type_loss / batch_size 
                self.logs['RT_loss'].append(rt_loss.item())
                self.loss += self.args.rt_weights * rt_loss 
            # if self.args.use_gd:
            #     dist_loss = node_dist_loss/batch_size
            #     self.loss += self.args.node_loss_delta * dist_loss
            #     self.logs['DIST_loss'].append(dist_loss.item())

            self.loss += ml_loss
            self.loss += og_loss
            self.logs['IL_loss'].append(ml_loss.item()) 
            self.logs['OG_loss'].append(og_loss.item())
            
        return traj 

    def get_results(self, detailed_output=False):
        output = []
        for k, v in self.results.items():
            if v['pred_objid'] is None:
                if self.args.record_rt:
                    output.append({'instr_id': k, 'trajectory': v['path'], 'predObjId': v['pred_objid'], 'record_rt': v['rt_records']})
                else:
                    output.append({'instr_id': k, 'trajectory': v['path'], 'predObjId': v['pred_objid']})
            else:
                if self.args.record_rt:
                    output.append({'instr_id': k, 'trajectory': v['path'], 'predObjId': int(v['pred_objid']), 'record_rt': v['rt_records']})
                else:
                    output.append({'instr_id': k, 'trajectory': v['path'], 'predObjId': int(v['pred_objid'])})
            if detailed_output:
                output[-1]['details'] = v['details']
        return output

        

        