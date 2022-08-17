from collections import defaultdict
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py 
import numpy as np 
from transformers import BertPreTrainedModel
from .vilmodel import BertXAttention
from .vilmodel import BertLayerNorm, BertOnlyMLMHead, GlocalTextPathCMT
from .ops import pad_tensors_wgrad, gen_seq_masks

ROOM2IND = {'balcony' : 23, 'bathroom': 0, 'classroom': 26, 'dining_booth': 27, 'entryway': 4, 'garage': 6, 
            'junk': 29, 'laundryroom': 9, 'living room': 11, 'meetingroom': 12, 'other_room': 24, 'porch': 15,
            'spa': 28, 'toilet': 18, 'utilityroom': 19, 'bar': 25, 'bedroom': 1, 'closet': 2,
            'dining_room': 3, 'familyroom': 5, 'hallway': 7, 'kitchen': 10, 'library': 8, 'lounge': 13,
            'office': 14, 'outdoor': 22,'rec': 16, 'stairs': 17, 'tv': 20, 'workout': 21,
}

IND2ROOM={ 23: 'balcony', 0: 'bathroom', 26: 'classroom', 27: 'dining_booth',4: 'entryway', 6: 'garage',
           29: 'junk', 9: 'laundryroom', 11: 'living room', 12: 'meetingroom',24:'other_room', 15: 'porch',
           28: 'spa', 18: 'toilet', 19 :'utilityroom', 25:'bar', 1:'bedroom', 2: 'closet',
           3: 'dining_room', 5: 'familyroom', 7 :'hallway', 10: 'kitchen', 8:'library', 13: 'lounge',
           14: 'office', 12: 'outdoor', 16: 'rec', 17: 'stairs',  20:'tv', 21: 'workout'
}

class RegionClassification(nn.Module):
    " for MRC(-kl)"
    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, label_dim))

    def forward(self, input_):
        output = self.net(input_)
        return output


class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class RoomPrediction(nn.Module):
    def __init__(self, output_size, input_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = input_size 
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, output_size))
    
    def forward(self, x):
        return self.net(x)


class RoomPredictionImg(nn.Module):
    def __init__(self, config, output_size, input_size):
        super().__init__()
        print("Use img embed rt head !")
        self.room_type_list = []
        if config.rp_embed_dir is not None:
            rp_order = sorted(ROOM2IND.items(), key=lambda x: x[1])
            rp_embed_file = h5py.File(config.rp_embed_dir,"r")
            for r in rp_order:
                if config.use_clip_feat:
                    rp_embed = rp_embed_file[r[0]+'_clip'][...][:, :config.image_feat_size]
                else:
                    rp_embed = rp_embed_file[r[0]+'_imgnet_feat'][...][:, :config.image_feat_size]
                    if len(rp_embed.shape) == 4:
                        rp_embed = np.squeeze(rp_embed)
                rp_img_tensor = torch.from_numpy(rp_embed)
                linear = nn.Linear(input_size, output_size).cuda()
                
                linear.weight.data.copy_(rp_img_tensor.cuda())
                self.room_type_list.append(linear)
           
        if not config.update_rp_embed:
            for layer in self.room_type_list:
                for para in layer.parameters():
                    para.requires_grad = False
        
    def forward(self, view_feat):
        outs = []
        for layer in self.room_type_list:
            outs.append(torch.sum(layer(view_feat),dim=-1).unsqueeze(-1))
        outs = torch.cat(outs,dim=-1)
        return outs 


class NodeDistReg(nn.Module):
    def __init__(self, input_size, config, hidden_size=None):
        super().__init__()
        self.config = config
        if hidden_size is None:
            hidden_size = input_size 

        self.cross_attn = BertXAttention(config, ctx_dim=768)
        self.score_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            BertLayerNorm(hidden_size, eps=1e-12),
            nn.Linear(hidden_size, 1)
        )
        self.dreamer_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
        self.dist_sap_head = ClsPrediction(self.config.hidden_size)
        
    def forward(self, x, ins_img_features):
        ins_img_features = ins_img_features #torch.from_numpy(ins_img_features).float().to(x.device)
        attention_output, attention_scores = self.cross_attn(x, ins_img_features)
        fuse_weight = torch.sigmoid(self.dreamer_fuse_linear(
            torch.cat([x[:, 0], attention_output[:, 0]], 1)
        ))
        return None, self.dist_sap_head(attention_output).squeeze(2), fuse_weight


class GlocalTextPathCMTPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.bert = GlocalTextPathCMT(config)

        if 'mlm' in config.pretrain_tasks:
            self.mlm_head = BertOnlyMLMHead(self.config)
        if 'mrc' in config.pretrain_tasks:
            self.image_classifier = RegionClassification(self.config.hidden_size, self.config.image_prob_size)
            if self.config.obj_prob_size > 0 and self.config.obj_prob_size != self.config.image_prob_size:
                self.obj_classifier = RegionClassification(self.config.hidden_size, self.config.obj_prob_size)
            else:
                self.obj_classifier = None

        if 'sap' in config.pretrain_tasks:
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            if config.glocal_fuse:
                self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            else:
                self.sap_fuse_linear = None
        
        if 'distsap' in config.pretrain_tasks:
            self.node_dis_reg_head = NodeDistReg(input_size=self.config.hidden_size, config=config)
            self.global_sap_head = ClsPrediction(self.config.hidden_size)
            self.global_distsap_head = ClsPrediction(self.config.hidden_size)
            self.local_sap_head = ClsPrediction(self.config.hidden_size)
            self.sap_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            self.sap_global_fuse_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)
            self.global_img_linear = ClsPrediction(self.config.hidden_size, input_size=self.config.hidden_size*2)

        if 'og' in config.pretrain_tasks:
            self.og_head = ClsPrediction(self.config.hidden_size)
        
        if 'rt' in config.pretrain_tasks:
            if not config.use_fix_rt_emb:
                self.rt_head = RoomPrediction(output_size=30, 
                                          input_size=self.config.hidden_size)
            else:
                self.rt_head = RoomPredictionImg(config,output_size=10,
                                         input_size=self.config.hidden_size)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if 'mlm' in self.config.pretrain_tasks:
            self._tie_or_clone_weights(self.mlm_head.predictions.decoder,
                self.bert.embeddings.word_embeddings)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        if task.startswith('mlm'):
            return self.forward_mlm(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['txt_labels'], compute_loss
            )
        elif task.startswith('mrc'):
            return self.forward_mrc(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['vp_view_mrc_masks'], batch['vp_view_probs'], 
                batch['vp_obj_mrc_masks'], batch['vp_obj_probs'], compute_loss
            )
        elif task.startswith('sap'):
            return self.forward_sap(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['gmap_visited_masks'],
                batch['global_act_labels'], batch['local_act_labels'], compute_loss
            )
        elif task.startswith('og'):
            return self.forward_og(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'], 
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'], 
                batch['traj_step_lens'], batch['traj_vp_view_lens'], batch['traj_vp_obj_lens'], 
                batch['traj_vpids'], batch['traj_cand_vpids'], 
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['vp_pos_fts'],
                batch['obj_labels'], compute_loss
            )
        elif task.startswith('rt'):
            return self.forward_rt(
                batch['txt_ids'], batch['txt_lens'], batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'],batch['traj_loc_fts'],batch['traj_nav_types'],
                batch['traj_step_lens'],batch['traj_vp_view_lens'],batch['traj_vp_obj_lens'],
                batch['traj_vpids'], batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'], batch['gmap_pos_fts'],
                batch['gmap_pair_dists'], batch['gmap_vpids'], batch['gmap_rt_labels'],
                batch['vp_pos_fts'], batch['gmap_vpids_mask'],compute_loss
            )
        elif task.startswith('distsap'):
            return self.forward_dsap(
                batch['txt_ids'], batch['txt_lens'],batch['traj_view_img_fts'],
                batch['traj_obj_img_fts'], batch['traj_loc_fts'], batch['traj_nav_types'],
                batch['traj_step_lens'], batch['traj_vp_view_lens'],batch['traj_vp_obj_lens'],
                batch['traj_vpids'],batch['traj_cand_vpids'],
                batch['gmap_lens'], batch['gmap_step_ids'],batch['gmap_pos_fts'],
                batch['gmap_pair_dists'],batch['gmap_vpids'],batch['vp_pos_fts'],
                batch['gmap_visited_masks'],batch['global_act_labels'], batch['local_act_labels'],
                batch['ins2img_feat'], batch['current_vpid_index'],compute_loss,
            )
        else:
            raise ValueError('invalid task')

    def forward_mlm(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        txt_labels, compute_loss
    ):
        txt_embeds = self.bert.forward_mlm(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(txt_embeds, txt_labels != -1)
        prediction_scores = self.mlm_head(masked_output)

        if compute_loss:
            mask_loss = F.cross_entropy(
                prediction_scores, txt_labels[txt_labels != -1], reduction='none'
            )
            return mask_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        '''get only the masked region (don't compute unnecessary hiddens)'''
    
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))

        return hidden_masked

    def forward_mrc(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        vp_view_mrc_masks, vp_view_probs, vp_obj_mrc_masks, vp_obj_probs, compute_loss=True
    ):
        _, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )
        
        # view point leng at the last position
        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens)]
        vp_view_embeds = pad_tensors_wgrad(
            [x[1:view_len+1] for x, view_len in zip(vp_embeds, vp_view_lens)]
        )   # [stop] at 0
        
    
        # only compute masked regions for better efficient=cy
        view_masked_output = self._compute_masked_hidden(vp_view_embeds, vp_view_mrc_masks)
 
        view_prediction_soft_labels = self.image_classifier(view_masked_output)
        view_mrc_targets = self._compute_masked_hidden(vp_view_probs, vp_view_mrc_masks)

        if traj_obj_img_fts is not None:
            vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens)]
            vp_obj_embeds = pad_tensors_wgrad(
                [x[view_len+1:view_len+obj_len+1] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)]
            )
            # vp_obj_mrc_masks = vp_obj_mrc_masks[:, :vp_obj_embeds.size(1)]
            obj_masked_output = self._compute_masked_hidden(vp_obj_embeds, vp_obj_mrc_masks)
            if self.obj_classifier is None:
                obj_prediction_soft_labels = self.image_classifier(obj_masked_output)
            else:
                obj_prediction_soft_labels = self.obj_classifier(obj_masked_output)
            obj_mrc_targets = self._compute_masked_hidden(vp_obj_probs, vp_obj_mrc_masks)
        else:
            obj_prediction_soft_labels, obj_mrc_targets = None, None

        if compute_loss:
            view_prediction_soft_labels = F.log_softmax(view_prediction_soft_labels, dim=-1)
            view_mrc_loss = F.kl_div(view_prediction_soft_labels, view_mrc_targets, reduction='none').sum(dim=1)
            if obj_prediction_soft_labels is None:
                mrc_loss = view_mrc_loss
            else:
                obj_prediction_soft_labels = F.log_softmax(obj_prediction_soft_labels, dim=-1)
                obj_mrc_loss = F.kl_div(obj_prediction_soft_labels, obj_mrc_targets, reduction='none').sum(dim=1)
                mrc_loss = torch.cat([view_mrc_loss, obj_mrc_loss], 0)
            return mrc_loss
        else:
            return view_prediction_soft_labels, view_mrc_targets, obj_prediction_soft_labels, obj_mrc_targets

    def forward_sap(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, compute_loss
    ):
        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
 
        if self.sap_fuse_linear is None:
            fuse_weights = 0.5
        else:
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gen_seq_masks(gmap_lens).logical_not(), -float('inf'))

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]
            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        fused_logits[i, j] += bw_logits

        if compute_loss:
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses
            return losses
        else:
            return global_logits, local_logits, fused_logits, global_act_labels, local_act_labels

    def forward_og(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        obj_labels, compute_loss
    ):
        gmap_embeds, vp_embeds = self.bert.forward(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
            return_gmap_embeds=False
        )

        vp_view_lens = [x[-1] for x in torch.split(traj_vp_view_lens, traj_step_lens, 0)]
        vp_obj_lens = [x[-1] for x in torch.split(traj_vp_obj_lens, traj_step_lens, 0)]
        obj_embeds = pad_tensors_wgrad([
            x[1+view_len: 1+view_len+obj_len] for x, view_len, obj_len in zip(vp_embeds, vp_view_lens, vp_obj_lens)
        ])
        obj_masks = gen_seq_masks(torch.stack(vp_obj_lens, 0))

        obj_logits = self.og_head(obj_embeds).squeeze(2)
        obj_logits.masked_fill_(obj_masks.logical_not(), -float('inf'))

        if compute_loss:
            losses = F.cross_entropy(obj_logits, obj_labels, reduction='none')
            return losses
        else:
            return obj_logits
    
    def forward_rt(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, gmap_rt_labels, vp_pos_fts,
        gmap_vpids_mask, compute_loss=True):
  
        gmap_embeds, _  = self.bert.forward(txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
                    traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
                    gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
                    gmap_vpids_mask, return_gmap_embeds=True, rt_task=True)
        
        mask_out = self._compute_masked_hidden(gmap_embeds[:,1:], gmap_rt_labels != -1)
        rt_logits = self.rt_head(mask_out)
        
        if compute_loss:
            rt_loss = F.cross_entropy(
                rt_logits, gmap_rt_labels[gmap_rt_labels != -1], reduction='none')
            return rt_loss
        else:
            return rt_logits
    
    def forward_dsap(
        self, txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types,
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
        gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        gmap_visited_masks, global_act_labels, local_act_labels, ins2img, current_vpid_index,
        compute_loss 
    ):  

        batch_size = txt_ids.size(0)

        gmap_embeds, vp_embeds = self.bert(
            txt_ids, txt_lens, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
            traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens, traj_vpids, traj_cand_vpids,
            gmap_lens, gmap_step_ids, gmap_pos_fts, gmap_pair_dists, gmap_vpids, vp_pos_fts,
        )
                                                                    # need to define
        _, dist_logits, fuse_weight2 = self.node_dis_reg_head(gmap_embeds, ins2img)
        
        gmap_masks = gen_seq_masks(gmap_lens)
        fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

        if self.config.const_fuse_gl:
            fuse_weights = self.config.const_fuse_gl_weight
        
        global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
        current_vpid_index = current_vpid_index.cpu().numpy()
        if self.config.switch_first_gd:
            global_logits[:,0] = global_logits[np.arange(len(current_vpid_index)), np.array(current_vpid_index)]
        global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
        global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
        

        local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
        vp_nav_masks = pad_tensors_wgrad(
            [x[-1]!=1 for x in torch.split(traj_nav_types, traj_step_lens)]
        )[:, :local_logits.size(1)-1]
        vp_nav_masks = torch.cat(
            [torch.zeros(len(vp_nav_masks), 1).bool().to(vp_nav_masks.device), vp_nav_masks], 1
        )   # add [stop]
        local_logits.masked_fill_(vp_nav_masks, -float('inf'))
        
        fused_logits2 = self.global_distsap_head(gmap_embeds).squeeze(2)
        if self.config.switch_first_gd:
            dist_logits[:,0] = dist_logits[np.arange(len(current_vpid_index)), np.array(current_vpid_index)]
            fused_logits2[:,0] = fused_logits2[np.arange(len(current_vpid_index)), np.array(current_vpid_index)]
        fused_logits2 = dist_logits * (1-fuse_weight2) + fused_logits2 * fuse_weight2
        fused_logits2.masked_fill_(gmap_visited_masks, -float('inf'))
        fused_logits2.masked_fill_(gmap_masks.logical_not(), -float('inf')) 

        dist_logits_copy = torch.clone(dist_logits)
        dist_logits_copy.masked_fill_(gmap_visited_masks, -float('inf'))
        dist_logits_copy.masked_fill_(gmap_masks.logical_not(), -float('inf')) 

        # fusion
        fused_logits = torch.clone(global_logits)
        fused_logits[:, 0] += local_logits[:, 0]   # stop
        for i in range(batch_size):
            visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
            tmp = {}
            bw_logits = 0
            n_accum = 0
            for j, cand_vpid in enumerate(traj_cand_vpids[i][-1]):
                if cand_vpid in visited_nodes:
                    bw_logits += local_logits[i, j+1]
                    n_accum += 1
                else:
                    tmp[cand_vpid] = local_logits[i, j+1]

            for j, vp in enumerate(gmap_vpids[i]):
                if j > 0 and vp not in visited_nodes:
                    if vp in tmp:
                        fused_logits[i, j] += tmp[vp]
                    else:
                        if n_accum != 0 and self.config.avg_local_emb:
                            fused_logits[i, j] += bw_logits/n_accum
                        else:
                            fused_logits[i, j] += bw_logits

        # fuse dist
        fused_logits += fused_logits2

        if compute_loss:
            dist_losses = F.cross_entropy(dist_logits_copy, global_act_labels, reduction='none')
            global_losses = F.cross_entropy(global_logits, global_act_labels, reduction='none')
            local_losses = F.cross_entropy(local_logits, local_act_labels, reduction='none')
            fused_losses = F.cross_entropy(fused_logits, global_act_labels, reduction='none')
            losses = global_losses + local_losses + fused_losses + dist_losses
            return losses
        else:
            return dist_logits_copy, global_logits, local_logits, fused_logits, global_act_labels, local_act_labels
 
    

    
