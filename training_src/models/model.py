import numpy as np 
import collections
import torch 
import torch.nn as nn 
import torch.nn.functional as  F 
from transformers import BertPreTrainedModel
import sys 
from .vilmodel import GlocalTextPathNavCMT

def get_vlnbert_models(args, config=None):
    from transformers import PretrainedConfig 
    from .vilmodel import GlocalTextPathNavCMT
    
    model_name_or_path = args.bert_ckpt_file 
    new_ckpt_weights = {}

    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path)
        for k,v in ckpt_weights.items():
            # if k.startswith('module'):
            #     k = k[7 :]
            if k.startswith('bert'):
                new_ckpt_weights[k[5:]] = v
            # if '_head' in k or 'sap_fuse' in k:
            #     new_ckpt_weights['bert.' + k] = v 
            # else:
            else:
                new_ckpt_weights[k] = v 
    
    if args.tokenizer == 'xlm': 
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    
    vis_config = PretrainedConfig.from_pretrained(cfg_name)
    
    if args.tokenizer == 'xlm':
        vis_config.type_vocab_size = 2 
    
    vis_config.max_action_steps = 100
    vis_config.image_feat_size = args.image_feat_size
    vis_config.angle_feat_size = args.angle_feat_size
    vis_config.obj_feat_size = args.obj_feat_size
    vis_config.obj_loc_size = 3
    vis_config.num_l_layers = args.num_l_layers
    vis_config.num_pano_layers = args.num_pano_layers
    vis_config.num_x_layers = args.num_x_layers
    vis_config.graph_sprels = args.graph_sprels

    vis_config.glocal_fuse = args.fusion in ['dynamic', 'fuse_ins_img']
    vis_config.num_v_layers = args.num_v_layers
    vis_config.h_graph = args.h_graph
    vis_config.rp_embed_dir = args.rp_embed_dir
    vis_config.features = args.features
    vis_config.update_rp_embed = args.update_rp_embed
    vis_config.num_layout_layers = args.num_layout_layers
    vis_config.global_fuse = args.global_fuse
    vis_config.use_img_room_head = args.use_img_room_head
    vis_config.use_gd = args.use_gd
    vis_config.fuse_dist_score_to_global=args.fuse_dist_score_to_global
    vis_config.gd_dreamer_type = args.gd_dreamer_type
    vis_config.const_fuse_gl = args.const_fuse_gl
    vis_config.const_fuse_gl_weight = args.const_fuse_gl_weight
    vis_config.const_fuse_gd_weight = args.const_fuse_gd_weight
    vis_config.const_fuse_gd = args.const_fuse_gd
    vis_config.avg_local_emb = args.avg_local_emb
    vis_config.gd_feat_dropout = args.gd_feat_dropout
    vis_config.bw_weight = args.bw_weight
    vis_config.switch_first_gd = args.switch_first_gd

    vis_config.fix_lang_embedding = args.fix_lang_embedding
    vis_config.fix_pano_embedding = args.fix_pano_embedding
    vis_config.fix_local_branch = args.fix_local_branch

    vis_config.update_lang_bert = not args.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1
    vis_config.use_lang2visn_attn = False
    
    visual_model = GlocalTextPathNavCMT.from_pretrained(
        pretrained_model_name_or_path = None,
        config = vis_config,
        state_dict = new_ckpt_weights
    )
    
    model_keys = [k[0] for k in visual_model.named_parameters()]
    load_keys = list(new_ckpt_weights.keys())
    
    num_load_weights = sum([1 for k in model_keys if k in load_keys])
    success_keys = [k for k in model_keys if k in load_keys]
    print("Model total weights %d" % len(model_keys))
    import json
    with open('./model_loaded.json', 'w') as f:
        json.dump(list(sorted(success_keys)), f, indent=4)
    print(f"Load {num_load_weights} parameters !!!")
    
    return visual_model

class VLNBert(nn.Module):

    def __init__(self, args):
        super().__init__()
        print('\n Initializing the VLN-BERT model ...')
        self.args = args  
       
        self.vln_bert = get_vlnbert_models(args, config=None)
        self.drop_env = nn.Dropout(p=args.feat_dropout)
    

    def forward(self, mode, batch):
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'language':            
            txt_embeds = self.vln_bert(mode, batch)
            return txt_embeds

        elif mode == 'panorama':
            batch['view_img_fts'] = self.drop_env(batch['view_img_fts'])
            if 'obj_img_fts' in batch:
                batch['obj_img_fts'] = self.drop_env(batch['obj_img_fts'])
            pano_embeds, pano_masks = self.vln_bert(mode, batch)
            return pano_embeds, pano_masks

        elif mode == 'navigation':
            outs = self.vln_bert(mode, batch)
            return outs
        elif mode == "navigation_with_room_type":
            outs = self.vln_bert(mode, batch)
            return outs
        elif mode == "navigation_with_layout_graph":
            outs = self.vln_bert(mode, batch)
            return outs 
        elif mode == "navigation_with_room_type_node_dist": # tuning version with multiple tuning config
            outs = self.vln_bert(mode, batch)
            return outs 
        elif mode=='navigation_with_rt_gd':  # stable version
            outs = self.vln_bert(mode, batch)
            return outs
        else:
            raise NotImplementedError('wrong mode: %s'%mode)



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()