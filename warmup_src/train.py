import os
import torch
from transformers import AutoTokenizer, PretrainedConfig, AutoModel
from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from utils.misc import NoOp, set_dropout, set_random_seed, set_cuda, wrap_model
from model.pretrain_cmt import GlocalTextPathCMTPreTraining
from utils.save import ModelSaver, save_training_meta, get_last_ckpts
from pretrainer_base import PretrainerBase
from tqdm import tqdm
import time
import torch.nn.functional as F
import sys
from utils.distributed import all_gather
from parser import load_parser, parse_with_config
class Pretrainer(PretrainerBase):
    def __init__(self, opts):
        self.opts = opts
        self.init_configs()
        self.init_model()
    
    def init_configs(self):
        self.default_gpu, n_gpu, self.device = set_cuda(self.opts)
        
        if self.default_gpu:
            LOGGER.info(
                'device : {} n_gpu : {},  distributed training :{}, 16-bits training :{}'.format(
                    self.device, n_gpu, bool(self.opts.local_rank != -1), self.opts.fp16 
                )
            )
        seed = self.opts.seed
        if self.opts.local_rank != -1:
            seed += self.opts.rank 
        set_random_seed(seed)
        # Model config 
        model_config = PretrainedConfig.from_json_file(self.opts.model_config)
        model_config.use_fix_rt_emb = self.opts.use_fix_rt_emb
        model_config.rp_embed_dir = self.opts.rt_embed_dir
        model_config.use_clip_feat = self.opts.use_clip_feat
        model_config.use_clip_feat_txt = self.opts.use_clip_feat_txt
        
        model_config.update_rp_embed = self.opts.update_rp_embed
        model_config.use_rt_task = self.opts.use_rt_task
        model_config.fuse_dist_score_to_global = self.opts.fuse_dist_score_to_global
        model_config.switch_first_gd = self.opts.switch_first_gd
        model_config.avg_local_emb = self.opts.avg_local_emb
        model_config.const_fuse_gl = self.opts.const_fuse_gl
        model_config.const_fuse_gl_weight = self.opts.const_fuse_gl_weight
        if self.opts.start_from == 1:
            model_config.pretrain_tasks = set(self.opts.train_datasets['REVERIE']['tasks_phase1'])
        elif self.opts.start_from == 2:
            model_config.pretrain_tasks = set(self.opts.train_datasets['REVERIE']['tasks_phase2'])
        else:
            raise ValueError(f"Error start phase {self.opts.start_from}")
        self.model_config = model_config

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.lang_bert_name)
        model_class = GlocalTextPathCMTPreTraining
        self.model_class = model_class
        # Prepare model
        if self.opts.checkpoint:
            checkpoint = torch.load(self.opts.checkpoint, map_location=lambda storage, loc: storage)
        else:
            checkpoint = {}
            if self.opts.init_pretrained == 'bert':
                tmp = AutoModel.from_pretrained(self.model_config.lang_bert_name)
                for param_name, param in tmp.named_parameters():
                    checkpoint[param_name] = param
                if self.model_config.lang_bert_name == 'xlm-roberta-base':
                    # embeddings.token_type_embeddings.weight (1 -> 2, the second is for image embedding)
                    checkpoint['embeddings.token_type_embeddings.weight'] = torch.cat(
                        [checkpoint['embeddings.token_type_embeddings.weight']] * 2, 0
                    )
                del tmp
            elif self.opts.init_pretrained == 'lxmert':
                tmp = torch.load(
                    '../datasets/pretrained/model_LXRT.pth', 
                    map_location=lambda storage, loc: storage
                )
                print(f"Loading weights from {'../datasets/pretrained/model_LXRT.pth'}")
                for param_name, param in tmp.items():
                    param_name = param_name.replace('module.', '')
                    if 'bert.encoder.layer' in param_name:
                        param_name = param_name.replace('bert.encoder.layer', 'bert.lang_encoder.layer')
                        checkpoint[param_name] = param
                    elif 'bert.encoder.x_layers' in param_name:
                        param_name1 = param_name.replace('bert.encoder.x_layers', 'bert.local_encoder.encoder.x_layers')
                        param_name2 = param_name.replace('bert.encoder.x_layers', 'bert.global_encoder.encoder.x_layers')
                        checkpoint[param_name1] = checkpoint[param_name2] = param
                    elif 'cls.predictions' in param_name:
                        param_name = param_name.replace('cls.predictions', 'mlm_head.predictions')
                        checkpoint[param_name] = param
                    else:
                        checkpoint[param_name] = param
                
                del tmp
        
        # update some training configs
        model = model_class.from_pretrained(
            pretrained_model_name_or_path=None, config=self.model_config, state_dict=checkpoint
        )
        model.train()
        mode_k = [k[0] for k in model.named_parameters()]
        ck_k = list(checkpoint.keys())
        print("Load weights :" , sum([1 for k in mode_k if k in ck_k]))
        
        set_dropout(model, self.opts.dropout)
        model = wrap_model(model, self.device, self.opts.local_rank)
        self.model = model
        
        if self.opts.use_clip_feat:
            print("Load clip feats ")
            self.opts.train_datasets["REVERIE"]["clip_ft_file"].append("../datasets/R2R/features/pth_clip_vit_l_14_336px.hdf5")
            self.ins2img_file = self.opts.train_datasets['REVERIE']["ins2img_file_clip"]
        else:
            self.ins2img_file = self.opts.train_datasets['REVERIE']["ins2img_file_imgnet"]
        
        del checkpoint

    def pretrain_phase1(self):
        """ Phase 1 pretraining with augmentated data""" 
        self.model_config.pretrain_tasks = set(self.opts.train_datasets['REVERIE']['tasks_phase1'])
        if self.default_gpu:
            save_training_meta(self.opts,'vit_aug_mlm.mrc.sap.og.rt/logs','vit_aug_mlm.mrc.sap.og.rt/ckpts')
            TB_LOGGER.create(os.path.join(self.opts.output_dir, 'vit_aug_mlm.mrc.sap.og.rt', 'logs'))
            self.pbar = tqdm(total=self.opts.num_train_steps)
            self.model_saver = ModelSaver(os.path.join(self.opts.output_dir, 'vit_aug_mlm.mrc.sap.og.rt', 'ckpts'))
            add_log_to_file(os.path.join(self.opts.output_dir, 'vit_aug_mlm.mrc.sap.og.rt','logs', 'log.txt'))
        else:
            LOGGER.disabled = True 
            self.pbar = NoOp()
            self.model_saver = NoOp()
        assert self.opts.use_ins2img is False 
        print("Start training with augment data!")
        print("Training tasks :",self.opts.train_datasets["REVERIE"]["tasks_phase1"])
        self.train(task_ext = 'phase1')
        print("FINISH TRAINING WITH AUGMENTATED DATASET !!!")
    
    def pretrain_phase2(self, ckpt_path=None):
        """ Phase 2 pretraining with training set"""
        self.model_config.pretrain_tasks = set(self.opts.train_datasets['REVERIE']['tasks_phase2'])
        if ckpt_path is not None:
            last_ckpt_of_aug = ckpt_path
        else:
            aug_model_ckpt_path = os.path.join(self.opts.output_dir,'vit_aug_mlm.mrc.sap.og.rt','ckpts')
            last_ckpt_of_aug = get_last_ckpts(aug_model_ckpt_path)
        print("Loading ckpt from ",last_ckpt_of_aug)
        print("\n\nLoading ckpt from ",last_ckpt_of_aug, file=sys.stderr)
        
        checkpoint = torch.load(last_ckpt_of_aug, map_location=lambda storage, loc: storage)
        model = self.model_class.from_pretrained(
            pretrained_model_name_or_path=None, config = self.model_config, state_dict = checkpoint
        )
        model.train()
        set_dropout(model, self.opts.dropout)
        model = wrap_model(model, self.device, self.opts.local_rank)
        del checkpoint
        self.model = model

        if self.default_gpu:
            # action prediction without dist fusiong
            save_training_meta(self.opts,'logs','ckpts')
            TB_LOGGER.create(os.path.join(self.opts.output_dir, 'logs'))
            self.pbar = tqdm(total=self.opts.num_train_steps)
            self.model_saver = ModelSaver(os.path.join(self.opts.output_dir, 'ckpts'))
            add_log_to_file(os.path.join(self.opts.output_dir,'logs','log.txt'))
        else:
            LOGGER.disabled = True 
            self.pbar = NoOp()
            self.model_saver = NoOp()
        self.opts.use_ins2img=True
        print("Start training with training data!")
        print("Training tasks :",self.opts.train_datasets["REVERIE"]["tasks_phase2"])
        self.train(task_ext = 'phase2')
        print("FINISH TRAINING WITH AUGMENTATED DATASET !!!")


    def validate(self, val_dataloaders, setname=''):
        self.model.eval()
        for task, loader in val_dataloaders.items():
            LOGGER.info(f"validate val{setname} on {task} task")
            if task.startswith('mlm'):
                val_log = self.validate_mlm(self.model, loader)
            elif task.startswith('mrc'):
                val_log = self.validate_mrc(self.model, loader)
            elif task.startswith('sap'):
                val_log = self.validate_sap(self.model, loader)
            elif task.startswith('og'):
                val_log = self.validate_og(self.model, loader)
            elif task.startswith('rt'):
                val_log = self.validate_rt(self.model, loader)
            elif task.startswith('distsap'):
                val_log = self.validate_distsap(self.model, loader)
            else:
                raise ValueError(f'Undefined task {task}')
            val_log = {f'val{setname}_{task}_{k}': v for k, v in val_log.items()}
            TB_LOGGER.log_scalar_dict(
                {f'valid{setname}_{task}/{k}': v for k, v in val_log.items()}
            )
        self.model.train()

    @torch.no_grad()
    def validate_rt(self, model, val_loader):
        LOGGER.info("start running RT validation...")
        val_loss = 0
        n_correct = 0
        n_node = 0
        st = time.time()
        for i, batch in enumerate(val_loader):
            scores = model(batch, task='rt', compute_loss=False)
            labels = batch['gmap_rt_labels']
            labels = labels[labels != -1]
            loss = F.cross_entropy(scores, labels, reduction='sum')
            val_loss += loss.item()
            n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
            n_node += labels.numel()
        val_loss = sum(all_gather(val_loss))
        n_correct = sum(all_gather(n_correct))
        n_node = sum(all_gather(n_node))
        tot_time = time.time()-st
        val_loss /= n_node
        acc = n_correct / n_node
        val_log = {'loss': val_loss,
                'acc': acc,
                'tok_per_s': n_node/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"acc: {acc*100:.2f}")
        return val_log

    @torch.no_grad()
    def validate_distsap(self, model, val_loader):
        LOGGER.info("start running DIST SAP validataion...")
        val_dloss, val_gloss, val_lloss, val_floss = 0, 0, 0, 0
        n_dcorrect, n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0, 0
        n_data = 0
        st = time.time()
        for i, batch in enumerate(val_loader):
            dist_logits, global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = \
                model(batch, task='distsap', compute_loss=False) 

            val_dloss += F.cross_entropy(dist_logits, global_act_labels, reduction='sum').data.item()
            val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
            val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
            val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
            n_dcorrect += torch.sum(torch.argmax(dist_logits, 1) == global_act_labels).item()
            n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
            n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
            n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
            n_data += len(global_act_labels)   
        
        n_data = sum(all_gather(n_data))
        val_dloss = sum(all_gather(val_dloss)) / n_data
        val_gloss = sum(all_gather(val_gloss)) / n_data
        val_lloss = sum(all_gather(val_lloss)) / n_data
        val_floss = sum(all_gather(val_floss)) / n_data
        dacc = sum(all_gather(n_dcorrect)) / n_data 
        gacc = sum(all_gather(n_gcorrect)) / n_data
        lacc = sum(all_gather(n_lcorrect)) / n_data
        facc = sum(all_gather(n_fcorrect)) / n_data

        tot_time = time.time()-st
        val_log = {'dis_loss': val_dloss, 'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
                'dacc': dacc,'gacc': gacc, 'lacc': lacc, 'facc': facc,
                'tok_per_s': n_data/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"dacc: {dacc*100:.2f}, gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
        return val_log


def build_args():
    parser = load_parser()
    opts = parse_with_config(parser)

    if os.path.exists(opts.output_dir) and os.listdir(opts.output_dir):
        LOGGER.warning(
            "Ouput directory ({}) already exists and is not empty".format(
                opts.output_dir
            )
        )
    return opts

if __name__ == "__main__":
    args = build_args()
    pretrainer = Pretrainer(args)
    print(args)
    print(args.start_from)

    if args.start_from == 1:
        pretrainer.pretrain_phase1()
        #pretrainer.pretrain_phase2()    
    else:
        if args.init_ckpt is not None:
            pretrainer.pretrain_phase2(args.init_ckpt)
        else:
            pretrainer.pretrain_phase2()
