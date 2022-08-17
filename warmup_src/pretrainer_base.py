import torch
from collections import defaultdict
from easydict import EasyDict
from data.loader import MetaLoader, PrefetchLoader, build_dataloader
from data.dataset import ReverieTextPathData
from data.tasks import (
    MlmDataset, mlm_collate,
    MrcDataset, mrc_collate,
    SapDataset, sap_collate,
    OGDataset, og_collate,
    RTDataset, rt_collate, 
    NodeDistDataset, node_dist_collate,
    SapWithDist, sap_with_dist_collate
    )
from model.pretrain_cmt import GlocalTextPathCMTPreTraining
from utils.logger import LOGGER, TB_LOGGER, RunningMeter
import time
from optim.misc import build_optimizer, update_optimizer
from optim import get_lr_sched
import torch.nn.functional as F
from utils.distributed import all_gather

def create_dataloaders(
    data_cfg, nav_db, tok, is_train: bool, device: torch.device, opts
):
    dataloaders = {}
    print(f" Tasks {data_cfg.tasks}")
    for k, task_name in enumerate(data_cfg.tasks):
        if task_name == 'mlm':
            task_dataset = MlmDataset(nav_db, tok)
            task_collate_fn = mlm_collate
        elif task_name == 'mrc':
            task_dataset = MrcDataset(nav_db, tok, opts.mrc_mask_prob)
            task_collate_fn = mrc_collate
        elif task_name == 'sap':
            task_dataset = SapDataset(nav_db, tok)
            task_collate_fn = sap_collate
        elif task_name == 'og':
            task_dataset = OGDataset(nav_db, tok)
            task_collate_fn = og_collate
        elif task_name == 'rt':
            task_dataset = RTDataset(nav_db, tok) 
            task_collate_fn = rt_collate
        elif task_name == 'distsap':
            task_dataset = SapWithDist(nav_db, tok)
            task_collate_fn = sap_with_dist_collate
        else:
            raise ValueError(f'Undefined task {task_name}')

        LOGGER.info(f"{task_name}: {len(task_dataset)} samples loaded")

        task_loader, pre_epoch = build_dataloader(
            task_name, task_dataset, task_collate_fn, is_train, opts
        )

        if is_train:
            ratio = data_cfg.mix_ratio[k]
            dataloaders[task_name] = (task_loader, ratio, pre_epoch)
        else:
            dataloaders[task_name] = PrefetchLoader(task_loader, device)
    return dataloaders

class PretrainerBase(object):
    def __init__(self):
        pass

    def train(self, task_ext):
        data_cfg = EasyDict(self.opts.train_datasets['REVERIE'])
        data_cfg.tasks = data_cfg["tasks_"+task_ext]
        data_cfg.mix_ratio = data_cfg["mix_ratio_"+task_ext]
        
        if self.opts.use_ins2img:
            data_cfg.train_traj_files = [data_cfg.train_traj_files[0]]
        else:
            data_cfg.train_traj_files = [data_cfg.train_traj_files[1]]
        print("Train with ",data_cfg.train_traj_files)
        print(f"Loading data from {data_cfg.train_traj_files}")
        train_nav_db = ReverieTextPathData(
            data_cfg.train_traj_files, data_cfg.vp2room_label_file, data_cfg.img_ft_file, data_cfg.obj_ft_file,
            data_cfg.scanvp_cands_file, data_cfg.connectivity_dir, data_cfg.clip_ft_file, self.ins2img_file,
            image_prob_size=self.model_config.image_prob_size,
            image_feat_size=self.model_config.image_feat_size,
            angle_feat_size=self.model_config.angle_feat_size,
            obj_feat_size=self.model_config.obj_feat_size,
            obj_prob_size=self.model_config.obj_prob_size,
            max_txt_len=self.opts.max_txt_len, max_objects=self.opts.max_objects,in_memory=True,
            use_real_dist = self.opts.use_real_dist
        )

        val_nav_db = ReverieTextPathData(
            data_cfg.val_seen_traj_files, data_cfg.vp2room_label_file, data_cfg.img_ft_file, data_cfg.obj_ft_file,
            data_cfg.scanvp_cands_file, data_cfg.connectivity_dir, data_cfg.clip_ft_file, self.ins2img_file,
            image_prob_size=self.model_config.image_prob_size,
            image_feat_size=self.model_config.image_feat_size, 
            angle_feat_size=self.model_config.angle_feat_size,
            obj_feat_size=self.model_config.obj_feat_size,
            obj_prob_size=self.model_config.obj_prob_size,
            max_txt_len=self.opts.max_txt_len, max_objects=self.opts.max_objects, in_memory=True,
            use_real_dist = self.opts.use_real_dist
        )
        val2_nav_db = ReverieTextPathData(
            data_cfg.val_unseen_traj_files, data_cfg.vp2room_label_file, data_cfg.img_ft_file, data_cfg.obj_ft_file,
            data_cfg.scanvp_cands_file, data_cfg.connectivity_dir,data_cfg.clip_ft_file, self.ins2img_file,
            image_prob_size=self.model_config.image_prob_size,
            image_feat_size=self.model_config.image_feat_size, 
            angle_feat_size=self.model_config.angle_feat_size,
            obj_feat_size=self.model_config.obj_feat_size,
            obj_prob_size=self.model_config.obj_prob_size,
            max_txt_len=self.opts.max_txt_len, max_objects=self.opts.max_objects, in_memory=True,
            use_real_dist = self.opts.use_real_dist,
        )

        # Build data loaders
        train_dataloaders = create_dataloaders(
            data_cfg, train_nav_db, self.tokenizer, True, self.device, self.opts 
        )
        val_dataloaders = create_dataloaders(
            data_cfg, val_nav_db, self.tokenizer, False, self.device, self.opts 
        )
        val2_dataloaders = create_dataloaders(
            data_cfg, val2_nav_db, self.tokenizer, False, self.device, self.opts 
        )

        meta_loader = MetaLoader(
            train_dataloaders,
            accum_steps=self.opts.gradient_accumulation_steps,
            distributed=self.opts.local_rank != -1,
            device=self.device 
        )
        meta_loader = PrefetchLoader(meta_loader, self.device)

        optimizer = build_optimizer(self.model, self.opts)
        task2scaler = {t: i for i,t in enumerate(train_dataloaders.keys())}
        
        global_step = 0

        LOGGER.info(f"***** Running training with {self.opts.world_size} GPUs *****")
        LOGGER.info("  Batch size = %d", self.opts.train_batch_size if self.opts.local_rank == -1 else self.opts.train_batch_size * self.opts.world_size)
        LOGGER.info("  Accumulate steps = %d", self.opts.gradient_accumulation_steps)
        LOGGER.info("  Num steps = %d", self.opts.num_train_steps)
        
        # to compute training statistics
        task2loss = {task: RunningMeter(f'loss/{task}')
                    for task in train_dataloaders.keys()}
        
        n_examples = defaultdict(int)
        n_in_units = defaultdict(int)
        n_loss_units = defaultdict(int)
        grad_norm = 0

        start_time = time.time()
        # quick hack for amp delay_unscale bug
        name_set = None
        warmup_optim_set = False
        normal_training_optim = True
        for step, (name, batch) in enumerate(meta_loader):
            if 'distsap' in self.model_config.pretrain_tasks:
                warmup_modules = ['global_distsap_head', 'node_dis_reg_head']
                if step < self.opts.gd_warmup_steps and not warmup_optim_set:
                    optimizer.param_groups.clear()
                    optimizer.state.clear()
                    name_set = update_optimizer(self.model, self.opts, optimizer, warmup_modules)
                    warmup_optim_set = True
                    normal_training_optim = False
                elif step >= self.opts.gd_warmup_steps and not normal_training_optim: # where warmup has been set before
                    update_optimizer(self.model, self.opts, optimizer, name_set=name_set)
                    normal_training_optim = True
                    
            # forward pass
            n_examples[name] += batch['txt_ids'].size(0)
            n_in_units[name] += batch['txt_lens'].sum().item()
            task = name.split('_')[0]
            # print(f"*********{task}**********")
            loss = self.model(batch, task=task, compute_loss=True)
                
            n_loss_units[name] += loss.size(0)
            loss = loss.mean()  # loss is not normalized in model
            
            # backward pass
            if self.opts.gradient_accumulation_steps > 1: # average loss 
                loss = loss / self.opts.gradient_accumulation_steps
            
            delay_unscale = (step+1) % self.opts.gradient_accumulation_steps != 0
            loss.backward()

            task2loss[name](loss.item())

            # optimizer update and logging
            if (step + 1) % self.opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, self.opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                TB_LOGGER.log_scalar_dict({ll.name: ll.val
                                        for ll in task2loss.values()
                                        if ll.val is not None})
                TB_LOGGER.step()

                # update model params
                if self.opts.grad_norm != -1:

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.opts.grad_norm
                    )
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                self.pbar.update(1)

                if global_step % self.opts.log_steps == 0:
                    # monitor training throughput
                    LOGGER.info(f'==============Step {global_step}===============')
                    for t in train_dataloaders.keys():
                        tot_ex = n_examples[t]
                        ex_per_sec = int(tot_ex / (time.time() - start_time))
                        tot_in = n_in_units[t]
                        in_per_sec = int(tot_in / (time.time() - start_time))
                        tot_l = n_loss_units[t]
                        l_per_sec = int(tot_l / (time.time() - start_time))
                        LOGGER.info(f'{t}: {tot_ex} examples trained at '
                                    f'{ex_per_sec} ex/s')
                        TB_LOGGER.add_scalar(f'perf/{t}_ex_per_s', ex_per_sec,
                                            global_step)
                        TB_LOGGER.add_scalar(f'perf/{t}_in_per_s', in_per_sec,
                                            global_step)
                        TB_LOGGER.add_scalar(f'perf/{t}_loss_per_s', l_per_sec,
                                            global_step)
                    LOGGER.info('===============================================')

                if global_step % self.opts.valid_steps == 0:
                    LOGGER.info(f'------Step {global_step}: start validation seen------')
                    self.validate(val_dataloaders, setname='_seen')
                    LOGGER.info(f'------Step {global_step}: start validation unseen------')
                    self.validate(val2_dataloaders, setname='_unseen')
                    self.model_saver.save(self.model, global_step)
            if global_step >= self.opts.num_train_steps:
                break

        if global_step % self.opts.valid_steps != 0:
            LOGGER.info(f'------Step {global_step}: start validation seen------')
            self.validate(val_dataloaders, setname='_seen')
            LOGGER.info(f'------Step {global_step}: start validation unseen------')
            self.validate(val2_dataloaders, setname='_unseen')
            self.model_saver.save(self.model, global_step)   

    @torch.no_grad()
    def validate_mlm(self, model, val_loader):
        LOGGER.info("start running MLM validation...")
        val_loss = 0
        n_correct = 0
        n_word = 0
        st = time.time()
        for i, batch in enumerate(val_loader):
            scores = model(batch, task='mlm', compute_loss=False)
            labels = batch['txt_labels']
            labels = labels[labels != -1]
            loss = F.cross_entropy(scores, labels, reduction='sum')
            val_loss += loss.item()
            n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
            n_word += labels.numel()
        val_loss = sum(all_gather(val_loss))
        n_correct = sum(all_gather(n_correct))
        n_word = sum(all_gather(n_word))
        tot_time = time.time()-st
        val_loss /= n_word
        acc = n_correct / n_word
        val_log = {'loss': val_loss,
                'acc': acc,
                'tok_per_s': n_word/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"acc: {acc*100:.2f}")
        return val_log

    def compute_accuracy_for_soft_targets(self, out, labels):
        outputs = out.max(dim=-1)[1]
        labels = labels.max(dim=-1)[1]  # argmax
        n_correct = (outputs == labels).sum().item()
        return n_correct

    @torch.no_grad()
    def validate_mrc(self, model, val_loader):
        LOGGER.info("start running MRC validation...")
        val_loss = 0
        n_feat = 0
        st = time.time()
        tot_score = 0
        for i, batch in enumerate(val_loader):
            view_logits, view_targets, obj_logits, obj_targets = model(batch, task='mrc', compute_loss=False)
            view_logprobs = F.log_softmax(view_logits, dim=-1)
            obj_logprobs = F.log_softmax(obj_logits, dim=-1)
            loss = F.kl_div(view_logprobs, view_targets, reduction='sum') + \
                F.kl_div(obj_logprobs, obj_targets, reduction='sum')
            tot_score += self.compute_accuracy_for_soft_targets(view_logits, view_targets) + \
                        self.compute_accuracy_for_soft_targets(obj_logits, obj_targets)
            val_loss += loss.item()
            n_feat += batch['vp_view_mrc_masks'].sum().item() + batch['vp_obj_mrc_masks'].sum().item()
        val_loss = sum(all_gather(val_loss))
        tot_score = sum(all_gather(tot_score))
        n_feat = sum(all_gather(n_feat))
        tot_time = time.time()-st
        val_loss /= n_feat
        val_acc = tot_score / n_feat
        val_log = {'loss': val_loss,
                'acc': val_acc,
                'feat_per_s': n_feat/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"score: {val_acc*100:.2f}")
        return val_log
        
    @torch.no_grad()
    def validate_sap(self, model, val_loader):
        LOGGER.info("start running SAP validation...")
        val_gloss, val_lloss, val_floss = 0, 0, 0
        n_gcorrect, n_lcorrect, n_fcorrect = 0, 0, 0
        n_data = 0
        st = time.time()
        for i, batch in enumerate(val_loader):
            global_logits, local_logits, fused_logits, global_act_labels, local_act_labels = \
                model(batch, task='sap', compute_loss=False)
            val_gloss += F.cross_entropy(global_logits, global_act_labels, reduction='sum').data.item()
            val_lloss += F.cross_entropy(local_logits, local_act_labels, reduction='sum').data.item()
            val_floss += F.cross_entropy(fused_logits, global_act_labels, reduction='sum').data.item()
            n_gcorrect += torch.sum(torch.argmax(global_logits, 1) == global_act_labels).item()
            n_lcorrect += torch.sum(torch.argmax(local_logits, 1) == local_act_labels).item()
            n_fcorrect += torch.sum(torch.argmax(fused_logits, 1) == global_act_labels).item()
            n_data += len(global_act_labels)

        n_data = sum(all_gather(n_data))
        val_gloss = sum(all_gather(val_gloss)) / n_data
        val_lloss = sum(all_gather(val_lloss)) / n_data
        val_floss = sum(all_gather(val_floss)) / n_data
        gacc = sum(all_gather(n_gcorrect)) / n_data
        lacc = sum(all_gather(n_lcorrect)) / n_data
        facc = sum(all_gather(n_fcorrect)) / n_data
        
        tot_time = time.time()-st
        val_log = {'gloss': val_gloss, 'lloss': val_lloss, 'floss': val_floss,
                'gacc': gacc, 'lacc': lacc, 'facc': facc,
                'tok_per_s': n_data/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"gacc: {gacc*100:.2f}, lacc: {lacc*100:.2f}, facc: {facc*100:.2f}")
        return val_log

    @torch.no_grad()
    def validate_og(self, model, val_loader):
        LOGGER.info("start running Object Grounding validation...")
        val_loss = 0
        n_correct = 0
        n_data = 0
        st = time.time()
        for i, batch in enumerate(val_loader):
            scores = model(batch, task='og', compute_loss=False)
            labels = batch['obj_labels']
            loss = F.cross_entropy(scores, labels, reduction='sum')
            val_loss += loss.item()
            n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
            n_data += labels.numel()
        val_loss = sum(all_gather(val_loss))
        n_correct = sum(all_gather(n_correct))
        n_data = sum(all_gather(n_data))
        tot_time = time.time()-st
        val_loss /= n_data
        acc = n_correct / n_data
        val_log = {'loss': val_loss,
                'acc': acc,
                'tok_per_s': n_data/tot_time}
        LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                    f"acc: {acc*100:.2f}")
        return val_log