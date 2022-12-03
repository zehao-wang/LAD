from multiprocessing.spawn import import_main_path
import os 
import json
import time 
import numpy as np 
import time 
import torch 
import sys 
from collections import defaultdict
from env import ReverNavBatchEnv
from tensorboardX import SummaryWriter
from rever_agent import ReverieMapAgent
from env_bases.reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps, load_vp2roomlabel
from env_bases.reverie.parser import parse_args
from utils.data import ImageFeaturesDB, Ins2ImageFeaturesDB
from utils.distributed import is_default_gpu, merge_dist_results, all_gather, init_distributed
from utils.logger import write_to_record_file
from utils.misc import set_random_seed

def build_dataset(args, rank = 0):
    
    print(f"Img Feature : {args.features}")
    print(f"Use h graph : {args.h_graph}")
    print(f"If use room type {args.use_room_type}")
    print(f"Using {args.tokenizer} tokenizer ")

    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)
    obj_db = ObjectFeatureDB(args.obj_ft_file, args.obj_feat_size)
    ins2img_db = Ins2ImageFeaturesDB(args.ins2img_ft_file, args.image_feat_size)
    obj2vps = load_obj2vps(os.path.join(args.anno_dir, "BBoxes.json"))
    vp2room_label = load_vp2roomlabel(os.path.join(args.anno_dir, "vp2room_label.json"))
    
    dataset_class = ReverNavBatchEnv

    train_instr_data = construct_instrs(
        args.anno_dir, args.dataset, ['train'],
        tokenizer = args.tokenizer, max_instr_len=args.max_instr_len
    )
    train_env = dataset_class(
        view_db = feat_db, ins2img_db = ins2img_db, obj_db = obj_db, instr_data = train_instr_data,
        connectivity_dir = args.connectivity_dir, obj2vps = obj2vps,
        vp2room = vp2room_label, batch_size = args.batch_size, max_objects = args.max_objects,
        angle_feat_size = args.angle_feat_size, seed = args.seed+rank,
        sel_data_idxs = None, name = 'train', multi_endpoints = args.multi_endpoints,
        multi_startpoints = args.multi_startpoints, args=args
    )
    
    val_env_names = ['val_train_seen','val_seen', 'val_unseen']

    if args.submit:
        val_env_names.append('test') 

    val_envs = {}

    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split],
            tokenizer = args.tokenizer, max_instr_len = args.max_instr_len
        )

        val_env = dataset_class(
            view_db = feat_db, ins2img_db = ins2img_db, obj_db = obj_db, instr_data = val_instr_data,
            connectivity_dir = args.connectivity_dir, obj2vps = obj2vps,
            vp2room = vp2room_label, batch_size = args.batch_size,
            angle_feat_size = args.angle_feat_size, seed = args.seed + rank,
            sel_data_idxs = None if args.world_size < 2 else (rank, args.world_size), 
            name = split, max_objects = None, multi_startpoints=False, 
            multi_endpoints = False, args=args
        )

        val_envs[split] = val_env 
    
    return train_env, val_envs 

def train(args, train_envs, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)
    
    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir = args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    
    agent_class = ReverieMapAgent
    agent = agent_class(args, train_envs, rank=rank)

    # resume file
    start_iter = 0 
    if args.resume_file is not None:
        start_iter = agent.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
    
    # firste evaluation
    if args.eval_first:
        loss_str = 'validation before training'
        for env_name, env in val_envs.items():
         
            agent.env = env 
            # Get validation distance from goal under test evaluation condition
            agent.test(use_dropout=False, feedback='argmax', iters=None)
            preds = agent.get_results()
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name 
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
    
    
    if default_gpu:
        write_to_record_file(
            '\nAgent training starts, start iteration: %s'%str(start_iter), record_file
        )
    best_val_spl = {'val_unseen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}, 'val_seen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}}
    best_val_sr = {'val_unseen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}, 'val_seen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}}
    best_val_rgs = {'val_unseen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}, 'val_seen': {"spl": 0., "sr": 0., "rgs":0., "state": ""}}
    
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        agent.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        agent.env = train_envs 
        start_time = time.time()
        agent.train(interval, feedback = args.feedback)
        end_time = time.time()
        running_time = end_time - start_time

        if default_gpu:
            # Log the training states to tensorboard
            total = max(sum(agent.logs['total']), 1)
            length = max(len(agent.logs['critic_loss']), 1)
            critic_loss = sum(agent.logs['critic_loss']) / total 
            policy_loss = sum(agent.logs['policy_loss']) / total 
            OG_loss = sum(agent.logs['OG_loss']) / max(len(agent.logs['OG_loss']), 1)
            IL_loss = sum(agent.logs['IL_loss']) / max(len(agent.logs['IL_loss']), 1)
            entropy = sum(agent.logs['entropy']) / total 
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/OG_loss", OG_loss, idx)
            writer.add_scalar("loss/IL_loss",IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            
            if args.use_room_type:
                RT_loss = sum(agent.logs['RT_loss']) / max(len(agent.logs['RT_loss']), 1)
                writer.add_scalar("loss/RT_loss",RT_loss, idx)
                # if args.use_gd:
                if False:
                    DIST_loss = sum(agent.logs['DIST_loss']) / max(len(agent.logs['DIST_loss']), 1)
                    writer.add_scalar("loss/DIST_loss",DIST_loss, idx)
                    write_to_record_file(
                        "\nIter %d time %.4f seconds total_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, \
                        policy_loss %.4f, critic_loss %.4f, room_type_loss %.4f, node_dist_loss %.4f" % (iter,running_time, total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss, RT_loss, DIST_loss),
                        record_file
                    )
                else:
                    write_to_record_file(
                        "\nIter %d time %.4f seconds total_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, \
                        policy_loss %.4f, critic_loss %.4f, room_type_loss %.4f " % (iter,running_time, total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss, RT_loss),
                        record_file
                    )
            else:
                write_to_record_file(
                    "\nIter %d time %.4f seconds total_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, OG_loss %.4f, \
                    policy_loss %.4f, critic_loss %.4f" % (iter, running_time, total, length, entropy, IL_loss, OG_loss, policy_loss, critic_loss),
                    record_file
                )
        
        # Run validation
        loss_str = "Current Iter {}".format(iter)
        for env_name, env in val_envs.items():
            agent.env = env 
           
            agent.test(use_dropout=False, feedback='argmax', iters=None)
            preds = agent.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)
                loss_str += " | "
                
                if env_name in best_val_spl:
                    # select model by spl based on different split's metric
                    if score_summary['spl'] >= best_val_spl[env_name]['spl']:
                        best_val_spl[env_name]['spl'] = score_summary['spl']
                        best_val_spl[env_name]['sr'] = score_summary['sr']
                        best_val_spl[env_name]['rgs'] = score_summary['rgs']
                        best_val_spl[env_name]['state'] = 'Best Iter  %d %s' % (iter, loss_str)
                        agent.save(idx, os.path.join(args.ckpt_dir, "best_spl_%s" % (env_name)))
                    # select model by sr
                    if score_summary['sr'] >= best_val_sr[env_name]['sr']:
                        best_val_sr[env_name]['spl'] = score_summary['spl']
                        best_val_sr[env_name]['sr'] = score_summary['sr']
                        best_val_sr[env_name]['rgs'] = score_summary['rgs']
                        best_val_sr[env_name]['state'] = 'Best Iter  %d %s' % (iter, loss_str)
                        agent.save(idx, os.path.join(args.ckpt_dir, "best_sr_%s" % (env_name)))
                    # select model by rgs
                    if score_summary['rgs'] >= best_val_rgs[env_name]['rgs']:
                        best_val_rgs[env_name]['spl'] = score_summary['spl']
                        best_val_rgs[env_name]['sr'] = score_summary['sr']
                        best_val_rgs[env_name]['rgs'] = score_summary['rgs']
                        best_val_rgs[env_name]['state'] = 'Best Iter  %d %s' % (iter, loss_str)
                        agent.save(idx, os.path.join(args.ckpt_dir, "best_rgs_%s" % (env_name)))

        if default_gpu:
            agent.save(idx, os.path.join(args.ckpt_dir, " latest_dict"))
            write_to_record_file("BEST RESULT TILL NOW BY SPL", record_file)
            for env_name in best_val_spl:
                write_to_record_file(env_name + ' | ' + best_val_spl[env_name]['state'], record_file)

            write_to_record_file("BEST RESULT TILL NOW BY SR", record_file)
            for env_name in best_val_sr:
                write_to_record_file(env_name + ' | ' + best_val_sr[env_name]['state'], record_file) 
            
            write_to_record_file("BEST RESULT TILL NOW BY RGS", record_file)
            for env_name in best_val_rgs:
                write_to_record_file(env_name + ' | ' + best_val_rgs[env_name]['state'], record_file) 
           
def valid(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = ReverieMapAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the agent model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file
        ))
    
    if default_gpu:
        with open(os.path.join(args.log_dir, "validation_args.json"), "w") as  outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)
    
    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False  else 'detail'
        output_file = os.path.join(args.pred_dir, "%s_%s_%s.json" % (
            prefix, env_name, args.fusion
        ))
        if os.path.exists(output_file):
            continue 
        agent.logs = defaultdict(list)
        agent.env = env 
        
        iters = None 
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters
        )
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output = args.detailed_output)
        preds = merge_dist_results(all_gather(preds))
        for pre in preds:
            pre["trajectory"] = [[x] for x in sum(pre["trajectory"],[])]

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s"% env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n',record_file)
            
            if args.submit:
                json.dump(
                    preds, open(output_file, 'w'),
                    sort_keys= True, indent=4, separators=(',',': ')
                )


def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs = build_dataset(args, rank=rank)

    if not args.test:
        train(args, train_env, val_envs, rank=rank)
    else:
        valid(args, train_env, val_envs, rank=rank)

if __name__ == "__main__":
    main()                                                                                                                              






    
