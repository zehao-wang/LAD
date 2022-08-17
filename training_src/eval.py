import os 
import json 
import MatterSim
from utils.distributed import is_default_gpu, merge_dist_results, all_gather, init_distributed
from utils.data import ImageFeaturesDB, Ins2ImageFeaturesDB
from env_bases.reverie.data_utils import ObjectFeatureDB, construct_instrs, load_obj2vps, load_vp2roomlabel
from env_bases.reverie.parser import parse_args
from env import ReverNavBatchEnv
from rever_agent import ReverieMapAgent
from utils.misc import set_random_seed

class Evaluater(object):
    """Class for evaluation """
    def __init__(self, args, splits,):
        r"""
        splits: should be a list eg: ['val_unseen','val_seen']
        """
        self.args = args 
        self.splits = splits 
        self.allowd_split_list = ['val_train_seen', 'val_seen', 'val_unseen', 'test']
        
        self.output_root = None 
        self._load_feat_db()
        self._build_dataset()

    def eval(self,rank=0):
        
        agent_class = ReverieMapAgent
        for env_name, env in self.eval_envs.items():
            
            agent = ReverieMapAgent(self.args, env, rank=rank)
            print(f"Loading model from {self.args.eval_ckpt_file}")
          
            agent.load(self.args.eval_ckpt_file)
            print(f"Running evaluation on {env_name}")
            agent.test(use_dropout=False, feedback='argmax', iters=None)
            preds = agent.get_results(detailed_output=False)

            for pre in preds:
                pre["trajectory"] = [[x] for x in sum(pre["trajectory"],[])]
                
            output_dir = self.args.output_dir
            # output_dir = os.path.join(sub_dir, "eval_results")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            save_dir = os.path.join(output_dir, "submit_"+env_name+".json")
            print(f"Saving results to {save_dir}")
            json.dump(
                preds, open(save_dir,"w"),
                sort_keys=True, indent=4, separators=(',',': '))
            print(f"Finish evaluation on {env_name} dataset !!!")
            if env_name != 'test':
                avg_metrics, _ = env.eval_metrics(preds) 
                print(f"Test results on {env_name}: ")
                print(avg_metrics)
        
            

    def _build_dataset(self):
        
        self.eval_envs = {}
        dataset_calss = ReverNavBatchEnv

        for split in self.splits:
            print(f"Evaluation on {split} !!!")
            assert split in self.allowd_split_list, f"Invalid split: {split}, split should be one of {self.allowd_split_list}" 
            
            instr_data = construct_instrs(
                self.args.anno_dir, self.args.dataset, [split],
                tokenizer = self.args.tokenizer, max_instr_len = self.args.max_instr_len
            )
            env = dataset_calss(
                view_db=self.feat_db, ins2img_db = self.ins2img_db, obj_db = self.obj_db,
                instr_data = instr_data, connectivity_dir = self.args.connectivity_dir,
                obj2vps = self.obj2vps, vp2room = self.vp2room_label, batch_size = self.args.batch_size,
                angle_feat_size = self.args.angle_feat_size, seed = self.args.seed + rank,
                sel_data_idxs = None if self.args.world_size < 2 else (rank, args.world_size),
                name=split, max_objects=None, multi_startpoints=False,
                multi_endpoints=False, args=self.args 
            )
            print(f"Load {split} env !!!")
            self.eval_envs[split] = env 
    
    def _load_feat_db(self):

        self.feat_db = ImageFeaturesDB(self.args.img_ft_file, self.args.image_feat_size)
        self.obj_db = ObjectFeatureDB(self.args.obj_ft_file, self.args.obj_feat_size)
        self.ins2img_db = Ins2ImageFeaturesDB(self.args.ins2img_ft_file, self.args.image_feat_size)
        self.obj2vps = load_obj2vps(os.path.join(self.args.anno_dir, "BBoxes.json"))
        self.vp2room_label = load_vp2roomlabel(os.path.join(self.args.anno_dir, "vp2room_label.json"))
        
        print("================================================")
        print(f"Load Image Feature From {self.args.img_ft_file}")
        print(f"Load Object Feature from {self.args.obj_ft_file}")
        print(f"Load Ins2img Feature from {self.args.ins2img_ft_file}")
        print("================================================")


if __name__ == "__main__":
    args = parse_args()
    rank = 0
    set_random_seed(args.seed+rank)
    evaluater = Evaluater(args, splits=["val_seen", "val_unseen", "test"])
    evaluater = Evaluater(args, splits=["val_seen", "val_unseen"])
    # evaluater = Evaluater(args, splits=["val_unseen"])
    # evaluater = Evaluater(args, splits=["test"])
    evaluater.eval(rank)
