import csv
import h5py 
import sys
import numpy as np 
import base64 
import os 
from collections import defaultdict
from tqdm import tqdm 

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
csv.field_size_limit(sys.maxsize)
VIEWPOINT_SIZE = 36
FEATURE_SIZE = 768

if __name__ =="__main__":
    h5_path = "../datasets/R2R/features/pth_vit_base_patch16_224_imagenet.hdf5" 
    tsv_root = "/scratch/leuven/333/vsc33366/reverie_imgs/"  # path of tsv file
    dump_h5_path = "../datasets/R2R/features/pth_clip_vit_l_14_336px.hdf5" # path to save h5 file 
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        print(f[keys[0]][...].shape)
    
    h5_scan_to_view = defaultdict(list)
    for k in keys:
        scan, view = k.split("_")
        h5_scan_to_view[scan].append(view)

    in_data = []
    tsv_path = []
    for tsv_file in os.listdir(tsv_root):
        tsv_path.append(tsv_root+str(tsv_file))
    all_csv_key = []
    for k, tsv_file in enumerate(tsv_path):
        print(f"Reading {k} th file {tsv_file}")
        with open(tsv_file, "rt") as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames = TSV_FIELDNAMES)
            for item in reader:
                item['scanId'] = item['scanId']
                item['image_h'] = int(item['image_h'])
                item['image_w'] = int(item['image_w'])
                item["viewpointId"] = item["viewpointId"]
                item['vfov'] = int(item['vfov'])
                item['features'] = np.frombuffer(base64.b64decode(item['features']),
                                dtype=np.float32).reshape((VIEWPOINT_SIZE, FEATURE_SIZE))
                in_data.append(item)
                all_csv_key.append(item['scanId']+"_"+item['viewpointId'])
    
    csv_scan_to_view = defaultdict(list)
    dump_file = h5py.File(dump_h5_path,"w")
    for data in tqdm(in_data):
        key = data["scanId"]+"_"+data["viewpointId"]
        assert key in keys 
        dump_file[key] = data['features']

    dump_file.close()
    print("DONE !!")
