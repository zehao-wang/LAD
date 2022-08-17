#import clip 
import torch 
import numpy as np 
import math 
import base64
import csv 
import json 
import sys 
import MatterSim
import os 
from PIL import Image
from ipdb import set_trace
from tqdm import tqdm 
from clip import clip 
import h5py 
import torchvision.models as models
from torchvision import transforms 
import torch.nn as nn

TSV_FIELDNAMES = ['scanId', 'viewpointId', 'image_w','image_h', 'vfov', 'features']
VIEWPOINT_SIZE = 36
FEATURE_SIZE = 768
BATCH_SIZE = 4 
GPU_ID = 0 
# TODO : set model name and path to save model
OUPUTFILE = "./viewid2img_clip.tsv" 
GRAPHTS = "../../datasets/REVERIE/connectivity/"

# Simulator image parameters
WIDTH = 640
HEIGHT = 480 
VFOV = 60 
csv.field_size_limit(sys.maxsize)

def load_clip_model(model_name):

    model,preprocess = clip.load(model_name)
    model.cuda().eval()

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Input resolution: ",input_resolution)
    print("Context length: ",context_length)
    print("Vocab size: ", vocab_size)

    return model, preprocess


def load_viewpointids():
    viewpointIds = []
    with open(GRAPHTS+"scans.txt") as f:
        scans = [ scan.strip() for scan in f.readlines()]
        for scan in scans:
            with open(GRAPHTS+scan+"_connectivity.json") as j:
                data = json.load(j)
                for item in data:
                    if item["included"]:
                        viewpointIds.append((scan, item["image_id"]))
    print("Loaded %d viewpoints" % len(viewpointIds))
    return viewpointIds


def transform_img(im):
    ''' Prep opencv 3 channel image for the network '''
    im = np.array(im, copy=True)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= np.array([[[103.1, 115.9, 123.2]]]) # BGR pixel mean
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, :, :, :] = im_orig
    blob = blob.transpose((0, 3, 1, 2))
    return blob


def build_tsv():
    # Set up the simulator 
    dataset_path =  "/data/leuven/333/vsc33366/projects/datasets/mp3d_preprocess_imgs/v1/scans"
    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDatasetPath(dataset_path)
    sim.setDiscretizedViewingAngles(True)
    sim.setBatchSize(1)
    sim.initialize()


    count = 0 
    # t_render = Timer()
    # t_net = Timer()
    print("Start loading model")
    clip_model, preprocess = clip.load("/user/leuven/333/vsc33366/.cache/clip/ViT-L-14-336px.pt")
    print("Finish loading model ")

    if os.path.exists(OUPUTFILE):
        print("File exist !")
        data = read_tsv(OUPUTFILE)
        existed_data = [d["scanId"]+"_"+d["viewpointId"] for d in data]
        print(f"Number of existed data : {len(existed_data)}")

    with open(OUPUTFILE, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        
        viewpointIds = load_viewpointids()

        for scanId, viewpointId in tqdm(viewpointIds):
            if scanId+"_"+viewpointId in existed_data:
                print(f"Sample {viewpointId} of {scanId} already exists !!")
                continue

            print(f"Processing scanId :{scanId}, viewpointId :{viewpointId}")
            #t_render.tic()
            # load all discretized views from this location
            input_imgs = []
            for ix in range(VIEWPOINT_SIZE):
                if ix == 0:
                    sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    sim.makeAction([0], [1.0], [1.0])
                else:
                    sim.makeAction([0], [1.0], [0])
                
                state = sim.getState()[0]
                rgb = np.array(state.rgb, copy=False)
                input_imgs.append(preprocess(Image.fromarray(rgb)))
                assert state.viewIndex == ix 
                
            with torch.no_grad():
                image_input = torch.tensor(np.stack(input_imgs)).cuda()
                img_feature = clip_model.encode_image(image_input).float().detach().cpu().numpy()

            assert VIEWPOINT_SIZE % BATCH_SIZE == 0
            ix = 0 
            print("Writting to file -->")
            writer.writerow({
                'scanId': scanId,
                'viewpointId': viewpointId,
                'image_w': WIDTH,
                'image_h': HEIGHT,
                'vfov': VFOV,
                'features': str(base64.b64encode(img_feature),"utf-8")
            })
            print("<--Finishing writting")
            count += 1
        #t_net.toc()

        # if count % 100 == 0:
        #     print('Processed %d / %d viewpoints' %\
        #       (count,len(viewpointIds)))


def read_tsv(infile):

    in_data = []
    with open(infile, "rt") as tsv_in_file:
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
    return in_data 


def get_vision_model(model_name="imgnet_resnet152"):
    if model_name == "imgnet_resnet152":
        resnet152 = models.resnet152(pretrained=True)
        modules = list(resnet152.children())[:-1]
        resnet152 = nn.Sequential(*modules)
        for p in resnet152.parameters():
            p.requires_grad = False 
        return resnet152

def get_feature_for_room_type(out_dir):
    dump_file = h5py.File(out_dir, "w")
    path = '../datasets/REVERIE/features/mp3d_room_imgs'
    all_room_types = os.listdir(path)
    print("Start loading clip and resnet models")
    clip_model, preprocess = clip.load("/user/leuven/333/vsc33366/.cache/clip/ViT-L-14-336px.pt") 
    imgnet_resnet = get_vision_model(model_name="imgnet_resnet152")
    imgnet_resnet = imgnet_resnet.cuda()
    print("Finish loading two models ! ")

    to_tensor = transforms.Compose(
        [transforms.Resize((256,256),2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
        ]
    )

    for r in tqdm(all_room_types):
        print(f"Processing room type {r}")
        clip_input_imgs = []
        resnet_input_imgs = []
        for i in range(10):
            img_path = os.path.join(path,r,'img_0_'+str(i)+'.jpeg')
            img = Image.open(img_path)

            image_resnet = to_tensor(img).unsqueeze(0)
            resnet_input_imgs.append(image_resnet)

            image_clip  = preprocess(img)
            clip_input_imgs.append(image_clip)
        
        resnet_input = torch.cat(resnet_input_imgs,dim=0)
        resetnet_feat = imgnet_resnet(resnet_input.cuda()).cpu().detach().numpy()
        dump_file[r+'_imgnet'] = resetnet_feat
        set_trace()
        clip_input = torch.tensor(np.stack(clip_input_imgs)).cuda()
        clip_img_feature = clip_model.encode_image(image_input).float().detach().cpu().numpy()
        dump_file[r+'_clip'] = clip_img_feature
    dump_file.close()
    print("done !!!")

if __name__ == '__main__':
    # either run build_tsv() to get clip feature of images of all environment or run get_feature_for_room_type to get feature for room type codebook
    build_tsv()
    # data = read_tsv(OUPUTFILE)
    # print("Completed %d viewpoints" % len(data))
    # get_feature_for_room_type("../datasets/REVERIE/features/room_type_feats_test.h5")

                


