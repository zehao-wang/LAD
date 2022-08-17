from PIL import Image
from IPython.display import display
from torchvision import transforms 
import torch as th
import torch.nn as nn 
import numpy as np 
import os 
import h5py 
import json 
import torchvision.models as models 
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from ipdb import set_trace
import clip 
import torch 
import numpy as np 
import random 
import jsonlines 

REVERIE_DATA_ROOT="../datasets/REVERIE/annotations/REVERIE_{split}_enc.json"
SOON_DATA_ROOT="../datasets/SOON/annotations/bert_enc/val_unseen_house_enc.jsonl"
SPLITS = ["val_unseen_test"]

def load_instr_datasets(anno_dir,split):
    file_path = anno_dir.format(split=split)
    with open(file_path) as f:
        data = json.load(f)
    return data 


def load_soon_datasets(soon_data_path):
    soon_data = []
    with open(soon_data_path,"r",encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            path_id = item["path_id"]
            instructions = item["instructions"]

            for i in range(len(instructions)):
                soon_data.append({
                    "instr_id"  : path_id + "-"+str(i),
                    "instruction" : instructions[i]["full"]
                })
    print(f"Load {len(soon_data)} soon episodes")

    return soon_data


def load_clip_model(model_name):
    """
    mode name list:  RN50, RN101, RN50x4, 
                     RN50x16, RN50x64, ViT-B/32, 
                     ViT-B/16, ViT-L/14,ViT-L/14@336px
    """
    model,preprocess = clip.load(model_name)
    model.cuda().eval()
    
    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Input resolution: ",input_resolution)
    print("Context length: ",context_length)
    print("Vocab size: ", vocab_size)

    return model, preprocess


# load reverie data
def construct_instrs(anno_dir,split, max_instr_len=512):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, split)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            if 'objId' in item:
                new_item['instr_id'] = '%s_%s_%d' % (str(item['path_id']), str(item['objId']), j)
            else:
                new_item['path_id'] = item['id']
                new_item['instr_id'] = '%s_%d' % (item['id'], j)
                new_item['objId'] = None
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

def build_base_model(options, has_cude, device):
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cude:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total_base parameters', sum(x.numel() for x in model.parameters()))
    return model, diffusion

def build_upsampler_model(options_up, has_cude, device):
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cude:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
    return model_up, diffusion_up

def text2image(prompt, model, diffusion, options, batch_size, device, guidance_scale=3.0):

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1 )

    guidance_scale = guidance_scale
    
    tokens = model.tokenizer.encode(prompt)
    tokens, mask = model.tokenizer.padded_tokens_and_mask(
        tokens,options['text_ctx']
    )
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
        [], options['text_ctx']
    )

    model_kwargs = dict(
        tokens=th.tensor(
            [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
        ),
        mask = th.tensor(
            [mask] * batch_size + [uncond_mask] * batch_size,
            dtype=th.bool, 
            device=device 
        ),
    )

    model.del_cache()
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device = device, 
            clip_denoised=True,
            progress=True,
            model_kwargs = model_kwargs,
            cond_fn=None,
        )[:batch_size]
    model.del_cache()
    
    num_of_samples = samples.shape[0]
    imgs_array= np.empty(samples.permute(0,2,3,1).shape, dtype=np.uint8)
    for i in range(num_of_samples):
        sample = samples[i].unsqueeze(0)
        scaled = ((sample + 1) * 127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped_img_array = scaled.permute(2,0,3,1).reshape([sample.shape[2], -1, 3]).numpy()
        imgs_array[i] = reshaped_img_array
    return imgs_array, samples 


def upsample_img(prompt, model_up, difficusion_up, options_up ,samples, batch_size,upsample_temp, device):
    tokens = model_up.tokenizer.encode(prompt)
    tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
        tokens, options_up["text_ctx"]
    )
    model_kwargs = dict(
        low_res = ((samples + 1)*127.5).round()/127.5 -1,
        tokens = th.tensor(
            [tokens] * batch_size, device=device
            ),
        mask = th.tensor(
            [mask] * batch_size,
            dtype=th.bool,
            device=device,
        ),
    )
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    with torch.no_grad():
        up_samples = difficusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs = model_kwargs,
            cond_fn=None,
        )[:batch_size]
    model_up.del_cache()
    
    num_of_samples = samples.shape[0]
    imgs_array= np.empty(up_samples.permute(0,2,3,1).shape, dtype=np.uint8)
    for i in range(num_of_samples):
        up_sample = up_samples[i].unsqueeze(0)
        scaled = ((up_sample + 1) * 127.5).round().clamp(0,255).to(th.uint8).cpu()
        reshaped_img_array = scaled.permute(2,0,3,1).reshape([up_sample.shape[2], -1, 3]).numpy()
        imgs_array[i] = reshaped_img_array
    return imgs_array, up_samples


def get_vision_model(model_name="imgnet_resnet152"):
    if model_name == "imgnet_resnet152":
        resnet152 = models.resnet152(pretrained=True)
        modules = list(resnet152.children())[:-1]
        resnet152 = nn.Sequential(*modules)
        for p in resnet152.parameters():
            p.requires_grad = False 
        return resnet152

if __name__ == "__main__":
    collect_clip_feats = True 
    collect_imgnet_feats = True 

    has_cude = th.cuda.is_available()
    device = th.device('cpu' if not has_cude else 'cuda')
    to_tensor = transforms.Compose(
        [transforms.Resize((256,256),2),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
        ]
    )

    # set parameters
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cude
    options['timestep_respacing'] = '50'

    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cude
    options_up['timestep_respacing'] = 'fast27'
    batch_size = 5
    #img_batch_size = 5
    guidance_scale=3.0
    upsample_temp=0.997
    
    model, diffusion = build_base_model(options,has_cude, device)
    model_up, difficusion_up = build_upsampler_model(options_up, has_cude, device)
    
    if collect_imgnet_feats:
        imgnet_resnet = get_vision_model(model_name="imgnet_resnet152")
        imgnet_resnet = imgnet_resnet.to(device)
    if collect_clip_feats:
        clip_model, clip_preprocess = load_clip_model(model_name="ViT-L/14@336px")

    for split in SPLITS:

        #img_file = h5py.File("/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_reverie_ins_to_imgs.h5".format(split=split), "w")
        img_file = h5py.File("/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_soon_ins_to_imgs.h5".format(split=split), "w")
    
        if collect_imgnet_feats:
            #save_dir = "/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_reverie_ins2img_imgnet_feats.h5".format(split=split)
            save_dir = "/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_soon_ins2img_imgnet_feats.h5".format(split=split)
            imgnet_feats_file = h5py.File(save_dir,"w")
        if collect_clip_feats:
            #save_dir = "/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_reverie_ins2img_clip_feats.h5".format(split=split)
            save_dir = "/staging/leuven/stg_00095/mingxiao/reverie/feats/{split}_soon_ins2img_clip_feats.h5".format(split=split)
            clip_visual_feats_file = h5py.File(save_dir,"w")
        
        #instrs = construct_instrs(REVERIE_DATA_ROOT, split)
        instrs = load_soon_datasets(SOON_DATA_ROOT)
        total_ins = len(instrs)
        for n , ins in enumerate(instrs):

            instr_id = ins['instr_id']
            instruction = ins['instruction']
      
            input_images = []
            if collect_imgnet_feats:
                imgnet_input_images = []
            if collect_clip_feats:
                clip_input_images = []
            #for _ in range(img_batch_size):
            image_array, samples = text2image(instruction, model, diffusion, options, batch_size, device, guidance_scale)
            image_up_array, sample_up = upsample_img(instruction, model_up, difficusion_up, options_up, samples, batch_size, upsample_temp, device)
        
            # for i in range(image_array.shape[0]):
            #     img = Image.fromarray(image_array[i])
            #     img.save("./tmp/test_img_"+str(i)+".jpeg")
            
            for k in range(image_up_array.shape[0]):
                img_up = Image.fromarray(image_up_array[k])
                # img_up.save("./tmp/test_img_up_"+str(k)+".jpeg")
                if collect_imgnet_feats:
                    image_input = to_tensor(img_up)
                    image_input = image_input.unsqueeze(0)
                    imgnet_input_images.append(image_input)
        
                if collect_clip_feats:
                    clip_img_input = clip_preprocess(img_up)
                    clip_img_input = clip_img_input.unsqueeze(0)
                    clip_input_images.append(clip_img_input)

                image_input = to_tensor(img_up).cpu().detach().numpy()
                input_images.append(image_input)
            
            if random.random() <= 0.01:
                r_index = random.randint(0,4)
                img_file[instr_id] = input_images[r_index]

            if collect_imgnet_feats:
                with torch.no_grad():
                    input_images = th.cat(imgnet_input_images,dim=0)
                    imgnet_features = imgnet_resnet(input_images.to(device))
                    imgnet_feats_file[instr_id] = imgnet_features.cpu().detach().numpy()
            if collect_clip_feats:
                with torch.no_grad():
                    clip_input_images = th.cat(clip_input_images,dim=0)
                    clip_features = clip_model.encode_image(clip_input_images.to(device)).float()
                    clip_visual_feats_file[instr_id] = clip_features.cpu().detach().numpy()
                
            print(f"Finish {n/total_ins}% of {split} dataset !!")
    
        img_file.close()
        imgnet_feats_file.close()
        clip_visual_feats_file.close()

    print("Finish !!")
    

    
