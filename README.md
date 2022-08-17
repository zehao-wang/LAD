# Layout-aware Dreamer for Embodied Referring Expression Grounding


## Environment
Matterport3DSimulator docker env [link](https://github.com/peteanderson80/Matterport3DSimulator)

## Data preparation
The data preparation including two step, preprocessing for image generation and token id extraction

### Downloads
1. Follow the insturction in [vln-duet](https://github.com/cshizhe/VLN-DUET), or download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) including processed annotations, features. Unzip the ```REVERIE``` and ```R2R``` folder into ```datasets```

2. Since we mainly use CLIP as our visual feature encoder, please follow the instruction in [link](https://github.com/openai/CLIP) and make sure to load ```ViT-L-14-336px.pt``` during training.
3. Make sure to install GLIDE for generation [GLIDE](https://github.com/openai/glide-text2im)


### Preprocessing
1. Generate imagined image of goal position 
Set datapath for generating REVERIE or SOON in the ge_ins2img_feats.py first 
then run : 
```
python ge_ins2img_feats.py --split {split} --encoder clip \
--input_dir datasets/REVERIE/annotations/REVERIE_{split}_enc.json \
--save_dir datasets/REVERIE/features/reverie_ins2img_clip.h5
```
Put the generated data in the directory ```datasets/REVERIE/features```

2. The room type codebook ```room_type_feats.h5``` has been provided at root directory

### Data arrangement
1. Make sure the ```datasets``` folder under root ```lad_src```
2. link matterport dataset to ```mp3d``` under ```lad_src``` folder
The structure  of these two dataset folders should be organized as
```
lad_src
├──  datasets
│    ├── REVERIE
│    │    ├── annotations
│    │    └── features
│    │        ├── obj.avg.top3.min80_vit_base_patch16_224_imagenet.hdf5 
│    │        └── full_reverie_ins2img_clip.h5
|    └── R2R
├──  mp3d
│    └── v1
          └── scans
```

## Running scripts

> Since ins2img consume too much disk space in our situition, for augmentation data in phase1, we do not include goal dreamer in the warmup training

#### Warmup stage - phase1 training with augmentation data for single-action prediction
```bash 
cd warmup_src
sh scripts/final_frt_gd_phase1.sh
```

#### Warmup stage - phase2 training with training data for single-action prediction
```bash 
cd warmup_src
sh scripts/final_frt_gd_phase2.sh # need replace phase_ckpt in this script by best phase1 results
```

#### Training stage
```bash 
cd training_src
sh scripts/final_frt_gd_finetuning_stable.sh # need replace phase_ckpt in this script by best phase1 results
```

#### Evaluation script
```bash 
cd training_src
sh scripts/eval.sh # need replace resumedir in this script to best training result obtained above
```
