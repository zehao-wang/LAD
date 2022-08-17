# Layout-aware Dreamer for Embodied Referring Expression Grounding


## Environment
Matterport3DSimulator docker env [link](https://github.com/peteanderson80/Matterport3DSimulator)

## Data preparation
The data preparation including two step, preprocessing for image generation and token id extraction
### Preprocessing

### Data arrangement

```
datasets
├──  datasets
│     └── full_reverie_ins2img_clip.h5
├──  mp3d
│   └── v1/scans # where the simulator scans stored
└── 
```

link matterport dataset to ```mp3d``` under ```lad_src``` folder, the structure should be 
```
mp3d
└── v1
     └── scans
```

## Running scripts

> Since ins2img consum too much disk space in our situition, for augmentation data in phase1, we do not include goal dreamer in the warmup training

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
