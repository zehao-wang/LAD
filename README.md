# Layout-aware Dreamer for Embodied Referring Expression Grounding

[Mingxiao Li*](https://www.kuleuven.be/wieiswie/en/person/00113732), [Zehao Wang*](https://homes.esat.kuleuven.be/~zwang), Tinne Tuytelaars, Marie-Francine Moens

AAAI 2023 main conference

[Paper](https://arxiv.org/abs/2212.00171)&nbsp;/ [BibTeX]()

## Environment
Please setup Matterport3DSimulator docker env following [link](https://github.com/peteanderson80/Matterport3DSimulator)

For missing packages, please check the corresponding version in ```requirements.txt```

## Data preparation
The data preparation including two step, preprocessing for image generation and token id extraction

### Downloads
1. Follow the insturction in [vln-duet](https://github.com/cshizhe/VLN-DUET), or download data from [Dropbox](https://www.dropbox.com/sh/u3lhng7t2gq36td/AABAIdFnJxhhCg2ItpAhMtUBa?dl=0) including processed annotations, features. Unzip the ```REVERIE``` and ```R2R``` folder into ```datasets```

2. Since we mainly use CLIP as our visual feature encoder, please follow the instruction in [link](https://github.com/openai/CLIP) and make sure to load ```ViT-L-14-336px.pt``` during training. Recommand to put in ```ckpts/ViT-L-14-336px.pt```
3. Make sure to install [GLIDE](https://github.com/openai/glide-text2im) for generation 
4. Download Matterport3D dataset from [link](https://niessner.github.io/Matterport/)
5. Additional data from lad is released at [link](https://drive.google.com/drive/folders/10xrt1iv38UC3mS0JtmoKhEMPItJ3Atbz?usp=share_link)


### Preprocessing
1. Generate imagined image of goal position 
```
python preprocess/ge_ins2img_feats.py --encoder clip --dataset reverie \
--input_dir datasets/REVERIE/annotations/REVERIE_{split}_enc.json \
--clip_save_dir datasets/REVERIE/features/reverie_ins2img_clip.h5 \
--collect_clip
```
Put the generated data in the directory ```datasets/REVERIE/features```

2. The room type codebook ```room_type_feats.h5``` has been provided at root directory

### Generate CLIP features for Matterport3 environment
1. Setup the output path and Matterport3D connectivity path in ```preprocess/get_all_imgs_fts.py```   
   Run bellow to get tsv file.    
   ```
      python preprocess/get_all_imgs_fts.py
   ```

2. Download the vit feature following [VLN-DUET](https://github.com/cshizhe/VLN-DUET) and put it in the directore of ```datasets/REVERIE/features```      
   Setup path in preprocess/convert_tsv2h5.py   
   Run to get .h5 file and put is in the directory ```datasets/REVERIE/features```   
   ```
   python preprocess/convert_tsv2h5.py
   ```

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
NOTE: The checkpoints of LAD model after warmup stage 2 and final LAD model trained on REVERIE dataset can be found [here](https://drive.google.com/drive/folders/177YBl9eNtPjmmB6E18LMAvFM6bh1keTa?usp=sharing)

## Acknowledgement
Credits to Shizhe Chen for the great baseline work [VLN-DUET](https://github.com/cshizhe/VLN-DUET):
```bibtex
@InProceedings{Chen_2022_DUET,
    author    = {Chen, Shizhe and Guhur, Pierre-Louis and Tapaswi, Makarand and Schmid, Cordelia and Laptev, Ivan},
    title     = {Think Global, Act Local: Dual-scale Graph Transformer for Vision-and-Language Navigation},
    booktitle = {CVPR},
    year      = {2022}
}
```

## License and Citation

```bibtex
@InProceedings{VLN_LAD_2023,
    author    = {Li, Mingxiao and Wang, Zehao and Tuytelaars, Tinne and Moens, Marie-Francine},
    title     = {Layout-aware Dreamer for Embodied Referring Expression Grounding},
    booktitle = {AAAI},
    year      = {2023}
}
```
