import torch 
import clip 

def get_tokenizer(args):
    if args.tokenizer == "clip":
        return clip.tokenize # including padding already

    from transformers import AutoTokenizer
    if args.tokenizer == "xlm":
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)

    return tokenizer 

