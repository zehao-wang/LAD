"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc lr helper
"""
from torch.optim import Adam, Adamax

from .adamw import AdamW
from .rangerlars import RangerLars

def build_optimizer(model, opts):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    elif opts.optim == 'rangerlars':
        OptimCls = RangerLars
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def update_optimizer(model, opts, optimizer, training_modules = None, name_set = None):
    param_optimizer = list(model.named_parameters())
    if name_set is not None:
        param_optimizer = [pair for pair in param_optimizer if pair[0] not in name_set]

    param_optimzer_tmp = []
    if training_modules is not None:
        for training_module in training_modules:
            param_optimzer_tmp.append([pair for pair in param_optimizer if training_module in pair[0]])
        param_optimizer = sum(param_optimzer_tmp, [])      
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    for group in optimizer_grouped_parameters:
        optimizer.add_param_group(group)
    out_name_set = {n for n, p in param_optimizer}
    return out_name_set
    