# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from torch import optim as optim


def build_optimizer(config, model, backbone_low_lr=True):
	"""
	Build optimizer, set weight decay of normalization to 0 by default.
	"""
	skip = {}
	skip_keywords = {}
	if hasattr(model, 'no_weight_decay'):
		skip = model.no_weight_decay()
	if hasattr(model, 'no_weight_decay_keywords'):
		skip_keywords = model.no_weight_decay_keywords()
	if backbone_low_lr:
		parameters = set_backbone_lr(model, skip, skip_keywords)
	else:
		parameters = set_weight_decay(model, skip, skip_keywords)

	opt_lower = config.train.optimizer.lower()
	optimizer = None
	if opt_lower == 'sgd':
		optimizer = optim.SGD(parameters, momentum=config.train.momentum, nesterov=True,
		                      lr=config.train.lr, weight_decay=config.train.weight_decay)

	elif opt_lower == 'adamw':
		optimizer = optim.AdamW(parameters, eps=config.train.eps, betas=config.train.betas,
		                        lr=config.train.lr, weight_decay=config.train.weight_decay)

	return optimizer


def set_backbone_lr(model, skip_list=(), skip_keywords=()):
	has_decay_add = []
	has_decay_backbone = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if 'backbone' not in name:
			has_decay_add.append(param)
		else:
			has_decay_backbone.append(param)
	return [{'params': has_decay_add},
	        {'params': has_decay_backbone, 'lr_scale': 0.1},
	        #  {'params': no_decay_add, 'weight_decay': 0.},
	        #  {'params': no_decay_backbone, 'weight_decay': 0.,'lr_scale':0.1}
	        ]


def set_weight_decay(model, skip_list=(), skip_keywords=()):
	has_decay = []
	no_decay = []

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue  # frozen weights
		if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
				check_keywords_in_name(name, skip_keywords):
			no_decay.append(param)
		# print(f"{name} has no weight decay")
		else:
			has_decay.append(param)
	return [{'params': has_decay},
	        {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
	isin = False
	for keyword in keywords:
		if keyword in name:
			isin = True
	return isin
