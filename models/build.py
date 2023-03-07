import os

import timm
import torch
import numpy as np
from models import vit
from models.IELT import InterEnsembleLearningTransformer

backbone = {
	'ViT-B_16': vit.get_b16_config(),
	'ViT-B_32': vit.get_b32_config(),
	'ViT-L_16': vit.get_l16_config(),
	'ViT-L_32': vit.get_l32_config(),
	'ViT-H_14': vit.get_h14_config(),
	'testing': vit.get_testing(),
}


def build_models(config, num_classes):
	if config.model.baseline_model:
		model = baseline_models(config, num_classes)
		load_pretrained(config, model)
	else:
		structure = backbone[config.model.name]
		model = InterEnsembleLearningTransformer(structure, config.data.img_size, num_classes, config.data.dataset,
		                                         config.model.label_smooth,
		                                         config.parameters.loss_alpha, config.parameters.cam,
		                                         config.parameters.dsm, config.parameters.fix,
		                                         config.parameters.update_warm, config.parameters.vote_perhead,
		                                         config.parameters.total_num,
		                                         config.parameters.assess)
	model.load_from(np.load(config.model.pretrained))
	return model


def baseline_models(config, num_classes):
	model = None
	type = config.model.type.lower()
	if type == 'resnet':
		model = timm.models.create_model('resnet50', pretrained=False, drop_path_rate=config.model.drop_path,
		                                 num_classes=num_classes)

	elif type == 'vit':
		model = timm.models.create_model('vit_base_patch16_224_in21k', pretrained=False,
		                                 num_classes=num_classes)
	elif type == 'swin':
		model = timm.models.create_model('swin_base_patch4_window12_384_in22k', pretrained=False,
		                                 num_classes=num_classes, drop_path_rate=config.model.drop_path)

	return model


def load_pretrained(config, model):
	if config.local_rank in [-1, 0]:
		print('-' * 11, 'Loading weight {:^22} for fine-tuning'.format(config.model.pretrained), '-' * 11)

	if os.path.splitext(config.model.pretrained)[-1].lower() in ('.npz', '.npy'):
		# numpy checkpoint, try to load via model specific load_pretrained fn
		if hasattr(model, 'load_pretrained'):
			model.load_pretrained(config.model.pretrained)
			if config.local_rank in [-1, 0]:
				print('-' * 20, 'Loaded successfully \'{:^22}\''.format(config.model.pretrained), '-' * 20)
			torch.cuda.empty_cache()
			return

	checkpoint = torch.load(config.model.pretrained, map_location='cpu')
	state_dict = None
	type = config.model.type.lower()

	if type == 'resnet':
		state_dict = checkpoint
		del state_dict['fc.weight']
		del state_dict['fc.bias']


	elif type == 'swin' or type == 'swinv2':
		state_dict = checkpoint['model']
		# delete relative_position_index since we always re-init it
		relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
		for k in relative_position_index_keys:
			del state_dict[k]

		# delete relative_coords_table since we always re-init it
		relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
		for k in relative_position_index_keys:
			del state_dict[k]

		# delete attn_mask since we always re-init it
		attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
		for k in attn_mask_keys:
			del state_dict[k]

		# bicubic interpolate relative_position_bias_table if not match
		relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
		# relative_position_bias_table_keys = [x for x in relative_position_bias_table_keys if 'layers.3.' not in x]
		for k in relative_position_bias_table_keys:
			relative_position_bias_table_pretrained = state_dict[k]
			relative_position_bias_table_current = model.state_dict()[k]
			L1, nH1 = relative_position_bias_table_pretrained.size()
			L2, nH2 = relative_position_bias_table_current.size()

			if nH1 != nH2:
				print(f"Error in loading {k}, passing......")
			else:
				if L1 != L2:
					# bicubic interpolate relative_position_bias_table if not match
					S1 = int(L1 ** 0.5)
					S2 = int(L2 ** 0.5)
					relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
						relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
						mode='bicubic')

					state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
		# bicubic interpolate absolute_pos_embed if not match
		absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
		for k in absolute_pos_embed_keys:
			# dpe
			absolute_pos_embed_pretrained = state_dict[k]
			absolute_pos_embed_current = model.state_dict()[k]
			_, L1, C1 = absolute_pos_embed_pretrained.size()
			_, L2, C2 = absolute_pos_embed_current.size()
			if C1 != C1:
				print(f"Error in loading {k}, passing......")
			else:
				if L1 != L2:
					S1 = int(L1 ** 0.5)
					S2 = int(L2 ** 0.5)
					absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
					absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
					absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
						absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
					absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
					absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
					state_dict[k] = absolute_pos_embed_pretrained_resized

		del state_dict['head.weight']
		del state_dict['head.bias']

	msg = model.load_state_dict(state_dict, strict=False)
	# print(msg)
	if config.local_rank in [-1, 0]:
		print('-' * 16, ' Loaded successfully \'{:^22}\' '.format(config.model.pretrained), '-' * 16)

	del checkpoint
	torch.cuda.empty_cache()


def freeze_backbone(model, freeze_params=False):
	if freeze_params:
		for name, parameter in model.named_parameters():
			if name.startswith('backbone'):
				parameter.requires_grad = False


if __name__ == '__main__':
	model = build_models(1, 200)
	print(model)
