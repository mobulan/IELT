import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist


class Timer:
	def __init__(self):
		self.times = []
		self.start()
		self.avg = 0.
		self.count = 0.
		self.sum = 0.

	def start(self):
		self.tik = time.time()

	def stop(self):
		t = time.time() - self.tik
		self.times.append(t)
		self.sum += t
		self.count += 1
		self.avg = self.sum / self.count
		return self.times[-1]

	def cumsum(self):
		return np.array(self.times).cumsum().tolist()


def simple_accuracy(preds, labels):
	count = preds.shape[0]
	result = (preds == labels).sum()
	return result / count


def reduce_mean(tensor):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= get_world_size()
	return rt


def count_parameters(model):
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	return params / 1000000


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
	save_state = {'model': model.state_dict(),
	              'optimizer': optimizer.state_dict(),
	              'lr_scheduler': lr_scheduler.state_dict(),
	              'max_accuracy': max_accuracy,
	              'scaler': loss_scaler.state_dict(),
	              'epoch': epoch,
	              'config': config}

	# save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
	save_path = os.path.join(config.data.log_path, "checkpoint.bin")
	torch.save(save_state, save_path)
	print("----- Saved model checkpoint to", config.data.log_path, '-----')


def save_preds(preds, y, all_preds=None, all_label=None, ):
	if all_preds is None:
		all_preds = preds.clone().detach()
		all_label = y.clone().detach()
	else:
		all_preds = torch.cat((all_preds, preds), 0)
		all_label = torch.cat((all_label, y), 0)
	return all_preds, all_label


def load_checkpoint(config, model, optimizer, scheduler, loss_scaler, log):
	log.info(f"--------------- Resuming form {config.model.resume} ---------------")
	checkpoint = torch.load(config.model.resume, map_location='cpu')
	msg = model.load_state_dict(checkpoint['model'], strict=False)
	log.info(msg)
	max_accuracy = 0.0
	if not config.eval_mode and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		config.defrost()
		config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
		config.freeze()
		if 'scaler' in checkpoint:
			loss_scaler.load_state_dict(checkpoint['scaler'])
		log.info(f"----- loaded successfully '{config.model.resume}' -- epoch {checkpoint['epoch']} -----")
		if 'max_accuracy' in checkpoint:
			max_accuracy = checkpoint['max_accuracy']

	del checkpoint
	torch.cuda.empty_cache()
	return max_accuracy


def eval_accuracy(all_preds, all_label, config):
	accuracy = simple_accuracy(all_preds, all_label)
	if config.local_rank != -1:
		dist.barrier(device_ids=[config.local_rank])
		val_accuracy = reduce_mean(accuracy)
	else:
		val_accuracy = accuracy
	return val_accuracy.item()


class NativeScalerWithGradNormCount:
	state_dict_key = "amp_scaler"

	def __init__(self):
		self._scaler = torch.cuda.amp.GradScaler()

	def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
		self._scaler.scale(loss).backward(create_graph=create_graph)
		if update_grad:
			if clip_grad is not None:
				assert parameters is not None
				self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
				norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
			else:
				self._scaler.unscale_(optimizer)
				norm = ampscaler_get_grad_norm(parameters)
			self._scaler.step(optimizer)
			self._scaler.update()
		else:
			norm = None
		return norm

	def state_dict(self):
		return self._scaler.state_dict()

	def load_state_dict(self, state_dict):
		self._scaler.load_state_dict(state_dict)


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
	if isinstance(parameters, torch.Tensor):
		parameters = [parameters]
	parameters = [p for p in parameters if p.grad is not None]
	norm_type = float(norm_type)
	if len(parameters) == 0:
		return torch.tensor(0.)
	device = parameters[0].grad.device
	if norm_type == math.inf:
		total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
	else:
		total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
		                                                norm_type).to(device) for p in parameters]), norm_type)
	return total_norm


def get_world_size():
	if not dist.is_available():
		return 1
	if not dist.is_initialized():
		return 1
	return dist.get_world_size()
