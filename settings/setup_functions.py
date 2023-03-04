import random
import socket

import numpy as np
import torch.backends.cudnn

from utils.eval import get_world_size
from utils.info import *


def SetupConfig(config, cfg_file=None):
	if cfg_file:
		config.defrost()
		print('-' * 28, '{:^22}'.format(cfg_file), '-' * 28)
		config.merge_from_file(cfg_file)
		config.freeze()
	return config


def SetupLogs(config, rank=0):
	write = config.write
	if rank not in [-1, 0]: return
	# 建立log对象
	if write:
		os.makedirs(config.data.log_path, exist_ok=True)
	log = Log(fname=config.data.log_path, write=write)
	# 输出
	PTitle(log, config.local_rank)
	PSetting(log, 'Data Settings', config.data.keys(), config.data.values(), newline=2, rank=config.local_rank)
	PSetting(log, 'Hyper Parameters', config.parameters.keys(), config.parameters.values(), rank=config.local_rank)
	PSetting(log, 'Training Settings', config.train.keys(), config.train.values(), rank=config.local_rank)
	PSetting(log, 'Other Settings', config.misc.keys(), config.misc.values(), rank=config.local_rank)

	# 返回log实例
	return log


def SetupDevice():
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		rank = int(os.environ["RANK"])
		world_size = int(os.environ['WORLD_SIZE'])
		torch.cuda.set_device(rank)
		torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
		torch.distributed.barrier()
	else:
		rank = -1
	nprocess = torch.cuda.device_count()
	torch.cuda.set_device(rank)
	# torch.backends.cudnn.benchmark = True
	return nprocess, rank


def SetSeed(config):
	seed = config.misc.seed + config.local_rank
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)


def ScaleLr(config):
	base_lr = config.train.lr * config.data.batch_size * get_world_size() / 512.0
	return base_lr


def LocateDatasets(config=None):
	def HostIp():
		try:
			s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
			s.connect(('8.8.8.8', 80))
			ip = s.getsockname()[0]
		finally:
			s.close()
		return ip

	ip = HostIp()
	print(ip)
	address = ip.split('.')[3]
	data_root = config.data.data_root
	batch_size = config.data.batch_size
	if ip == '210.45.215.179':
		data_root = '/DATA/meiyiming/ly/dataset'
		batch_size = config.data.batch_size // 2
	elif ip == '210.45.215.197':
		data_root = '/DATA/linjing/ly/dataset'
		batch_size = config.data.batch_size // 2
	# elif address == '65' or address == '199' or address == '100':
	# 	data_root = 'D:\\Experiment\\Datasets'
	# 	batch_size = config.data.batch_size // 4
	return data_root, batch_size
