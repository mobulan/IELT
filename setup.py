from settings.defaults import _C
from settings.setup_functions import *

config = _C.clone()
cfg_file = os.path.join('configs', 'cub.yaml')
config = SetupConfig(config, cfg_file)
config.defrost()

## Log Name and Perferences
config.write = True			# comment it to disable all the log writing
config.train.checkpoint = True		# comment it to disable saving the checkpoint
config.misc.exp_name = f'{config.data.dataset}'
config.misc.log_name = f'IELT'
config.cuda_visible = '0,1,2,3'

# Environment Settings
config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                    + time.strftime(' %m-%d_%H-%M', time.localtime()))

config.model.pretrained = os.path.join(config.model.pretrained,
                                       config.model.name + config.model.pre_version + config.model.pre_suffix)
os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
os.environ['OMP_NUM_THREADS'] = '1'

# Setup Functions
config.nprocess, config.local_rank = SetupDevice()
config.data.data_root, config.data.batch_size = LocateDatasets(config)
config.train.lr = ScaleLr(config)
log = SetupLogs(config, config.local_rank)
if config.write and config.local_rank in [-1, 0]:
	with open(config.data.log_path + '/config.json', "w") as f:
		f.write(config.dump())
config.freeze()
SetSeed(config)
