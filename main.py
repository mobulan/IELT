from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.build import build_models, freeze_backbone
from setup import config, log
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler


def build_model(config, num_classes):
	model = build_models(config, num_classes)
	if torch.__version__[0] == 2:
		model = torch.compile(model, mode="max-autotune")
	model.cuda()
	freeze_backbone(model, config.train.freeze_backbone)
	model_without_ddp = model
	n_parameters = count_parameters(model)

	config.defrost()
	config.model.num_classes = num_classes
	config.model.parameters = f'{n_parameters:.3f}M'
	config.freeze()
	if config.local_rank in [-1, 0]:
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(), rank=config.local_rank)
		log.save(model)
	return model, model_without_ddp


def main(config):
	# Timer
	prepare_timer = Timer()
	prepare_timer.start()
	train_timer = Timer()
	eval_timer = Timer()

	# Initialize the Tensorboard Writer
	writer = None
	if config.write: writer = SummaryWriter(config.data.log_path)

	# Prepare dataset
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn = build_loader(config)
	step_per_epoch = len(train_loader)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch

	# Build model
	model, model_without_ddp = build_model(config, num_classes)
	if not config.model.baseline_model:
		model.encoder.warm_steps = config.parameters.update_warm * step_per_epoch
	if config.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
		                                                  broadcast_buffers=False,
		                                                  find_unused_parameters=False)
	optimizer = build_optimizer(config, model, backbone_low_lr=False)
	loss_scaler = NativeScalerWithGradNormCount()
	# Build learning rate scheduler

	if config.train.lr_epoch_update:
		scheduler = build_scheduler(config, optimizer, 1)
	else:
		scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# Determine criterion
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	if config.data.mixup > 0.:
		criterion = SoftTargetCrossEntropy()
	elif config.model.label_smooth:
		criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# Function Mode
	if config.model.resume:
		best_acc = load_checkpoint(config, model_without_ddp, optimizer, scheduler, loss_scaler, log)
		best_epoch = config.start_epoch
		accuracy, loss = valid(config, model, test_loader, best_epoch, train_accuracy, writer)
		log.info(f'Epoch {best_epoch:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
		         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
		         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:
			return

	if config.misc.throughput:
		throughput(test_loader, model, log, config.local_rank)
		return

	# Record result in Markdown Table
	mark_table = PMarkdownTable(log, ['Epoch', 'Accuracy', 'Best Accuracy',
	                                  'Best Epoch', 'Loss'], rank=config.local_rank)

	# End preparation
	torch.cuda.synchronize()
	prepare_time = prepare_timer.stop()
	PSetting(log, 'Training Information',
	         ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
	          'Warm Epochs'],
	         [train_samples, test_samples, total_batch_size,
	          f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
	         newline=2, rank=config.local_rank)

	# Train Function
	sub_title(log, 'Start Training', rank=config.local_rank)
	for epoch in range(config.train.start_epoch, config.train.epochs):
		train_timer.start()
		if config.local_rank != -1:
			train_loader.sampler.set_epoch(epoch)
		if not config.misc.eval_mode:
			train_accuracy = train_one_epoch(config, model, criterion, train_loader, optimizer,
			                                 epoch, scheduler, loss_scaler, mixup_fn, writer)
		train_timer.stop()

		# Eval Function
		eval_timer.start()
		if epoch < 5 or (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:
			accuracy, loss = valid(config, model, test_loader, epoch, train_accuracy, writer)
			if config.local_rank in [-1, 0]:
				if best_acc < accuracy:
					best_acc = accuracy
					best_epoch = epoch + 1
					if config.write and epoch > 10 and config.train.checkpoint:
						save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
				log.info(f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
				         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
				         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
				if config.write:
					mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
					                     f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
			pass  # Eval
		eval_timer.stop()
		pass  # Train

	# Finish Training
	if writer is not None:
		writer.close()
	train_time = train_timer.sum / 60
	eval_time = eval_timer.sum / 60
	total_time = train_time + eval_time
	PSetting(log, "Finish Training",
	         ['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Total Time'],
	         [f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time:.2f} min'],
	         newline=2, rank=config.local_rank)
	pass


def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, scheduler, loss_scaler,
                    mixup_fn=None, writer=None):
	model.train()
	optimizer.zero_grad()

	step_per_epoch = len(train_loader)
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()
	epochs = config.train.epochs
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Train {epoch + 1:^3}/{epochs:^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])
	all_preds, all_label = None, None
	for step, (x, y) in enumerate(train_loader):
		global_step = epoch * step_per_epoch + step
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		if mixup_fn:
			x, y = mixup_fn(x, y)
		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			if config.model.baseline_model:
				logits = model(x)
			else:
				logits = model(x, y)

		logits, loss = loss_in_iters(logits, y, criterion)

		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
		                        parameters=model.parameters(), create_graph=is_second_order)

		optimizer.zero_grad()
		if config.train.lr_epoch_update:
			scheduler.step_update(epoch + 1)
		else:
			scheduler.step_update(global_step + 1)
		loss_scale_value = loss_scaler.state_dict()["scale"]

		if mixup_fn is None:
			preds = torch.argmax(logits, dim=-1)
			all_preds, all_label = save_preds(preds, y, all_preds, all_label)
		torch.cuda.synchronize()

		if grad_norm is not None:
			norm_meter.update(grad_norm)
		scaler_meter.update(loss_scale_value)
		loss_meter.update(loss.item(), y.size(0))

		lr = optimizer.param_groups[0]['lr']
		if writer:
			writer.add_scalar("train/loss", loss_meter.val, global_step)
			writer.add_scalar("train/lr", lr, global_step)
			writer.add_scalar("train/grad_norm", norm_meter.val, global_step)
			writer.add_scalar("train/scaler_meter", scaler_meter.val, global_step)

		# set_postfix require dic input
		p_bar.set_postfix(loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		p_bar.update()

	# After Training an Epoch
	p_bar.close()
	train_accuracy = eval_accuracy(all_preds, all_label, config) if mixup_fn is None else 0.0
	return train_accuracy


def loss_in_iters(output, targets, criterion):
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets)
	else:
		logits, loss = output
		return logits, loss


@torch.no_grad()
def valid(config, model, test_loader, epoch=-1, train_acc=0.0, writer=None):
	criterion = torch.nn.CrossEntropyLoss()
	model.eval()

	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()

	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			logits = model(x)

		logits, loss = loss_in_iters(logits, y, criterion)

		acc = accuracy(logits, y)[0]
		if config.local_rank != -1:
			acc = reduce_mean(acc)

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg), loss="%2.5f" % loss_meter.avg,
		                  tra="{:2.3f}".format(train_acc * 100))
		p_bar.update()
		pass

	p_bar.close()
	if writer:
		writer.add_scalar("test/accuracy", acc_meter.avg, epoch + 1)
		writer.add_scalar("test/loss", loss_meter.avg, epoch + 1)
		writer.add_scalar("test/train_acc", train_acc * 100, epoch + 1)
	return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, log, rank):
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return


if __name__ == '__main__':
	main(config)
