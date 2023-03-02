import os
import platform
import time

import torch


class Log:
	def __init__(self, fname=None, write=True):
		super(Log, self).__init__()
		self.write = write
		time_name = time.strftime('%Y-%m-%d_%H-%M', time.localtime())
		if fname:
			self.fname = os.path.join(fname, time_name + '.log')
			self.tname = os.path.join(fname, time_name + '.md')
			self.mname = os.path.join(fname, 'model' + '.txt')
		else:
			self.fname = time_name + '.log'
			self.tname = time_name + '.md'
			self.tname = time_name + '.txt'

		# Create the file
		if self.write:
			with open(self.fname, 'w') as f:
				pass

			with open(self.tname, 'w') as f:
				pass

			with open(self.mname, 'w') as f:
				pass

	def info(self, *info, end='\n'):
		print(*info, flush=True, end=end)
		if self.write:
			with open(self.fname, 'a+') as f:
				print(*info, file=f, flush=True, end=end)

	def markdown(self, *info, end='\n'):
		if self.write:
			with open(self.tname, 'a+') as f:
				print(*info, file=f, flush=True, end=end)
		pass

	def save(self, *info, ):
		if self.write:
			with open(self.mname, 'w') as f:
				print(*info, file=f, flush=True)


def PTitle(log, rank=0):
	if rank not in [-1, 0]: return
	log.info('=' * 80)
	log.info(' Fine-Grained Visual Classification via Internal Ensemble Learning Transformer\n'
	         '                            Pytorch Implementation')
	log.info('=' * 80)
	log.info('Author:\t\tXu Qin,\t\tWang Jiahui,\t\tJiang Bo,\t\tLuo Bin\n'
	         'Institute:\tAnhui University\t\t\t\t\tDate: 2023-02-13')
	log.info('-' * 80)
	log.info(f'Python Version: {platform.python_version()}         '
	         f'Pytorch Version: {torch.__version__}         Cuda Version: {torch.version.cuda}')
	log.info('-' * 80, '\n')
	pass


class PMarkdownTable:
	def __init__(self, log, titles, rank=0):
		if rank not in [-1, 0]: return
		super(PMarkdownTable, self).__init__()
		title_line = '| '
		align_line = '| '
		for i in range(len(titles)):
			title_line = title_line + titles[i] + ' |'
			align_line = align_line + '--- |'
		log.markdown(title_line)
		log.markdown(align_line)

	def add(self, log, values, rank=0):
		if rank not in [-1, 0]: return
		value_line = '| '
		for i in range(len(values)):
			value_line = value_line + str(values[i]) + '|'
		log.markdown(value_line)

	pass


def PSetting(log, title=None, param_name=None, values=None, newline=3, rank=0):
	if rank not in [-1, 0]: return
	if title is not None:
		log.info('=' * 28, '{:^22}'.format(title), '=' * 28)
	for i, (name, value) in enumerate(zip(param_name, values)):
		name = str(name)
		param_name = list(param_name)
		if isinstance(value, tuple):
			value = f'{value[0]},{value[1]}'
		if isinstance(value, list):
			value = str(value)
		if value is None:
			value = f'None'
		if newline == 3:
			if (i + 1) % newline == 0 and name != param_name[-1]:
				log.info(f'{name:14}{value :<12}')
				log.info('- ' * 40)
			else:
				log.info(f'{name:14}{value :<12}', end='  ')

		else:  # newline==2
			if len(name) < 14:
				if (i + 1) % newline == 0 and name != param_name[-1]:
					log.info(f'{name:14}{value :<23}')
					log.info('- ' * 40)
				else:
					log.info(f'{name:14}{value :<23}', end='   ')
			else:
				if (i + 1) % newline == 0 and name != param_name[-1]:
					log.info(f'{name:18}{value :<19}')
					log.info('- ' * 40)
				else:
					log.info(f'{name:18}{value :<19}', end='   ')
	log.info()


def sub_title(log, title, rank=0):
	if rank not in [-1, 0]: return
	if len(title) < 22:
		log.info('=' * 28, '{:^22}'.format(title), '=' * 28)
	elif len(title) < 30:
		log.info('=' * 24, '{:^30}'.format(title), '=' * 24)
	else:
		log.info('=' * 20, '{:^38}'.format(title), '=' * 20)


if __name__ == '__main__':
	log = Log(write=False)
	PTitle(log, -1)
