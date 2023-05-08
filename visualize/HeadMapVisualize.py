import sys
sys.path.append("..")
import os
import torch
from PIL import Image
from torchvision import transforms,datasets
from torchvision.transforms import InterpolationMode
from utils.eval import load_checkpoint
from main import build_model
from setup import config
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy as np

transform = transforms.Compose([
									transforms.Resize((600,), InterpolationMode.BILINEAR),
									# transforms.Resize((448, 448), InterpolationMode.BILINEAR),
			                        transforms.CenterCrop((448, 448)),
									# transforms.ColorJitter(brightness=0.4, contrast=0.4),
			                        transforms.ToTensor()])
base_root = '../figures/paper_img/headmap'

def show_grid_images(imgs, rows, cols, titles=None, scale=3, cmap='rainbow'):
	'''打印若干行列的图片'''
	figsize = (cols * scale, rows * scale)
	_,axes = plt.subplots(rows,cols,figsize=figsize)
	axes = axes.flatten()
	for i , ax in enumerate(axes):
		ax.imshow(imgs[i],cmap=cmap)
		# ax.imshow(show_img.permute(1,2,0),alpha=0.5)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.axes.set_title(titles[i])
	return axes

def show_individual(imgs,layer_id):
	'''打印每一层注意力图的单张图片'''

	# 将每个头的注意力图汇总平均
	os.makedirs(f'{base_root}/converge', exist_ok=True)
	sum_img = imgs.mean(0)
	plt.imsave(f'{base_root}/converge/layer {layer_id:2}.jpg', sum_img)

	# 每个头的注意力图单独保存
	save_root = f'{base_root}/layer {layer_id + 1:2} attention map'
	os.makedirs(save_root, exist_ok=True)
	number = imgs.shape[0]
	for i in range(number):
		# plt.imshow(imgs[i])
		plt.imsave(f'{save_root}/map {i:2}.jpg',imgs[i])
		# plt.show()

def normal_weights(weights):
	'''调整注意力图'''

	# 每一层的注意力图尺寸都是(k,n+1,n+1)
	print(weights.shape)
	# 只取class token那一行所以第三维是（0，），取除了class token的n列，所以是（1：）
	attention_map = weights[:, :, 0, 1:]
	attention_map = attention_map.reshape(11, 12, 28, 28)
	# 放大图像，让保存的图像在ps里变得清晰一些
	attention_map = ndimage.zoom(attention_map, (1, 1, 16, 16), order=0)
	return attention_map

def build_test_model(checkpoint_root):

	config.defrost()
	config.write = False
	config.rank = 0
	config.misc.eval_mode = True
	config.parameters.assess = True
	config.model.resume = checkpoint_root
	config.freeze()
	model,_ = build_model(config,200)
	load_checkpoint(config,model)

	return model

def preprocess_img(img_path):
	img = Image.open(img_path)
	show_img = img.copy()
	show_img = transform(show_img)
	plt.imshow(show_img.permute(1, 2, 0))
	os.makedirs(base_root, exist_ok=True)
	plt.imsave(f'{base_root}/sample.jpg', show_img.permute(1, 2, 0).detach().cpu().numpy())
	plt.show()
	# 分别返回准备向量化的img和展示用的RGB格式img
	return img,show_img

def head_attention_map(model,img,show_img):
	model.eval()
	# 处理图像
	img = transform(img)
	img = img.unsqueeze(0).to(device='cuda')

	# 固定每层选择数，当然也可以不固定，用检查点中保存的层选择数
	model.encoder.select_num = torch.ones(11, device='cuda')*12
	_, assist_list = model(img)

	# 调整参考数据
	layer_weights, layer_selected, layer_score, final_selected = assist_list
	layer_weights = torch.stack(layer_weights).squeeze().cpu().detach().numpy()
	layer_selected = [a.cpu().float() - 1 for a in layer_selected]
	layer_score = [a.cpu().float() for a in layer_score]
	# Cross-layer Refinement module的选择结果
	final_selected = final_selected.squeeze().cpu() - 1
	attention_map = normal_weights(layer_weights)

	for i in range(model.encoder.layer_num-1):
		# 该层每个头的注意力图（网格展示，单张保存）
		print(f'Layer {i+1} Attention Map of Each Head')
		axes = show_grid_images(attention_map[i],3,4)
		show_individual(attention_map[i],i)
		plt.show()

		# 该层Multi-head Voting Module最后的选择结果
		print(f'Layer {i+1} Final Selected')
		count = np.bincount(layer_selected[i],minlength=784)
		count = count.reshape(28,28)
		count = ndimage.zoom(count,16,order=0)
		plt.imshow(count,cmap='gray')
		plt.imshow(show_img.permute(1,2,0),alpha=0.5)
		final_root = f'{base_root}/final_select'
		os.makedirs(final_root,exist_ok=True)
		# 弄成黑白，方便ps里正片叠底制作图片
		plt.imsave(f'{final_root}/final_select {i+1:2}.jpg',count,cmap='gray')
		plt.show()

		# 该层的增强后的得分图
		layer_score[i] = layer_score[i].reshape(28, 28).detach().numpy()
		layer_score[i] = ndimage.zoom(layer_score[i], 16, order=0)
		score_root = f'{base_root}/scoremap'
		os.makedirs(score_root,exist_ok=True)
		plt.imsave(f'{score_root}/scoremap {i+1:2}.jpg',layer_score[i],cmap='hot')
		plt.show()


if __name__ == '__main__':
	img_path = os.path.join('../figures/Crested_Auklet_0003_794962.jpg')
	model = build_test_model('../output/cub/IELT 05-08_16-10/checkpoint.bin')
	tensor_img, show_img = preprocess_img(img_path)
	head_attention_map(model, tensor_img, show_img)