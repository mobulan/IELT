from os.path import join
from typing import Tuple
from typing import Union, Sequence

import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy import io
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import *


class CUB(VisionDataset):
	"""`CUB-200-2011 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_ Dataset.
		Args:
			root (string): Root directory of the dataset.
			train (bool, optional): If True, creates dataset from training set, otherwise
			   creates from test set.
			transform (callable, optional): A function/transform that  takes in an PIL image
			   and returns a transformed version. E.g, ``transforms.RandomCrop``
			target_transform (callable, optional): A function/transform that takes in the
			   target and transforms it.
			download (bool, optional): If true, downloads the dataset from the internet and
			   puts it in root directory. If dataset is already downloaded, it is not
			   downloaded again.
	"""
	base_folder = 'CUB_200_2011/images'
	# url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
	file_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
	filename = 'CUB_200_2011.tgz'
	tgz_md5 = '97eceeb196236b17998738112f37df78'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(CUB, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train
		if download:
			self._download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

	def _load_metadata(self):
		images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
		                     names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
		                                 sep=' ', names=['img_id', 'target'])
		train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
		                               sep=' ', names=['img_id', 'is_training_img'])

		data = images.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')

		class_names = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'classes.txt'),
		                          sep=' ', names=['class_name'], usecols=[1])
		self.class_names = class_names['class_name'].to_list()
		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

	def _check_integrity(self):
		try:
			self._load_metadata()
		except Exception:
			return False

		for index, row in self.data.iterrows():
			filepath = os.path.join(self.root, self.base_folder, row.filepath)
			if not os.path.isfile(filepath):
				print(filepath)
				return False
		return True

	def _download(self):
		import tarfile

		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		download_file_from_google_drive(self.file_id, self.root, self.filename, self.tgz_md5)

		with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
			tar.extractall(path=self.root)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = sample.target - 1  # Targets start at 1 by default, so shift to 0
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target


class Cars(VisionDataset):
	"""`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset.
	Args:
		root (string): Root directory of the dataset.
		train (bool, optional): If True, creates dataset from training set, otherwise
			creates from test set.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		download (bool, optional): If true, downloads the dataset from the internet and
			puts it in root directory. If dataset is already downloaded, it is not
			downloaded again.
	"""
	file_list = {
		'imgs': ('http://imagenet.stanford.edu/internal/car196/car_ims.tgz', 'car_ims'),
		'annos': ('http://imagenet.stanford.edu/internal/car196/cars_annos.mat', 'cars_annos.mat')
	}

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(Cars, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train

		# if self._check_exists():
		# 	print('Files already downloaded and verified.')
		# elif download:
		# 	self._download()
		# else:
		# 	raise RuntimeError(
		# 		'Dataset not found. You can use download=True to download it.')

		loaded_mat = scipy.io.loadmat(os.path.join(self.root, self.file_list['annos'][1]))
		loaded_mat = loaded_mat['annotations'][0]
		self.samples = []
		for item in loaded_mat:
			if self.train != bool(item[-1][0]):
				path = str(item[0][0])
				label = int(item[-2][0]) - 1
				self.samples.append((path, label))

	def __getitem__(self, index):
		path, target = self.samples[index]
		path = os.path.join(self.root, path)

		image = self.loader(path)
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def __len__(self):
		return len(self.samples)

	def _check_exists(self):
		print(os.path.join(self.root, self.file_list['imgs'][1]))
		return os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))

	def _download(self):
		print('Downloading...')
		for url, filename in self.file_list.values():
			download_url(url, root=self.root, filename=filename)
		print('Extracting...')
		archive = os.path.join(self.root, self.file_list['imgs'][1])
		extract_archive(archive)


# class Cars(Dataset):
#
# 	def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None):
# 		"""
# 		Args:
# 			mat_anno (string): Path to the MATLAB annotation file.
# 			data_dir (string): Directory with all the images.
# 			transform (callable, optional): Optional transform to be applied
# 				on a sample.
# 		"""
#
# 		self.full_data_set = io.loadmat(mat_anno)
# 		self.car_annotations = self.full_data_set['annotations']
# 		self.car_annotations = self.car_annotations[0]
#
# 		if cleaned is not None:
# 			cleaned_annos = []
# 			print("Cleaning up datas set (only take pics with rgb chans)...")
# 			clean_files = np.loadtxt(cleaned, dtype=str)
# 			for c in self.car_annotations:
# 				if c[-1][0] in clean_files:
# 					cleaned_annos.append(c)
# 			self.car_annotations = cleaned_annos
#
# 		self.car_names = scipy.io.loadmat(car_names)['class_names']
# 		self.car_names = np.array(self.car_names[0])
#
# 		self.data_dir = data_dir
# 		self.transform = transform
#
# 	def __len__(self):
# 		return len(self.car_annotations)
#
# 	def __getitem__(self, idx):
# 		img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
# 		image = Image.open(img_name).convert('RGB')
# 		car_class = self.car_annotations[idx][-2][0][0]
# 		car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
# 		assert car_class < 196
#
# 		if self.transform:
# 			image = self.transform(image)
#
# 		# return image, car_class, img_name
# 		return image, car_class
#
# 	def map_class(self, id):
# 		id = np.ravel(id)
# 		ret = self.car_names[id - 1][0][0]
# 		return ret
#
# 	def show_batch(self, img_batch, class_batch):
#
# 		for i in range(img_batch.shape[0]):
# 			ax = plt.subplot(1, img_batch.shape[0], i + 1)
# 			title_str = self.map_class(int(class_batch[i]))
# 			img = np.transpose(img_batch[i, ...], (1, 2, 0))
# 			ax.imshow(img)
# 			ax.set_title(title_str.__str__(), {'fontsize': 5})
# 			plt.tight_layout()


class Dogs(VisionDataset):
	"""`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
		Args:
			root (string): Root directory of the dataset.
			train (bool, optional): If True, creates dataset from training set, otherwise
			   creates from test set.
			transform (callable, optional): A function/transform that  takes in an PIL image
			   and returns a transformed version. E.g, ``transforms.RandomCrop``
			target_transform (callable, optional): A function/transform that takes in the
			   target and transforms it.
			download (bool, optional): If true, downloads the dataset from the internet and
			   puts it in root directory. If dataset is already downloaded, it is not
			   downloaded again.
	"""
	download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
		super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

		self.loader = default_loader
		self.train = train

		if download:
			self.download()

		split = self.load_split()

		self.images_folder = join(self.root, 'Images')
		self.annotations_folder = join(self.root, 'Annotation')
		self._breeds = list_dir(self.images_folder)

		self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

		self._flat_breed_images = self._breed_images

	def __len__(self):
		return len(self._flat_breed_images)

	def __getitem__(self, index):
		image_name, target = self._flat_breed_images[index]
		image_path = join(self.images_folder, image_name)
		image = self.loader(image_path)

		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def download(self):
		import tarfile

		if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
			if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
				print('Files already downloaded and verified')
				return

		for filename in ['images', 'annotation', 'lists']:
			tar_filename = filename + '.tar'
			url = self.download_url_prefix + '/' + tar_filename
			download_url(url, self.root, tar_filename, None)
			print('Extracting downloaded file: ' + join(self.root, tar_filename))
			with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
				tar_file.extractall(self.root)
			os.remove(join(self.root, tar_filename))

	def load_split(self):
		if self.train:
			split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
		else:
			split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
			labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

		split = [item[0][0] for item in split]
		labels = [item[0] - 1 for item in labels]
		return list(zip(split, labels))

	def stats(self):
		counts = {}
		for index in range(len(self._flat_breed_images)):
			image_name, target_class = self._flat_breed_images[index]
			if target_class not in counts.keys():
				counts[target_class] = 1
			else:
				counts[target_class] += 1

		print("%d samples spanning %d classes (avg %f per class)" % (len(self._flat_breed_images), len(counts.keys()),
		                                                             float(len(self._flat_breed_images)) / float(
			                                                             len(counts.keys()))))

		return counts


class Aircraft(Dataset):
	img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

	def __init__(self, root, train=True, transform=None):
		self.train = train
		self.root = root
		self.class_type = 'variant'
		self.split = 'trainval' if self.train else 'test'
		self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
		                                 'images_%s_%s.txt' % (self.class_type, self.split))
		self.transform = transform

		(image_ids, targets, classes, class_to_idx) = self.find_classes()
		samples = self.make_dataset(image_ids, targets)

		self.loader = default_loader

		self.samples = samples
		self.classes = classes
		self.class_to_idx = class_to_idx

	def __getitem__(self, index):
		path, target = self.samples[index]
		sample = self.loader(path)
		sample = self.transform(sample)
		return sample, target

	def __len__(self):
		return len(self.samples)

	def find_classes(self):
		# read classes file, separating out image IDs and class names
		image_ids = []
		targets = []
		with open(self.classes_file, 'r') as f:
			for line in f:
				split_line = line.split(' ')
				image_ids.append(split_line[0])
				targets.append(' '.join(split_line[1:]))

		# index class names
		classes = np.unique(targets)
		class_to_idx = {classes[i]: i for i in range(len(classes))}
		targets = [class_to_idx[c] for c in targets]

		return image_ids, targets, classes, class_to_idx

	def make_dataset(self, image_ids, targets):
		assert (len(image_ids) == len(targets))
		images = []
		for i in range(len(image_ids)):
			item = (os.path.join(self.root, self.img_folder,
			                     '%s.jpg' % image_ids[i]), targets[i])
			images.append(item)
		return images


class NABirds(VisionDataset):
	"""`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.
		Args:
			root (string): Root directory of the dataset.
			train (bool, optional): If True, creates dataset from training set, otherwise
			   creates from test set.
			transform (callable, optional): A function/transform that  takes in an PIL image
			   and returns a transformed version. E.g, ``transforms.RandomCrop``
			target_transform (callable, optional): A function/transform that takes in the
			   target and transforms it.
			download (bool, optional): If true, downloads the dataset from the internet and
			   puts it in root directory. If dataset is already downloaded, it is not
			   downloaded again.
	"""
	base_folder = 'images'
	filename = 'nabirds.tar.gz'
	md5 = 'df21a9e4db349a14e2b08adfd45873bd'

	def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
		super(NABirds, self).__init__(root, transform=transform, target_transform=target_transform)
		if download is True:
			msg = ("The dataset is no longer publicly accessible. You need to "
			       "download the archives externally and place them in the root "
			       "directory.")
			raise RuntimeError(msg)
		elif download is False:
			msg = ("The use of the download flag is deprecated, since the dataset "
			       "is no longer publicly accessible.")
			warnings.warn(msg, RuntimeWarning)

		dataset_path = root
		# if not os.path.isdir(dataset_path):
		# 	if not check_integrity(os.path.join(root, self.filename), self.md5):
		# 		raise RuntimeError('Dataset not found or corrupted.')
		# 	extract_archive(os.path.join(root, self.filename))
		self.loader = default_loader
		self.train = train

		image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
		                          sep=' ', names=['img_id', 'filepath'])
		image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
		                                 sep=' ', names=['img_id', 'target'])
		# Since the raw labels are non-continuous, map them to new ones
		self.label_map = self.get_continuous_class_map(image_class_labels['target'])
		train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
		                               sep=' ', names=['img_id', 'is_training_img'])
		data = image_paths.merge(image_class_labels, on='img_id')
		self.data = data.merge(train_test_split, on='img_id')
		# Load in the train / test split
		if self.train:
			self.data = self.data[self.data.is_training_img == 1]
		else:
			self.data = self.data[self.data.is_training_img == 0]

		# Load in the class data
		self.class_names = self.load_class_names(dataset_path)
		self.class_hierarchy = self.load_hierarchy(dataset_path)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, self.base_folder, sample.filepath)
		target = self.label_map[sample.target]
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return img, target

	def get_continuous_class_map(self, class_labels):
		label_set = set(class_labels)
		return {k: i for i, k in enumerate(label_set)}

	def load_class_names(self, dataset_path=''):
		names = {}

		with open(os.path.join(dataset_path, 'classes.txt')) as f:
			for line in f:
				pieces = line.strip().split()
				class_id = pieces[0]
				names[class_id] = ' '.join(pieces[1:])

		return names

	def load_hierarchy(self, dataset_path=''):
		parents = {}

		with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
			for line in f:
				pieces = line.strip().split()
				child_id, parent_id = pieces
				parents[child_id] = parent_id

		return


class OxfordFlowers(Dataset):
	def __init__(self, root, train=True, transform=None):
		self.transform = transform
		self.root = root
		self.loader = default_loader
		train_set = pd.read_csv(os.path.join(self.root, 'train.txt'),
		                        sep=' ', names=['img_path', 'target'])
		test_set = pd.read_csv(os.path.join(self.root, 'test.txt'),
		                       sep=' ', names=['img_path', 'target'])
		if train:
			self.data = train_set
		else:
			self.data = test_set

	def __getitem__(self, idx):
		sample = self.data.iloc[idx]
		path = os.path.join(self.root, sample.img_path)
		target = sample.target
		img = self.loader(path)

		if self.transform is not None:
			img = self.transform(img)
		return img, target

	def __len__(self):
		return len(self.data)


class OxfordIIITPet(VisionDataset):
	"""`Oxford-IIIT Pet Dataset   <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_.

	Args:
		root (string): Root directory of the dataset.
		split (string, optional): The dataset split, supports ``"trainval"`` (default) or ``"test"``.
		target_types (string, sequence of strings, optional): Types of target to use. Can be ``category`` (default) or
			``segmentation``. Can also be a list to output a tuple with all specified target types. The types represent:

				- ``category`` (int): Label for one of the 37 pet categories.
				- ``segmentation`` (PIL image): Segmentation trimap of the image.

			If empty, ``None`` will be returned as target.

		transform (callable, optional): A function/transform that  takes in a PIL image and returns a transformed
			version. E.g, ``transforms.RandomCrop``.
		target_transform (callable, optional): A function/transform that takes in the target and transforms it.
		download (bool, optional): If True, downloads the dataset from the internet and puts it into
			``root/oxford-iiit-pet``. If dataset is already downloaded, it is not downloaded again.
	"""

	_RESOURCES = (
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
		("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", "95a8c909bbe2e81eed6a22bccdf3f68f"),
	)
	_VALID_TARGET_TYPES = ("category", "segmentation")

	def __init__(
			self,
			root: str,
			train: bool = True,
			transform: Optional[Callable] = None,
			target_types: Union[Sequence[str], str] = "category",
			target_transform: Optional[Callable] = None,
			download: bool = False,
	):
		if train:
			split = "trainval"
		else:
			split = "test"
		self._split = verify_str_arg(split, "split", ("trainval", "test"))
		if isinstance(target_types, str):
			target_types = [target_types]
		self._target_types = [
			verify_str_arg(target_type, "target_types", self._VALID_TARGET_TYPES) for target_type in target_types
		]

		super().__init__(root, transform=transform, target_transform=target_transform)
		self._images_folder = os.path.join(self.root, "images")
		self._anns_folder = os.path.join(self.root, "annotations")
		self._segs_folder = os.path.join(self.root, "trimaps")

		if download:
			self._download()

		if not self._check_exists():
			raise RuntimeError("Dataset not found. You can use download=True to download it")

		image_ids = []
		self._labels = []
		with open(os.path.join(self._anns_folder, f"{self._split}.txt")) as file:
			for line in file:
				image_id, label, *_ = line.strip().split()
				image_ids.append(image_id)
				self._labels.append(int(label) - 1)

		self.classes = [
			" ".join(part.title() for part in raw_cls.split("_"))
			for raw_cls, _ in sorted(
				{(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, self._labels)},
				key=lambda image_id_and_label: image_id_and_label[1],
			)
		]
		self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

		self._images = [os.path.join(self._images_folder, f"{image_id}.jpg") for image_id in image_ids]
		self._segs = [os.path.join(self._segs_folder, f"{image_id}.png") for image_id in image_ids]

	def __len__(self) -> int:
		return len(self._images)

	def __getitem__(self, idx: int) -> Tuple[Any, Any]:
		image = Image.open(self._images[idx]).convert("RGB")

		target: Any = []
		for target_type in self._target_types:
			if target_type == "category":
				target.append(self._labels[idx])
			else:  # target_type == "segmentation"
				target.append(Image.open(self._segs[idx]))

		if not target:
			target = None
		elif len(target) == 1:
			target = target[0]
		else:
			target = tuple(target)

		if self.transform:
			image = self.transform(image)

		return image, target

	def _check_exists(self) -> bool:
		for folder in (self._images_folder, self._anns_folder):
			if not (os.path.exists(folder) and os.path.isdir(folder)):
				return False
		else:
			return True

	def _download(self) -> None:
		if self._check_exists():
			return

		for url, md5 in self._RESOURCES:
			download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)


if __name__ == '__main__':
	root = "D:\\实验\\数据集\\cars"
	train_set = Cars(root, train=False, transform=None)
	print(len(train_set))
