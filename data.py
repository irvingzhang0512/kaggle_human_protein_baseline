from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from imgaug import augmenters as iaa
import random
import pathlib
import cv2

random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)


class HumanDataset(Dataset):
    def __init__(self, file_names, labels, base_path, augument=True, mode="train"):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
        self.file_names = [os.path.join(base_path, file_name) for file_name in file_names]
        self.labels = labels
        self.augument = augument
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = self.read_images(index)
        if not self.mode == "test":
            y = self.labels[index]
        else:
            y = self.file_names[index]
        if self.augument:
            img = self.augumentor(img)
        img = T.Compose([T.ToPILImage(), T.ToTensor()])(img)
        return img.float(), y

    def read_images(self, index):
        filename = self.file_names[index]
        images = np.zeros(shape=(512, 512, 4))
        r = np.array(Image.open(filename + "_red.png"))
        g = np.array(Image.open(filename + "_green.png"))
        b = np.array(Image.open(filename + "_blue.png"))
        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)
        y = np.array(Image.open(filename + "_yellow.png"))
        images[:, :, 3] = y.astype(np.uint8)
        images = images.astype(np.uint8)  # [0, 255]
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images, (config.img_width, config.img_height))

    def augumentor(self, image):
        augment_img = iaa.SomeOf((0, 6), [
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
        ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug
