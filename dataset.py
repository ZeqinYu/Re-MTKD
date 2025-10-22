import random
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, txt_file, img_size, resize=False):
        f = open(txt_file, 'r')
        self.files = f.readlines()
        f.close()
        self.img_size = img_size
        self.resize = resize
        transforms_list = [transforms.ToPILImage(), transforms.ToTensor()]
        transforms_mask_list = [transforms.ToPILImage(), transforms.ToTensor()]
        transforms_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transforms_list)
        self.transform_mask = transforms.Compose(transforms_mask_list)

    def __getitem__(self, idx):
        tp_path = self.files[idx].split(' ')[0]
        mask_path = self.files[idx].split(' ')[1].split('\n')[0]
        tp = cv2.cvtColor(cv2.imread(tp_path, 1), cv2.COLOR_BGR2RGB)
        height, width = tp.shape[:2]
        if mask_path == 'None':
            mask = np.zeros((height, width)).astype(np.float32)
        else:
            mask = cv2.imread(mask_path, 0)
        if 255 in mask:
            class_gt = 1
        else:
            class_gt = 0
        if tp is None or mask is None:
            raise IOError
        if self.resize:
            h, w, _ = tp.shape
            if [h, w] != self.img_size:
                tp = cv2.resize(tp, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        tp = self.transform(tp)
        mask = self.transform_mask(mask)
        mask = mask.float()

        return tp, mask, class_gt

    def __len__(self):
        return len(self.files)
