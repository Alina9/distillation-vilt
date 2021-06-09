import os
import numpy as np
import torch.utils.data as data
# import skimage.io as io
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms


class CocoCaptions(data.Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        caption_transform (callable, optional): A function/transform that takes in the
            caption and transforms it.
    """

    def __init__(self, root, annFile, start=0, end=None, transform=None, caption_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ann_ids = list(self.coco.anns.keys())
        self.ann_ids = self.filter_ids(self.coco, self.ann_ids)
        self.transform = transform
        self.caption_transform = caption_transform
        if end is None:
            self.ann_ids = self.ann_ids[start:]
        else:
            self.ann_ids = self.ann_ids[start:end]

    def filter_ids(self, coco, ids):
        image_id_to_ids = {}

        for i in ids:
            image_id = coco.anns[i]['image_id']

            if image_id not in image_id_to_ids:
                image_id_to_ids[image_id] = []

            image_id_to_ids[image_id].append(i)

        image_ids_gen = filter(lambda i: len(image_id_to_ids[i]) >= 5, image_id_to_ids.keys())

        ids = []
        for image_id in image_ids_gen:
            ids += image_id_to_ids[image_id][:5]

        return ids

    def get_caption(self, index):
        id = self.ann_ids[index]
        coco_item = self.coco.anns[id]

        caption = coco_item['caption']
        if self.caption_transform is not None:
            caption = self.caption_transform(caption)
        return caption

    def get_image(self, index):
        id = self.ann_ids[index]
        coco_item = self.coco.anns[id]

        img_id = coco_item['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, img_id

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, caption)
        """
        id = self.ann_ids[index]
        coco_item = self.coco.anns[id]

        caption = coco_item['caption']
        if self.caption_transform is not None:
            caption = self.caption_transform(caption)

        img_id = coco_item['image_id']
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, caption

    def __len__(self):
        l = len(self.ann_ids)
        return l


class CocoCaptionsOnline(data.Dataset):
    def __init__(self, annFile, transform=None, caption_transform=None):
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())
        self.ann_ids = list(self.coco.anns.keys())
        self.ann_ids = self.filter_ids(self.coco, self.ann_ids)
        self.transform = transform
        self.caption_transform = caption_transform

    def filter_ids(self, coco, ids):
        image_id_to_ids = {}

        for i in ids:
            image_id = coco.anns[i]['image_id']

            if image_id not in image_id_to_ids:
                image_id_to_ids[image_id] = []

            image_id_to_ids[image_id].append(i)

        image_ids_gen = filter(lambda i: len(image_id_to_ids[i]) >= 5, image_id_to_ids.keys())

        ids = []
        for image_id in image_ids_gen:
            ids += image_id_to_ids[image_id][:5]

        return ids

    def __getitem__(self, index):
        id = self.ann_ids[index]
        coco_item = self.coco.anns[id]

        caption = coco_item['caption']
        if self.caption_transform is not None:
            caption = self.caption_transform(caption)

        img_id = coco_item['image_id']
        img = self.coco.loadImgs(img_id)[0]
        img = io.imread(img['coco_url'])
        img = Image.fromarray(np.array(img)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, caption

    def __len__(self):
        return len(self.ann_ids)


class DatasetWrapper(CocoCaptions):
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.size * 5
