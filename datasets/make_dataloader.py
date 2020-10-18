import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import cv2
from .naic import NAIC
from .veri import VeRi
from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import RandomIdentitySampler
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.functional import image_compression, rot90
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise,\
    GaussianBlur, Resize, Normalize, RandomRotate90

__factory = {
    'veri': VeRi,
    'naic' :NAIC

}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids, img_paths

def create_train_transforms(size):
    return Compose([
        # ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        RandomRotate90(),
        Resize(height=size[0], width=size[1]),
        PadIfNeeded(min_height=size[0], min_width=size[1], border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        ToGray(p=0.1),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )


def create_val_transforms(size):
    return Compose([
        Resize(height=size[0], width=size[1]),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def make_dataloader(cfg):
    if cfg.DATASETS.ALBUMENTATION:
        print('______________________Using ALBUMENTATION!_________________________')
        train_transforms = create_train_transforms(size=cfg.INPUT.SIZE_TRAIN)
        val_transforms = create_val_transforms(size=cfg.INPUT.SIZE_TEST)
    else:
        if cfg.DATASETS.HARD_AUG:
            train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
                T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        else:
            train_transforms = T.Compose([
                T.Resize(cfg.INPUT.SIZE_TRAIN),
                T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                T.Pad(cfg.INPUT.PADDING),
                T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
            ])
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, add_2019=cfg.DATASETS.ADD_2019)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms, is_albumentations=cfg.DATASETS.ALBUMENTATION)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms,
                                 is_albumentations=cfg.DATASETS.ALBUMENTATION)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_set_normal = ImageDataset(dataset.query_normal + dataset.gallery_normal, val_transforms,
                                  is_albumentations=cfg.DATASETS.ALBUMENTATION)
    val_loader_normal = DataLoader(
        val_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader_green, val_loader_normal, len(dataset.query_green), len(dataset.query_normal), num_classes


def make_dataloader_all(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, add_2019=cfg.DATASETS.ADD_2019)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_query = ImageDataset(dataset.query_green + dataset.query_normal, val_transforms)
    val_loader_query = DataLoader(
        val_set_query, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_set_gallery = ImageDataset(dataset.gallery_green + dataset.gallery_normal, val_transforms)
    val_loader_gallery = DataLoader(
        val_set_gallery, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    num_query = len(dataset.query_green) + len(dataset.query_normal)

    return train_loader, val_loader_query, val_loader_gallery, num_query, num_classes


def make_dataloader_val(cfg):
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, add_2019=cfg.DATASETS.ADD_2019)
    num_classes = dataset.num_train_pids

    val_set_query = ImageDataset(dataset.query_val, val_transforms)
    val_loader_query = DataLoader(
        val_set_query, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    val_set_gallery = ImageDataset(dataset.gallery_val, val_transforms)
    val_loader_gallery = DataLoader(
        val_set_gallery, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    num_query = len(dataset.query_val)

    return val_loader_query, val_loader_gallery, num_query, num_classes


def make_dataloader_Pseudo(cfg):
    if cfg.DATASETS.HARD_AUG:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    num_classes = dataset.num_train_pids

    train_set = ImageDataset(dataset.train, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set_green = ImageDataset(dataset.query_green + dataset.gallery_green, val_transforms)
    val_loader_green = DataLoader(
        val_set_green, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader_green, len(dataset.query_green), num_classes, dataset, train_set, train_transforms
