import ast
import glob
import json
import logging
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import (
    Dataset,
    IterableDataset,
    DataLoader,
    DistributedSampler,
    ConcatDataset,
)

from saicinpainting.evaluation.data import (
    InpaintingDataset as InpaintingEvaluationDataset,
    OurInpaintingDataset as OurInpaintingEvaluationDataset,
    ceil_modulo,
    InpaintingEvalOnlineDataset,
    pad_img_to_modulo,
    scale_image,
)
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


def _resolve_user_data_path(path):
    """
    Turn config paths like ./scratch/... into absolute paths.

    Hydra changes cwd to outputs/<run>/, so plain abspath() would resolve under the run dir.
    Prefer the directory where the user started the job (Hydra original cwd).
    """
    if path is None:
        return None
    p = os.path.expanduser(str(path))
    if os.path.isabs(p):
        return os.path.normpath(p)
    try:
        from hydra.utils import get_original_cwd

        base = get_original_cwd()
    except Exception:
        base = os.getcwd()
    return os.path.normpath(os.path.join(base, p))


_NPY_MAGIC = b"\x93NUMPY"


def _load_npy(path):
    """
    Load an array from a path usually named *.npy.

    - Real binary NumPy arrays (magic bytes \\x93NUMPY) use ``np.load``.
    - Some pipelines save **text** nested lists (JSON or Python literals) but use a ``.npy``
      name; ``np.load`` then fails with pickle errors (e.g. invalid load key '[').
      Those are parsed with ``json`` or ``ast.literal_eval`` and converted with ``asarray``.
    """
    with open(path, "rb") as f:
        header = f.read(6)
    if len(header) >= 6 and header.startswith(_NPY_MAGIC):
        return np.load(path, allow_pickle=True)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Empty file: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError) as e:
            raise ValueError(
                f"Could not parse {path!r} as JSON or Python literal (nested lists). "
                f"Use np.save for binary .npy, or valid JSON / Python list text. Original error: {e}"
            ) from e
    return np.asarray(data, dtype=np.float32)


def _multichannel_hwc_to_uint8(img: np.ndarray, max_pixel_value=None) -> np.ndarray:
    """
    Albumentations CLAHE / HueSaturation expect uint8 RGB-style pipelines; multi-channel
    training uses a separate transform list. When the source range is known, keep float32
    precision in [0, 1]; otherwise fall back to the historical uint8 conversion.
    """
    if max_pixel_value is not None:
        if max_pixel_value <= 0:
            raise ValueError(f"max_pixel_value must be positive, got {max_pixel_value}")
        return np.clip(
            img.astype(np.float32) / float(max_pixel_value), 0.0, 1.0
        ).astype(np.float32)

    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return np.clip(img.astype(np.float32) * (255.0 / 65535.0), 0, 255).astype(
            np.uint8
        )
    x = img.astype(np.float32)
    out = np.empty(x.shape, dtype=np.uint8)
    for c in range(x.shape[2]):
        ch = x[:, :, c]
        hi = float(ch.max())
        lo = float(ch.min())
        if hi <= 1.0 + 1e-5:
            out[:, :, c] = np.clip(np.round(ch * 255.0), 0, 255).astype(np.uint8)
        elif hi > lo:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255).astype(
                np.uint8
            )
        else:
            out[:, :, c] = 0
    return out


class InpaintingTrainDataset(Dataset):
    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(
            glob.glob(os.path.join(indir, "**", "*.jpg"), recursive=True)
        )
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img, mask=mask)


class InpaintingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = (
            webdataset.Dataset(indir)
            .shuffle(shuffle_buffer)
            .decode("rgb")
            .to_tuple("jpg")
        )
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype("uint8")
            img = self.transform(image=img)["image"]
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img, mask=mask)


class ImgSegmentationDataset(Dataset):
    def __init__(
        self,
        indir,
        mask_generator,
        transform,
        out_size,
        segm_indir,
        semantic_seg_n_classes,
    ):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(
            glob.glob(os.path.join(indir, "**", "*.jpg"), recursive=True)
        )

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)["image"]
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes = self.load_semantic_segm(path)
        result = dict(image=img, mask=mask, segm=segm, segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir, self.segm_indir).replace(
            ".jpg", ".png"
        )
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int) - 1, 0, None))
        ohe = F.one_hot(
            tensor.long(), num_classes=self.semantic_seg_n_classes
        )  # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


# Add this to saicinpainting/training/data/datasets.py


class MultiChannelInpaintingTrainDataset(Dataset):
    """Dataset for multi-channel satellite/hyperspectral images"""

    def __init__(
        self, indir, mask_generator, transform, n_channels=3, max_pixel_value=None
    ):
        self.in_files = (
            list(glob.glob(os.path.join(indir, "**", "*.tif"), recursive=True))
            + list(glob.glob(os.path.join(indir, "**", "*.png"), recursive=True))
            + list(glob.glob(os.path.join(indir, "**", "*.npy"), recursive=True))
        )
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0
        self.n_channels = n_channels
        self.max_pixel_value = max_pixel_value

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        # MAY NEED TO FIX
        path = self.in_files[item]

        # Read multi-channel image (supports .tif, .png, .npy)
        if path.endswith(".npy"):
            img = _load_npy(path)  # Shape: (H, W, C)
        else:
            # Use rasterio or PIL for multi-channel GeoTIFFs
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Ensure correct shape (H, W, C)
        if img.ndim == 2:
            img = np.stack([img] * self.n_channels, axis=-1)
        elif img.shape[2] != self.n_channels:
            # Handle channel mismatch by selecting first n_channels or padding
            if img.shape[2] > self.n_channels:
                img = img[:, :, : self.n_channels]
            else:
                pad = self.n_channels - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, pad)), mode="constant")

        # Albumentations (CLAHE, HSV, etc.) expect uint8 for typical RGB pipelines; use
        # transform_variant multichannel_light / satellite_multichannel — then ToFloat().
        img = _multichannel_hwc_to_uint8(img, max_pixel_value=self.max_pixel_value)

        # Apply augmentations
        img = self.transform(image=img)["image"]
        img = np.transpose(img, (2, 0, 1))

        # Generate mask
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1

        return dict(image=img, mask=mask)


class MultiChannelInpaintingEvalDataset(Dataset):
    """Validation dataset for multi-channel inpainting with precomputed masks."""

    def __init__(
        self,
        datadir,
        img_suffix=".npy",
        n_channels=3,
        max_pixel_value=None,
        pad_out_to_modulo=None,
        scale_factor=None,
        mask_subdir=None,
        **kwargs,
    ):
        self.datadir = _resolve_user_data_path(datadir)
        self.img_suffix = (
            img_suffix if str(img_suffix).startswith(".") else f".{img_suffix}"
        )
        self.n_channels = n_channels
        self.max_pixel_value = max_pixel_value
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor
        self.mask_subdir = mask_subdir

        # Strategy 1 (LaMa default): masks named *mask*.png; image = path before "_mask" + img_suffix
        self.mask_filenames = sorted(
            glob.glob(os.path.join(self.datadir, "**", "*mask*.png"), recursive=True)
        )
        self.img_filenames = [
            fname.rsplit("_mask", 1)[0] + self.img_suffix
            for fname in self.mask_filenames
        ]
        self._drop_missing_images()

        # Strategy 2: enumerate *img_suffix and pair with stem_mask.{png,tif} in same folder
        # (needed when mask filenames do not contain the substring "mask", or strategy 1 mis-pairs)
        if len(self.mask_filenames) == 0:
            self._pair_from_image_glob()

        if len(self.mask_filenames) == 0:
            n_mask_png = len(
                glob.glob(
                    os.path.join(self.datadir, "**", "*mask*.png"), recursive=True
                )
            )
            n_npy = len(
                glob.glob(
                    os.path.join(self.datadir, "**", f"*{self.img_suffix}"),
                    recursive=True,
                )
            )
            raise ValueError(
                f"No validation pairs found under {datadir!r} "
                f"(resolved={self.datadir!r}, isdir={os.path.isdir(self.datadir)}, "
                f"glob *mask*.png={n_mask_png}, glob *{self.img_suffix}={n_npy}). "
                f"Expected either (1) masks matching **/*mask*.png with images "
                f"{{path_before_mask}}{self.img_suffix}, or (2) for each *{self.img_suffix} file, "
                f"a sibling <stem>_mask.png (or .tif) next to the image, or under mask_subdir. "
                f"Example: patch_01{self.img_suffix} + patch_01_mask.png in the same folder. "
                f"If glob counts are 0, relative paths may have been resolved from Hydra's run "
                f"directory; this codebase now resolves against Hydra's original cwd — sync this "
                f"file — or set an absolute data.val.indir."
            )

    def _drop_missing_images(self):
        """Keep only pairs whose image file exists (avoids silent empty sets)."""
        imgs, masks = [], []
        for img_path, mask_path in zip(self.img_filenames, self.mask_filenames):
            if os.path.isfile(img_path):
                imgs.append(img_path)
                masks.append(mask_path)
        self.img_filenames = imgs
        self.mask_filenames = masks

    def _pair_from_image_glob(self):
        pattern = os.path.join(self.datadir, "**", f"*{self.img_suffix}")
        img_paths = sorted(glob.glob(pattern, recursive=True))
        img_out, mask_out = [], []
        for img_path in img_paths:
            stem, _ = os.path.splitext(os.path.basename(img_path))
            if stem.lower().endswith("mask"):
                continue
            d = os.path.dirname(img_path)
            found = None
            for name in (
                f"{stem}_mask.png",
                f"{stem}_mask.PNG",
                f"{stem}_mask.tif",
                f"{stem}_mask.TIF",
                f"{stem}_mask.tiff",
            ):
                cand = os.path.join(d, name)
                if os.path.isfile(cand):
                    found = cand
                    break
            if found is None and self.mask_subdir:
                sub = os.path.join(self.datadir, self.mask_subdir)
                for ext in (".png", ".PNG", ".tif", ".TIF", ".tiff", ".npy"):
                    cand = os.path.join(sub, f"{stem}{ext}")
                    if os.path.isfile(cand):
                        found = cand
                        break
            if found is not None:
                img_out.append(img_path)
                mask_out.append(found)
        self.img_filenames = img_out
        self.mask_filenames = mask_out

    def __len__(self):
        return len(self.mask_filenames)

    def _load_multichannel_image(self, path):
        if path.endswith(".npy"):
            img = _load_npy(path)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img.ndim == 2:
            img = np.stack([img] * self.n_channels, axis=-1)
        elif img.shape[2] != self.n_channels:
            if img.shape[2] > self.n_channels:
                img = img[:, :, : self.n_channels]
            else:
                pad = self.n_channels - img.shape[2]
                img = np.pad(img, ((0, 0), (0, 0), (0, pad)), mode="constant")

        if self.max_pixel_value is not None:
            if self.max_pixel_value <= 0:
                raise ValueError(
                    f"max_pixel_value must be positive, got {self.max_pixel_value}"
                )
            img = np.clip(
                img.astype(np.float32) / float(self.max_pixel_value), 0.0, 1.0
            )
        elif img.dtype != np.float32:
            img = (
                img.astype(np.float32) / 255.0
                if np.max(img) > 1
                else img.astype(np.float32)
            )
        return np.transpose(img, (2, 0, 1))

    def _load_mask_gray(self, path):
        if path.endswith(".npy"):
            m = _load_npy(path)
            if m.ndim == 3:
                m = m[..., 0]
            m = m.astype(np.float32)
            if np.max(m) > 1:
                m /= 255.0
            return m
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        return m.astype(np.float32)

    def __getitem__(self, i):
        image = self._load_multichannel_image(self.img_filenames[i])
        mask = self._load_mask_gray(self.mask_filenames[i])
        if np.max(mask) > 1:
            mask /= 255.0
        result = dict(image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result["image"] = scale_image(result["image"], self.scale_factor)
            result["mask"] = scale_image(
                result["mask"], self.scale_factor, interpolation=cv2.INTER_NEAREST
            )

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result["unpad_to_size"] = result["image"].shape[1:]
            result["image"] = pad_img_to_modulo(result["image"], self.pad_out_to_modulo)
            result["mask"] = pad_img_to_modulo(result["mask"], self.pad_out_to_modulo)

        return result


def get_transforms(transform_variant, out_size):
    if transform_variant == "default":
        transform = A.Compose(
            [
                A.RandomScale(scale_limit=0.2),  # +/- 20%
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
                ),
                A.ToFloat(),
            ]
        )
    elif transform_variant == "distortions":
        transform = A.Compose(
            [
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.7, 1.3), rotate=(-40, 40), shear=(-0.1, 0.1)),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
                ),
                A.ToFloat(),
            ]
        )
    elif transform_variant == "distortions_scale05_1":
        transform = A.Compose(
            [
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.5, 1.0), rotate=(-40, 40), shear=(-0.1, 0.1), p=1),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
                ),
                A.ToFloat(),
            ]
        )
    elif transform_variant == "distortions_scale03_12":
        transform = A.Compose(
            [
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(scale=(0.3, 1.2), rotate=(-40, 40), shear=(-0.1, 0.1), p=1),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
                ),
                A.ToFloat(),
            ]
        )
    elif transform_variant == "distortions_scale03_07":
        transform = A.Compose(
            [
                IAAPerspective2(scale=(0.0, 0.06)),
                IAAAffine2(
                    scale=(0.3, 0.7),  # scale 512 to 256 in average
                    rotate=(-40, 40),
                    shear=(-0.1, 0.1),
                    p=1,
                ),
                A.PadIfNeeded(min_height=out_size, min_width=out_size),
                A.OpticalDistortion(),
                A.RandomCrop(height=out_size, width=out_size),
                A.HorizontalFlip(),
                A.CLAHE(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(
                    hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
                ),
                A.ToFloat(),
            ]
        )
    elif transform_variant in ("multichannel_light", "satellite_multichannel"):
        # No CLAHE (uint8-only) or HueSaturation (RGB-only); safe for H×W×3 and similar.
        transform = A.Compose(
            [
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.ToFloat(),
            ]
        )
    # elif transform_variant in ("distortions_light", "light_distortions"):
    #     # satellite_256.yaml historically used the mis-spelled alias "light_distortions"
    #     transform = A.Compose(
    #         [
    #             IAAPerspective2(scale=(0.0, 0.02)),
    #             IAAAffine2(scale=(0.8, 1.8), rotate=(-20, 20), shear=(-0.03, 0.03)),
    #             A.PadIfNeeded(min_height=out_size, min_width=out_size),
    #             A.RandomCrop(height=out_size, width=out_size),
    #             A.HorizontalFlip(),
    #             A.CLAHE(),
    #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    #             A.HueSaturationValue(
    #                 hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
    #             ),
    #             A.ToFloat(),
    #         ]
    #     )
    # elif transform_variant == "non_space_transform":
    #     transform = A.Compose(
    #         [
    #             A.CLAHE(),
    #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    #             A.HueSaturationValue(
    #                 hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
    #             ),
    #             A.ToFloat(),
    #         ]
    #     )
    # elif transform_variant == "no_augs":
    #     transform = A.Compose([A.ToFloat()])
    else:
        raise ValueError(f"Unexpected transform_variant {transform_variant}")
    return transform


def make_default_train_dataloader(
    indir,
    kind="default",
    out_size=512,
    mask_gen_kwargs=None,
    transform_variant="default",
    mask_generator_kind="mixed",
    dataloader_kwargs=None,
    ddp_kwargs=None,
    **kwargs,
):
    indir = _resolve_user_data_path(indir)

    LOGGER.info(
        f"Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}"
    )

    mask_generator = get_mask_generator(
        kind=mask_generator_kind, kwargs=mask_gen_kwargs
    )
    transform = get_transforms(transform_variant, out_size)

    if kind == "default":
        dataset = InpaintingTrainDataset(
            indir=indir, mask_generator=mask_generator, transform=transform, **kwargs
        )
    elif kind == "default_web":
        dataset = InpaintingTrainWebDataset(
            indir=indir, mask_generator=mask_generator, transform=transform, **kwargs
        )
    elif kind == "img_with_segm":
        dataset = ImgSegmentationDataset(
            indir=indir,
            mask_generator=mask_generator,
            transform=transform,
            out_size=out_size,
            **kwargs,
        )
    # In make_default_train_dataloader function
    elif kind == "multichannel":
        dataset = MultiChannelInpaintingTrainDataset(
            indir=indir,
            mask_generator=mask_generator,
            transform=transform,
            n_channels=kwargs.get("n_channels", 3),
            max_pixel_value=kwargs.get("max_pixel_value"),
        )
    else:
        raise ValueError(f"Unknown train dataset kind {kind}")

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ("default_web",)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs["shuffle"] = False
        dataloader_kwargs["sampler"] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and "shuffle" in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs["shuffle"]

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(
    indir, kind="default", out_size=512, transform_variant="default", **kwargs
):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset(
            [
                make_default_val_dataset(
                    idir,
                    kind=kind,
                    out_size=out_size,
                    transform_variant=transform_variant,
                    **kwargs,
                )
                for idir in indir
            ]
        )

    indir = _resolve_user_data_path(indir)

    LOGGER.info(f"Make val dataloader {kind} from {indir}")
    mask_generator = get_mask_generator(
        kind=kwargs.get("mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs")
    )

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == "default":
        dataset_kwargs = dict(kwargs)
        dataset_kwargs.pop("n_channels", None)
        dataset_kwargs.pop("max_pixel_value", None)
        dataset_kwargs.pop("mask_subdir", None)
        dataset = InpaintingEvaluationDataset(indir, **dataset_kwargs)
    elif kind == "our_eval":
        dataset_kwargs = dict(kwargs)
        dataset_kwargs.pop("n_channels", None)
        dataset_kwargs.pop("max_pixel_value", None)
        dataset_kwargs.pop("mask_subdir", None)
        dataset = OurInpaintingEvaluationDataset(indir, **dataset_kwargs)
    elif kind == "img_with_segm":
        dataset = ImgSegmentationDataset(
            indir=indir,
            mask_generator=mask_generator,
            transform=transform,
            out_size=out_size,
            **kwargs,
        )
    elif kind == "online":
        dataset = InpaintingEvalOnlineDataset(
            indir=indir,
            mask_generator=mask_generator,
            transform=transform,
            out_size=out_size,
            **kwargs,
        )
    elif kind == "multichannel":
        dataset = MultiChannelInpaintingEvalDataset(
            datadir=indir,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown val dataset kind {kind}")

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(
    img_height, img_width, min_size=128, max_size=512, area=256 * 256, round_to_mod=16
):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(
            max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod)
        )
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(
            max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod)
        )
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
