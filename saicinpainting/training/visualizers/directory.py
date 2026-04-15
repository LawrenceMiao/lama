import os

import cv2
import numpy as np

from saicinpainting.training.visualizers.base import BaseVisualizer, visualize_mask_and_images_batch
from saicinpainting.utils import check_and_warn_input_range


class DirectoryVisualizer(BaseVisualizer):
    DEFAULT_KEY_ORDER = 'image predicted_image inpainted'.split(' ')

    def __init__(self, outdir, key_order=DEFAULT_KEY_ORDER, max_items_in_batch=10,
                 last_without_mask=True, rescale_keys=None):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        self.key_order = key_order
        self.max_items_in_batch = max_items_in_batch
        self.last_without_mask = last_without_mask
        self.rescale_keys = rescale_keys

    def __call__(self, epoch_i, batch_i, batch, suffix='', rank=None):
        check_and_warn_input_range(batch['image'], 0, 1, 'DirectoryVisualizer target image')
        # key_order may list discr_* tensors; those exist only if training_model.store_discr_outputs_for_vis
        keys = [k for k in self.key_order if k in batch]
        rescale = (
            [k for k in (self.rescale_keys or []) if k in batch]
            if self.rescale_keys
            else None
        )
        vis_img = visualize_mask_and_images_batch(
            batch,
            keys,
            max_items=self.max_items_in_batch,
            last_without_mask=self.last_without_mask,
            rescale_keys=rescale,
        )

        vis_img = np.clip(vis_img * 255, 0, 255).astype('uint8')

        curoutdir = os.path.join(self.outdir, f'epoch{epoch_i:04d}{suffix}')
        os.makedirs(curoutdir, exist_ok=True)
        rank_suffix = f'_r{rank}' if rank is not None else ''
        out_fname = os.path.join(curoutdir, f'batch{batch_i:07d}{rank_suffix}.jpg')

        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_fname, vis_img)
