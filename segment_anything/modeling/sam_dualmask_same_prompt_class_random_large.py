# each class has its corresponding point embedding
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder1 import ImageEncoderViT1
from .image_encoder2 import ImageEncoderViT2
from .mask_decoder import MaskDecoder
from .prompt_encoder_prompt_class import PromptEncoder_prompt_class
import numpy as np
from skimage.measure import label
import cv2
from scipy import ndimage
from copy import deepcopy


class BatchedInputNoiser:
    def __init__(self):
        self.gaussian_var = 0.3
        self.salt_pepper_density = 0.1
        self.rotate_range = (-20, 20)
        self.translate_pixels = 10
        self.blur_kernel = 5

def add_noise_to_batched_input(self, batched_input):

    noisy_batched_input = []
    for sample in batched_input:
        noisy_sample = deepcopy(sample)
        image = noisy_sample["image"].copy()  #

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        gaussian_noise = np.random.randn(*image.shape) * self.gaussian_var
        image = image + gaussian_noise
        image = np.clip(image, image.min(), image.max())  # 保持值范围

        mask = np.random.rand(*image.shape) < self.salt_pepper_density
        salt_mask = mask & (np.random.rand(*image.shape) < 0.5)
        pepper_mask = mask & ~salt_mask
        image[salt_mask] = image.max()
        image[pepper_mask] = image.min()

        angle = np.random.uniform(*self.rotate_range)
        image = ndimage.rotate(image, angle, order=3, reshape=False)  # 保持尺寸不变

        tx = np.random.randint(-self.translate_pixels, self.translate_pixels + 1)
        ty = np.random.randint(-self.translate_pixels, self.translate_pixels + 1)
        image = ndimage.shift(image, (ty, tx), order=3, mode="constant")
        image = ndimage.gaussian_filter(image, sigma=self.blur_kernel / 6)
        if isinstance(noisy_sample["image"], torch.Tensor):
            image = torch.from_numpy(image).to(noisy_sample["image"].device)

        noisy_sample["image"] = image
        noisy_batched_input.append(noisy_sample)

    return noisy_batched_input
def MaskToBoxSimple(mask):
    mask = mask.squeeze()
    #find coordinates of points in the region
    row, col = np.argwhere(mask).T
    # find the four corner coordinates
    y0,x0 = row.min(),col.min()
    y1,x1 = row.max(),col.max()

    return [x0,y0,x1,y1]

class Sam_dualmask_same_prompt_class_random_large(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder1: ImageEncoderViT1,
        image_encoder2: ImageEncoderViT2,
        prompt_encoder: PromptEncoder_prompt_class,
        mask_decoder1: MaskDecoder,
        mask_decoder2: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:

        super().__init__()
        self.image_encoder1 = image_encoder1
        self.image_encoder2 = image_encoder2
        self.prompt_encoder = prompt_encoder
        self.mask_decoder1 = mask_decoder1
        self.mask_decoder2 = mask_decoder2
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def forward_train(self, batched_input, multimask_output, image_size, prompt_idx, prompt):
        noiser = BatchedInputNoiser()
        noisy_batched_input = noiser.add_noise_to_batched_input(batched_input)
        input_images = self.preprocess(batched_input)
        Noise_images = self.preprocess(noisy_batched_input)
        image_embeddings1 = self.image_encoder1(input_images)
        image_embeddings2 = self.image_encoder2(Noise_images)

        if prompt_idx == 0:
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings1,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            # generate prompts based on the coarse prediction
            points_prompt, points_prompt_random, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks1, image_size, True)

            if prompt == 'point':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=None
                )
                sparse_embeddings_r, _ = self.prompt_encoder(
                    points=points_prompt_random, boxes=None, masks=None
                )
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )


            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings2,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks2_r, iou_predictions2_r, _ = self.mask_decoder2(
                image_embeddings=image_embeddings2,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_r,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )
        
        elif prompt_idx == 1:  
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                sparse_embeddings = sparse_embeddings.detach()
                dense_embeddings = dense_embeddings.detach()
            
        
            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings2,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output 
            )

            # generate prompts based on the coarse prediction
            points_prompt, points_prompt_random, box_prompt, mask_prompt = self.prompt_generate_random_fast(low_res_masks2, image_size, True) 

            if prompt == 'point':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=None
                )
                sparse_embeddings_r, dense_embeddings_r = self.prompt_encoder(
                    points=points_prompt_random, boxes=None, masks=None
                )
            elif prompt == 'box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=None
                )
            elif prompt == 'mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=mask_prompt
                )
            elif prompt == 'point-box':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=None
                )
            elif prompt == 'point-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=None, masks=mask_prompt
                )
            elif prompt == 'box-mask':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=box_prompt, masks=mask_prompt
                )
            elif prompt == 'all':
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points_prompt, boxes=box_prompt, masks=mask_prompt
                )
            else:
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings1,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks1_r, iou_predictions1_r, _ = self.mask_decoder1(
                image_embeddings=image_embeddings1,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_r,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )
        
        else:
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
            )

            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=image_embeddings1,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )

            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=image_embeddings2,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output  
            )


        masks1 = self.postprocess_masks(
            low_res_masks1,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )
        masks2 = self.postprocess_masks(
            low_res_masks2,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size)
        )

        if prompt_idx != -1:
            if prompt_idx == 1:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'low_res_logits1_r': low_res_masks1_r,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2
                }
            else:
                outputs = {
                    'masks': masks1,
                    'iou_predictions1': iou_predictions1,
                    'low_res_logits1': low_res_masks1,
                    'masks2': masks2,
                    'iou_predictions2': iou_predictions2,
                    'low_res_logits2': low_res_masks2,
                    'low_res_logits2_r': low_res_masks2_r
                }
        else:
            outputs = {
                'masks': masks1,
                'iou_predictions1': iou_predictions1,
                'low_res_logits1': low_res_masks1,
                'masks2': masks2,
                'iou_predictions2': iou_predictions2,
                'low_res_logits2': low_res_masks2
            }
        return outputs

    @torch.no_grad()
    def forward_test(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings1 = self.image_encoder1(input_images)
        image_embeddings2 = self.image_encoder2(input_images)
        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings1, image_embeddings2):
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            low_res_masks1, iou_predictions1, _ = self.mask_decoder1(
                image_embeddings=embed1.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            low_res_masks2, iou_predictions2, _ = self.mask_decoder2(
                image_embeddings=embed2.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            input_size = image_record["image"].shape[-2:]
            original_size = image_record["original_size"]

            masks1 = self.postprocess_masks(low_res_masks1, input_size, original_size)
            masks2 = self.postprocess_masks(low_res_masks2, input_size, original_size)

            avg_low_res_logits = (low_res_masks1 + low_res_masks2) / 2
            avg_iou_predictions = (iou_predictions1 + iou_predictions2) / 2
            avg_masks = (masks1 + masks2) / 2
            avg_masks_binary = avg_masks > self.mask_threshold

            outputs.append(
                {
                    "masks": avg_masks_binary,
                    "iou_predictions": avg_iou_predictions,
                    "low_res_logits": avg_low_res_logits,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def prompt_generate_random_fast(self, coarse_mask, img_size, israndom = False):  # generate point prompts
        b, num_class, h, w = coarse_mask.shape

        coarse_mask_np = torch.argmax(coarse_mask, dim = 1)
        coarse_mask_np = F.interpolate(coarse_mask_np.unsqueeze(1).float(), (img_size, img_size), mode="nearest").squeeze(1)
        coarse_mask_np = coarse_mask_np.detach().cpu().numpy()

        # points: BxNx2 tensor & boxes
        points_prompt = np.zeros([b, num_class, 2])
        points_label = np.zeros([b, num_class])
        points_prompt_random = np.zeros([b, num_class, 2])
        for idx in range(b):  # iterate over each image
            for cls in range(num_class): # find points for each class
                # obtain the binary mask
                mask_cls = (coarse_mask_np[idx] == cls).astype(np.uint8)
                if mask_cls.max() > 0:
                    label_msk, region_ids = label(mask_cls, connectivity=2, return_num=True)
                    ratio_list, regionid_list = [], []
                    for region_id in range(1, region_ids+1):
                        #find coordinates of points in the region
                        binary_msk = np.where(label_msk==region_id, 1, 0)

                        # clean some region that is abnormally small
                        r = np.sum(binary_msk) / np.sum(mask_cls)
                        # print('curr mask over all mask ratio', r)
                        ratio_list.append(r)
                        regionid_list.append(region_id)

                    ratio_list, regionid_list = zip(*sorted(zip(ratio_list, regionid_list)))
                    regionid_list = regionid_list[::-1]

                    binary_msk = np.where(label_msk==regionid_list[0], 1, 0)
                    
                    if israndom:  
                        cY_r, cX_r = np.where(binary_msk==1)
                        random_idx = np.random.randint(0, len(cX_r))
                        points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = int(cX_r[random_idx]), int(cY_r[random_idx])

                    # Calculates the distance to the closest zero pixel for each pixel of the source image.
                    # Ref from RITM: https://github.com/SamsungLabs/ritm_interactive_segmentation/blob/aa3bb52a77129e477599b5edfd041535bc67b259/isegm/data/points_sampler.py
                    # NOTE: numpy and opencv have inverse definition of row and column
                    # NOTE: SAM and opencv have the same definition
                    padded_mask = np.uint8(np.pad(binary_msk, ((1, 1), (1, 1)), 'constant'))
                    dist_img = cv2.distanceTransform(padded_mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)[1:-1, 1:-1]
                    cY, cX = np.where(dist_img==dist_img.max())
                    random_idx = np.random.randint(0, len(cX))
                    points_prompt[idx,cls,0], points_prompt[idx,cls,1] = int(cX[random_idx]), int(cY[random_idx])
                    
                    if cls > 0:
                        points_label[idx,cls] = cls
                
                else:
                    points_prompt[idx,cls,0], points_prompt[idx,cls,1] = points_prompt[idx,0,0], points_prompt[idx,0,1]
                    points_prompt_random[idx,cls,0], points_prompt_random[idx,cls,1] = points_prompt[idx,0,0], points_prompt[idx,0,1]
                    points_label[idx,cls] = 0

        points_prompt = torch.tensor(points_prompt).to(coarse_mask.device)
        points_label = torch.tensor(points_label).to(coarse_mask.device)
        points_prompt = (points_prompt, points_label)

        if israndom:  
            points_prompt_random = torch.tensor(points_prompt_random).to(coarse_mask.device)
            points_prompt_random = (points_prompt_random, points_label)

            return points_prompt, points_prompt_random, None, None

        return points_prompt, None, None
