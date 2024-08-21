# Last modified: 2024-04-30
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import io
import os
import random
import tarfile
from enum import Enum
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from src.util.depth_transform import DepthNormalizerBase


class DatasetMode(Enum):
    RGB_ONLY = "rgb_only"
    EVAL = "evaluate"
    TRAIN = "train"
    RELIGHT = "relight"


class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4


def read_image_from_tar(tar_obj, img_rel_path):
    image = tar_obj.extractfile("./" + img_rel_path)
    image = image.read()
    image = Image.open(io.BytesIO(image))

def normalize_rgb(x):
    return x / 255.0 * 2 - 1  # [0, 255] -> [-1, 1]

class BaseDepthDataset(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        #filename_ls_path: str = None,
        dataset_dir: str ,
        #disp_name: str = 'something',
        #min_depth: float = 0,
        #max_depth: float = 1,
        # has_filled_depth: bool = False,
        # name_mode: DepthFileNameMode = 'something',
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        #rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        rgb_transform = normalize_rgb,
        experiment_type = 'front' #or 'random'
        # **kwargs,
    ):
        super().__init__()
        self.mode = mode
        # dataset info
        #self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        assert os.path.exists(
            self.dataset_dir
        ), f"Dataset does not exist at: {self.dataset_dir}"
        #self.disp_name = disp_name
        #self.has_filled_depth = has_filled_depth
        #self.name_mode: DepthFileNameMode = name_mode
        #self.min_depth = min_depth
        #self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane
        
        #relighting arguments
        self.experiment_type = experiment_type
        #self.data = self._get_relight_data_path()
        self.data = self._get_objaverse_data_path()
        # Load filenames
        # with open(self.filename_ls_path, "r") as f:
        #     self.filenames = [
        #         s.split() for s in f.readlines()
        #     ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )
        # self.objects = []
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        # if DatasetMode.TRAIN==self.mode or DatasetMode.RELIGHT==self.mode:
        #     rasters = self._training_preprocess(rasters)
            
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):

        #data = self._get_relight_data_path()
        rand_number = random.randint(1,4)
        img1_rel_path = self.data[index]['images'][0]
        img2_rel_path = self.data[index]['images'][rand_number]

        rasters = {}

        #if DatasetMode.RELIGHT==self.mode or :
        rasters.update(self._load_rgb_data(img1_rel_path=img1_rel_path,img2_rel_path=img2_rel_path))
        other = {"index": index, "rgb_relative_path": img2_rel_path}
        #tempp
        

        return rasters, other
    
    def _get_data_item1(self, index):

        #data = self._get_relight_data_path()
        img1_rel_path = self.data[index]['source_image']
        img2_rel_path = self.data[index]['target_image']

        rasters = {}

        #if DatasetMode.RELIGHT==self.mode or :
        rasters.update(self._load_rgb_data(img1_rel_path=img1_rel_path,img2_rel_path=img2_rel_path))
        other = {"index": index, "rgb_relative_path": img1_rel_path}
        # else:

        #     # RGB data
        #     rasters.update(self._load_rgb_data(rgb_rel_path=img1_rel_path))

        #     # Depth data
        #     if DatasetMode.RGB_ONLY != self.mode:
        #         # load data
        #         depth_data = self._load_depth_data(
        #             depth_rel_path=img2_rel_path, filled_rel_path=filled_rel_path
        #         )
        #         rasters.update(depth_data)
        #         # valid mask
        #         rasters["valid_mask_raw"] = self._get_valid_mask(
        #             rasters["depth_raw_linear"]
        #         ).clone()
        #         rasters["valid_mask_filled"] = self._get_valid_mask(
        #             rasters["depth_filled_linear"]
        #         ).clone()

        #     other = {"index": index, "rgb_relative_path": img1_rel_path}

        return rasters, other

    def _load_rgb_data(self, img1_rel_path,img2_rel_path=None):
        # Read RGB data

        #if DatasetMode.RELIGHT == self.mode or DatasetMode.TRAIN == self.mode:
            img1 = self._read_rgb_file(img1_rel_path)
            img1_norm = img1 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            img2 = self._read_rgb_file(img2_rel_path)
            img2_norm = img2 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

            img = np.hstack((np.transpose(img1,(1,2,0)),np.transpose(img2,(1,2,0))))
            Image.fromarray(img.astype(np.uint8)).save('./input_confirmation.jpg')

            outputs = {
                    "condition_img_int": torch.from_numpy(img1).int(),
                    "condition_img_norm": torch.from_numpy(img1_norm).float(),
                    "relit_img_int": torch.from_numpy(img2).int(),
                    "relit_img_norm": torch.from_numpy(img2_norm).float(),
                }
            return outputs
    
        # else:
        #     rgb = self._read_rgb_file(img1_rel_path)
        #     rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

        #     outputs = {
        #         "rgb_int": torch.from_numpy(rgb).int(),
        #         "rgb_norm": torch.from_numpy(rgb_norm).float(),
        #     }
        #     return outputs
    
    def _get_relight_data_path(self):

        data = []

        if DatasetMode.TRAIN == self.mode:
            root_dir = f'{self.dataset_dir}/../../sphere/test'
        if DatasetMode.EVAL == self.mode:
            root_dir = f'{self.dataset_dir}/../../sphere/val'
        if DatasetMode.RELIGHT == self.mode:
            root_dir = f'{self.dataset_dir}/../../sphere/test'
        for obj_dir in os.listdir(root_dir):
            obj_path = os.path.join(root_dir,obj_dir)
            if not os.path.isdir(obj_path):
                continue

            front_light_image = os.path.join(obj_path,'render_Front_Light.png')
            random_images = [f for f in os.listdir(obj_path) if f.startswith('render_Random') and f.endswith('.png')]
            light_directions = [f for f in os.listdir(obj_path) if f.startswith('light_direction_Random') and f.endswith('.npy')]    

            if self.experiment_type == 'front':
                for target_image in random_images:
                    target_light = f"light_direction_Random_{target_image.split('_', 2)[-1].split('.')[0]}.npy"
                    if target_light in light_directions:
                        data.append({
                            'source_image': front_light_image,
                            'target_image': os.path.join(obj_path, target_image),
                            'target_light': os.path.join(obj_path, target_light)
                        })

            elif self.experiment_type == 'random':
                for source_image in random_images:
                    for target_image in random_images:
                        if source_image != target_image:
                            target_light = f"light_direction_Random_{target_image.split('_', 2)[-1].split('.')[0]}.npy"
                            if target_light in light_directions:
                                data.append({
                                    'source_image': os.path.join(obj_path, source_image),
                                    'target_image': os.path.join(obj_path, target_image),
                                    'target_light': os.path.join(obj_path, target_light)
                                })            
        
        return data
    
    def _get_objaverse_data_path(self):
        data = []
        
        if DatasetMode.TRAIN == self.mode:
          root_dir = f'{self.dataset_dir}'
        if DatasetMode.EVAL == self.mode:
            root_dir = f'{self.dataset_dir}'
        if DatasetMode.RELIGHT == self.mode:
            root_dir = f'{self.dataset_dir}'

        self.objects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))]

        for obj in self.objects:
            obj_path = os.path.join(root_dir,obj)
            view_dirs = [d for d in os.listdir(obj_path) if d.startswith('view_')]
            env_dir = os.path.join(obj_path,'environments')
            env_maps = [f for f in os.listdir(env_dir) if f.endswith('.exr')]

            for view_dir in view_dirs:
                view_path = os.path.join(obj_path,view_dir)
                images_dir = os.path.join(view_path,'image')
                camera_file = os.path.join(view_path,f'camera_{view_dir.split("_")[1]}.npy')
                images = [f for f in os.listdir(images_dir) if f.endswith('.png')]

                data.append({
                    'images':[os.path.join(images_dir,image) for image in images],
                    'env_maps':[os.path.join(env_dir,em) for em in env_maps]
                })

        return data

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            #image_to_read = os.path.join(self.dataset_dir, img_rel_path)
            image_to_read = img_rel_path
        image = Image.open(image_to_read).convert('RGB')  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["depth_filled_norm"] = self.depth_transform(
            rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None

class BaseDepthDataset1(Dataset):
    def __init__(
        self,
        mode: DatasetMode,
        filename_ls_path: str,
        dataset_dir: str,
        disp_name: str,
        min_depth: float,
        max_depth: float,
        has_filled_depth: bool,
        name_mode: DepthFileNameMode,
        depth_transform: Union[DepthNormalizerBase, None] = None,
        augmentation_args: dict = None,
        resize_to_hw=None,
        move_invalid_to_far_plane: bool = True,
        #rgb_transform=lambda x: x / 255.0 * 2 - 1,  #  [0, 255] -> [-1, 1],
        rgb_transform = normalize_rgb,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mode = mode
        # dataset info
        self.filename_ls_path = filename_ls_path
        self.dataset_dir = dataset_dir
        # assert os.path.exists(
        #     self.dataset_dir
        # ), f"Dataset does not exist at: {self.dataset_dir}"
        self.disp_name = disp_name
        self.has_filled_depth = has_filled_depth
        self.name_mode: DepthFileNameMode = name_mode
        self.min_depth = min_depth
        self.max_depth = max_depth

        # training arguments
        self.depth_transform: DepthNormalizerBase = depth_transform
        self.augm_args = augmentation_args
        self.resize_to_hw = resize_to_hw
        self.rgb_transform = rgb_transform
        self.move_invalid_to_far_plane = move_invalid_to_far_plane

        # Load filenames
        with open(self.filename_ls_path, "r") as f:
            self.filenames = [
                s.split() for s in f.readlines()
            ]  # [['rgb.png', 'depth.tif'], [], ...]

        # Tar dataset
        self.tar_obj = None
        self.is_tar = (
            True
            if os.path.isfile(dataset_dir) and tarfile.is_tarfile(dataset_dir)
            else False
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rasters, other = self._get_data_item(index)
        if DatasetMode.TRAIN == self.mode:
            rasters = self._training_preprocess(rasters)
            
        # merge
        outputs = rasters
        outputs.update(other)
        return outputs

    def _get_data_item(self, index):
        img1_rel_path, img2_rel_path, filled_rel_path = self._get_data_path(index=index)

        rasters = {}

        if DatasetMode.RELIGHT == self.mode:
            rasters.update(self._load_rgb_data(img1_rel_path=img1_rel_path,img2_rel_path=img2_rel_path))
            other = {"index": index, "rgb_relative_path": img1_rel_path}
        else:

            # RGB data
            rasters.update(self._load_rgb_data(img1_rel_path=img1_rel_path,img2_rel_path=None))

            # Depth data
            if DatasetMode.RGB_ONLY != self.mode:
                # load data
                depth_data = self._load_depth_data(
                    depth_rel_path=img2_rel_path, filled_rel_path=filled_rel_path
                )
                rasters.update(depth_data)
                # valid mask
                rasters["valid_mask_raw"] = self._get_valid_mask(
                    rasters["depth_raw_linear"]
                ).clone()
                rasters["valid_mask_filled"] = self._get_valid_mask(
                    rasters["depth_filled_linear"]
                ).clone()

            other = {"index": index, "rgb_relative_path": img1_rel_path}

        return rasters, other

    def _load_rgb_data(self, img1_rel_path,img2_rel_path):
        # Read RGB data
        if self.mode == DatasetMode.RELIGHT:
            
            if img2_rel_path is not None:
                img1 = self._read_rgb_file(img1_rel_path)
                img1_norm = img1 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
                img2 = self._read_rgb_file(img2_rel_path)
                img2_norm = img2 / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]


                outputs = {
                    "condition_img_int": torch.from_numpy(img1).int(),
                    "condition_img_norm": torch.from_numpy(img1_norm).float(),
                    "relit_img_int": torch.from_numpy(img2).int(),
                    "relit_img_norm": torch.from_numpy(img2_norm).float(),
                }
    
        else:
            rgb = self._read_rgb_file(img1_rel_path)
            rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

            outputs = {
                "rgb_int": torch.from_numpy(rgb).int(),
                "rgb_norm": torch.from_numpy(rgb_norm).float(),
            }
        return outputs

    def _load_depth_data(self, depth_rel_path, filled_rel_path):
        # Read depth data
        outputs = {}
        depth_raw = self._read_depth_file(depth_rel_path).squeeze()
        depth_raw_linear = torch.from_numpy(depth_raw).float().unsqueeze(0)  # [1, H, W]
        outputs["depth_raw_linear"] = depth_raw_linear.clone()

        if self.has_filled_depth:
            depth_filled = self._read_depth_file(filled_rel_path).squeeze()
            depth_filled_linear = torch.from_numpy(depth_filled).float().unsqueeze(0)
            outputs["depth_filled_linear"] = depth_filled_linear
        else:
            outputs["depth_filled_linear"] = depth_raw_linear.clone()

        return outputs

    def _get_data_path(self, index):
        filename_line = self.filenames[index]

        # Get data path
        rgb_rel_path = filename_line[0]

        depth_rel_path, filled_rel_path = None, None
        if DatasetMode.RGB_ONLY != self.mode:
            depth_rel_path = filename_line[1]
            if self.has_filled_depth:
                filled_rel_path = filename_line[2]
        return rgb_rel_path, depth_rel_path, filled_rel_path

    def _read_image(self, img_rel_path) -> np.ndarray:
        if self.is_tar:
            if self.tar_obj is None:
                self.tar_obj = tarfile.open(self.dataset_dir)
            image_to_read = self.tar_obj.extractfile("./" + img_rel_path)
            image_to_read = image_to_read.read()
            image_to_read = io.BytesIO(image_to_read)
        else:
            image_to_read = os.path.join(self.dataset_dir, img_rel_path)
        image = Image.open(image_to_read)  # [H, W, rgb]
        image = np.asarray(image)
        return image

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        rgb = self._read_image(rel_path)
        rgb = np.transpose(rgb, (2, 0, 1)).astype(int)  # [rgb, H, W]
        return rgb

    def _read_depth_file(self, rel_path):
        depth_in = self._read_image(rel_path)
        #  Replace code below to decode depth according to dataset definition
        depth_decoded = depth_in

        return depth_decoded

    def _get_valid_mask(self, depth: torch.Tensor):
        valid_mask = torch.logical_and(
            (depth > self.min_depth), (depth < self.max_depth)
        ).bool()
        return valid_mask

    def _training_preprocess(self, rasters):
        # Augmentation
        if self.augm_args is not None:
            rasters = self._augment_data(rasters)

        # Normalization
        rasters["depth_raw_norm"] = self.depth_transform(
            rasters["depth_raw_linear"], rasters["valid_mask_raw"]
        ).clone()
        rasters["depth_filled_norm"] = self.depth_transform(
            rasters["depth_filled_linear"], rasters["valid_mask_filled"]
        ).clone()

        # Set invalid pixel to far plane
        if self.move_invalid_to_far_plane:
            if self.depth_transform.far_plane_at_max:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_max
                )
            else:
                rasters["depth_filled_norm"][~rasters["valid_mask_filled"]] = (
                    self.depth_transform.norm_min
                )

        # Resize
        if self.resize_to_hw is not None:
            resize_transform = Resize(
                size=self.resize_to_hw, interpolation=InterpolationMode.NEAREST_EXACT
            )
            rasters = {k: resize_transform(v) for k, v in rasters.items()}

        return rasters

    def _augment_data(self, rasters_dict):
        # lr flipping
        lr_flip_p = self.augm_args.lr_flip_p
        if random.random() < lr_flip_p:
            rasters_dict = {k: v.flip(-1) for k, v in rasters_dict.items()}

        return rasters_dict

    def __del__(self):
        if hasattr(self, "tar_obj") and self.tar_obj is not None:
            self.tar_obj.close()
            self.tar_obj = None


def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename
