import os
import csv
import random
import numpy as np
import copy
import json
from itertools import chain
import pandas as pd
from torch.utils.data import Dataset
import SimpleITK as sitk
from torch.utils.data.sampler import Sampler
import torch
import copy
import math


axes = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]], dtype=np.float32)


class FixedWindowNormalizer(object):
    """
    use fixed mean and stddev to normalize image intensities
    intensity = (intensity - mean) / stddev
    if clip is enabled:
        intensity = np.clip((intensity - mean) / stddev, -1, 1)
    """
    def __init__(self, window, level, clip=True):
        """ constructor """
        assert level > 0, 'level must be positive'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.window = window
        self.level = level
        self.clip = clip

    def normalize(self, image):
        intensity = image.to_numpy()
        min_value = self.window - self.level / 2.0
        max_value = self.window + self.level / 2.0
        #print(min_value, max_value)
        intensity[intensity > max_value] = max_value
        intensity[intensity < min_value] = min_value
        image.from_numpy(intensity)
        mean = (max_value + min_value)/2
        std = (max_value - min_value)/2
        ctools.intensity_normalize(image, mean, std, clip=self.clip)

    def __call__(self, image):
        """ normalize image """
        if isinstance(image, Image3d):
            self.normalize(image)
        elif isinstance(image, (list, tuple)):
            for im in image:
                assert isinstance(im, Image3d)
                self.normalize(im)
        else:
            raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

    def static_obj(self):
        """ get a static normalizer object by removing randomness """
        obj = FixedWindowNormalizer(self.mean, self.stddev, self.clip)
        return obj

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 0, 'window': self.window, 'level': self.level, 'clip': self.clip}
        return obj



def readlines(file):
    """
    read lines by removing '\n' in the end of line
    :param file: a text file
    :return: a list of line strings
    """
    fp = codecs.open(file, 'r', encoding='utf-8')
    linelist = fp.readlines()
    fp.close()
    for i in range(len(linelist)):
        linelist[i] = linelist[i].rstrip('\n')
    return linelist


def read_train_txt(imlist_file):
    """ read single-modality txt file
    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths
    """
    lines = readlines(imlist_file)
    num_cases = int(lines[0])

    if len(lines)-1 < num_cases * 2:
        raise ValueError('too few lines in imlist file')

    im_list, seg_list = [], []
    for i in range(num_cases):
        im_path, seg_path = lines[2 + i * 2], lines[1 + i * 2]
        if not os.path.isfile(im_path) or not os.path.isfile(seg_path):
            print('not exist!: ', im_path)
            continue
        assert os.path.isfile(im_path), 'image not exist: {}'.format(im_path)
        assert os.path.isfile(seg_path), 'mask not exist: {}'.format(seg_path)
        im_list.append([im_path])
        seg_list.append(seg_path)

    return im_list, seg_list


def resample_nn(input_image, new_size):
    """
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampled_image = resampler.Execute(input_image)

    return resampled_image


def resample_thick_image(images, spacing, default_values, interpolation='NN', thickness=None):
    if thickness is None:
        new_spacing = [spacing[0], spacing[1], max(images[0].spacing()[2], 5)]
    else:
        new_spacing = [spacing[0], spacing[1], thickness]
    target_size = np.ceil(images[0].size() / np.array(new_spacing) * images[0].spacing())
    frame.set_spacing(new_spacing)
    frame.set_axes(axes)
    for i in range(len(images)):
        images[i].set_axes(axes)
        images[i] =resample_nn(images[i], target_size, default_value=default_values[i])
    return images


def rect_region(asp, n=None, dim=3):
    if isinstance(n, int):
        ind = np.where(asp == n)
    else:
        ind = np.where(asp > 0)
    min_, max_ = [], []
    if len(ind[0]) == 0:
        return rect_region(asp, n=None, dim=dim)
    for i in range(dim):
        #pad =  i == 0 else 5
        pad = 0 if i == 0 else 5
        min_.append(max(np.min(ind[i])-pad, 0))
        max_.append(min(np.max(ind[i])+pad, asp.shape[i]-1))
    return np.array(min_), np.array(max_)


def cal_size(delta):
    if delta == 1:
        return [1, 0]
    if delta % 2 == 0:
        return [int(delta / 2), int(delta / 2)]
    return [int((delta - 1) / 2), int((delta - 1) / 2) + 1]


def random_crop(bounds, target_size, ori_shape, rand=None):
    if rand is None:
        rand = [5, 5]
    min_l, max_l, min_r, max_r = bounds
    center_l = np.around(min_l / 2 + max_l / 2)
    center_r = np.around(min_r / 2 + max_r / 2)
    trans = np.random.uniform(-1 * rand, rand, size=[2]).astype(np.int16)
    center_l[1:] += trans
    center_r[1:] -= trans
    min_l = np.clip(center_l - target_size // 2, a_min=0, a_max=999).astype(int)
    min_r = np.clip(center_r - target_size // 2, a_min=0, a_max=999).astype(int)
    max_l = np.minimum(ori_shape, center_l + target_size // 2).astype(int)
    max_r = np.minimum(ori_shape, center_r + target_size // 2).astype(int)
    return min_l, max_l, min_r, max_r, center_l, center_r


def cal_new_size(pl, pr):
    min_l, max_l = pl
    min_r, max_r = pr
    size_l = np.array(max_l) - np.array(min_l)
    size_r = np.array(max_r) - np.array(min_r)
    size_delta = size_l - size_r
    for i in range(3):
        pad_size = cal_size(abs(size_delta[i]))
        if size_delta[i] > 0:
            min_r[i] = min_r[i] - pad_size[0]
            max_r[i] = max_r[i] + pad_size[1]
        elif size_delta[i] < 0:
            min_l[i] = min_l[i] - pad_size[0]
            max_l[i] = max_l[i] + pad_size[1]
    return min_l, max_l, min_r, max_r


class ClassificationDataset(Dataset):
    def __init__(self, imlist_file,
                 tag_list_file,
                 mask,
                 num_classes,
                 crop_size,
                 spacing,
                 region_id,
                 default_values,
                 interpolation,
                 random_translation,
                 normalizers,
                 cbf_ratio_file=None,
                 is_train=True,
                 cbf_label=True,
                 cbf_thres=0.1,
                 subcor=False,
                 normalizer_aug=True
                 ):
        self.mask = mask
        if not self.mask:
            self.im_list = read_train_txt(imlist_file)
        else:
            self.im_list, self.mask_list = read_train_txt(imlist_file)
        self.tag_file = pd.read_csv(tag_list_file)
        self.num_classes = num_classes
        self.spacing = np.array(spacing, dtype=np.double)
        assert self.spacing.size == 2, 'only 2-element of spacing is supported'
        self.crop_size = np.array(crop_size)
        assert self.crop_size.size == 3, 'only 3-element of crop size is supported'
        self.region_id = region_id
        self.random_translation = np.array(random_translation)
        assert self.random_translation.size == 2, 'only 2-element of random translation is supported'
        self.crop_normalizers = normalizers
        self.normalizer_aug = (normalizer_aug and is_train)
        self.default_values = default_values
        self.interpolation = interpolation
        if cbf_ratio_file is not None:
            self.cbf_ratio = pd.read_csv(cbf_ratio_file)
        else:
            self.cbf_ratio = None
        self.is_train = is_train
        if self.is_train is True:
            self.random_translation = np.array([0, 0])
        self.cbf_label = cbf_label
        self.cbf_thres = min(cbf_thres, 0.2)
        if self.cbf_label is True and cbf_ratio_file is not None:
            self.tag_file = copy.deepcopy(self.cbf_ratio)
            for i in range(1, 21):
                self.tag_file[str(i)] = self.cbf_ratio[str(i)].apply(lambda x: 1 if x >= cbf_thres else 0)
        if subcor is True:
            self.all_regions = [1, 2, 3, 4]
        else:
            self.all_regions = [5, 6, 7, 8, 9, 10]
        if subcor is True:
            self.pos_prob = 0.85
        else:
            self.pos_prob = 0.82

    def __len__(self):
        return len(self.im_list)

    def generate_crop_bb(self, mask, region_id):
        pl = rect_region(mask, region_id)
        pr = rect_region(mask, region_id + 10)
        bounds = cal_new_size(pl, pr)
        min_l, max_l, min_r, max_r, center_l, center_r\
            = random_crop(bounds, self.crop_size, mask.shape, self.random_translation)
        return min_l, max_l, min_r, max_r, center_l, center_r

    def crop_array(self, image, min_ind, max_ind):
        mat = image.to_numpy()
        return mat[min_ind[0]:max_ind[0], min_ind[1]:max_ind[1], min_ind[2]:max_ind[2]]

    def crop_image3d(self, image, center, padvalue):
        size = self.crop_size[::-1]
        image_crop = center_crop(image, center, image.spacing(), size=size, padvalue=padvalue)
        return image_crop.to_numpy()

    def num_modality(self):
        """ get the number of input image modalities """
        return len(self.im_list[0])

    def find_positive_region(self, tag_row):
        pos = []
        neg = []
        for region_id in self.all_regions:
            if int(tag_row[str(region_id)].item()) == 1 or int(tag_row[str(region_id + 10)].item()) == 1:
                pos.append(region_id)
            else:
                neg.append(region_id)
        return pos, neg

    def __getitem__(self, index):
        image_paths, mask_path = self.im_list[index], self.mask_list[index]
        case = os.path.basename(os.path.dirname(image_paths[0]))
        tag_row = self.tag_file[self.tag_file['ID'] == case]
        case_name = case + '_' + os.path.basename(image_paths[0])
        images = []
        for image_path in image_paths:
            try:
                image = sitk.ReadImage(image_path, dtype=np.float32)
            except:
                print('!!!!!!!!!!!!:', image_path)
                raise ValueError('Fail to read image')
            images.append(image)

        mask = sitk.ReadImage(mask_path, dtype=np.int16)

        if self.region_id is None:
            pos, neg = self.find_positive_region(tag_row)
            if len(pos) == 0 or len(neg) == 0:
                region_id = random.choice(self.all_regions)
            elif random.random() <= self.pos_prob:
                region_id = random.choice(pos)
            else:
                region_id = random.choice(neg)
        else:
            region_id = self.region_id

        i = images[0].to_numpy()
        ww = np.ones_like(i)
        ww[(i >= -5) & (i < 0)] = 0
        ww[i >= 70] = 0
        ww = ww.astype(np.float32)
        mask_depress = mask.deep_copy()
        mask_depress.from_numpy(ww)

        for idx in range(len(images)):
            if self.crop_normalizers[idx] is not None:
                if not self.normalizer_aug:
                    self.crop_normalizers[idx](images[idx])
                else:
                    wl = 30 + 5 * random.random() - 2.5
                    ww = 60 + 5 * random.random() - 2.5
                    normalizer = FixedWindowNormalizer(wl, ww)
                    normalizer(images[idx])
        if random.random() <= 0.2 and self.is_train:
            thickness = random.choice([6, 7, 8, 9])
            print(thickness)
        else:
            thickness = None
        images = resample_thick_image(images, self.spacing, self.default_values, self.interpolation, thickness)
        mask = resample_thick_image([mask], self.spacing, [0], 'NN', thickness)[0]
        mask.set_frame(images[0].frame())
        mask_depress = resample_thick_image([mask_depress], self.spacing, [0], 'NN', thickness)[0]
        mask_depress.set_frame(images[0].frame())
        try:
            min_l, max_l, min_r, max_r, center_l, center_r = self.generate_crop_bb(mask.to_numpy(), region_id)
        except:
            print(case)
            raise ValueError
        images_left = []
        images_right = []
        for idx in range(len(images)):
            tmp_img1 = self.crop_image3d(images[idx], center_l[::-1], padvalue=self.default_values[idx])
            tmp_img2 = self.crop_image3d(images[idx], center_r[::-1], padvalue=self.default_values[idx])
            tmp_img2 = np.flip(tmp_img2, 2)
            images_left.append(torch.Tensor(tmp_img1.copy()).unsqueeze(0))
            images_right.append(torch.Tensor(tmp_img2.copy()).unsqueeze(0))
        images_left = torch.cat(images_left, dim=0)
        images_right = torch.cat(images_right, dim=0)
        mask_left = self.crop_image3d(mask, center_l[::-1], 0)
        mask_left = torch.Tensor(mask_left.copy()).unsqueeze(0)
        mask_left[mask_left != region_id] = 0
        mask_left[mask_left == region_id] = 1
        mask_right = self.crop_image3d(mask, center_r[::-1], 0)
        mask_right = torch.Tensor(np.flip(mask_right, 2).copy()).unsqueeze(0)
        mask_right[mask_right != region_id + 10] = 0
        mask_right[mask_right == region_id + 10] = 1

        mask_depress_left = self.crop_image3d(mask_depress, center_l[::-1], 0)
        mask_depress_left = torch.Tensor(mask_depress_left.copy()).unsqueeze(0) * mask_left
        mask_depress_right = self.crop_image3d(mask_depress, center_r[::-1], 0)
        mask_depress_right = torch.Tensor(np.flip(mask_depress_right, 2).copy()).unsqueeze(0) * mask_right

        if random.random() <= 0.2 and self.is_train:
            images_left = torch.flip(images_left, dims=[1])
            images_right = torch.flip(images_right, dims=[1])
            mask_left = torch.flip(mask_left, dims=[1])
            mask_right = torch.flip(mask_right, dims=[1])
            mask_depress_left = torch.flip(mask_depress_left, dims=[1])
            mask_depress_right = torch.flip(mask_depress_right, dims=[1])
        if random.random() <= 0.2 and self.is_train:
            images_left = torch.flip(images_left, dims=[2])
            images_right = torch.flip(images_right, dims=[2])
            mask_left = torch.flip(mask_left, dims=[2])
            mask_right = torch.flip(mask_right, dims=[2])
            mask_depress_left = torch.flip(mask_depress_left, dims=[2])
            mask_depress_right = torch.flip(mask_depress_right, dims=[2])
        if random.random() <= 0.2 and self.is_train:
            images_left = torch.flip(images_left, dims=[3])
            images_right = torch.flip(images_right, dims=[3])
            mask_left = torch.flip(mask_left, dims=[3])
            mask_right = torch.flip(mask_right, dims=[3])
            mask_depress_left = torch.flip(mask_depress_left, dims=[3])
            mask_depress_right = torch.flip(mask_depress_right, dims=[3])

        tag_left = int(tag_row[str(region_id)].item())
        tag_right = int(tag_row[str(region_id + 10)].item())

        if torch.sum(mask_depress_left) > 0:
            hu_left = float(torch.mean((1 + images_left)[mask_depress_left == 1]))
        else:
            hu_left = float(0)
        if torch.sum(mask_depress_right) > 0:
            hu_right = float(torch.mean((1 + images_right)[mask_depress_right == 1]))
        else:
            hu_right = float(0)

        if self.cbf_ratio is not None:

            if self.cbf_label is not True:
                power = 0.5
                inv_power = 2
                base = 0.1
            else:
                power = math.sqrt((math.log(0.2 - self.cbf_thres) - math.log(0.2))
                                  / (math.log(self.cbf_thres) - math.log(0.2)))
                inv_power = 1 / power
                base = 0
            try:
                ratio_left = self.cbf_ratio[self.cbf_ratio.ID == case][str(region_id)].item()
                if tag_left == 1:
                    weight_left = (min(ratio_left, 0.2) / 0.2) ** power + base
                else:
                    weight_left = (max(0.2 - ratio_left, 0) / 0.2) ** inv_power + base
                ratio_right = self.cbf_ratio[self.cbf_ratio.ID == case][str(region_id + 10)].item()
                if tag_right == 1:
                    weight_right = (min(ratio_right, 0.2) / 0.2) ** power + base
                else:
                    weight_right = (max(0.2 - ratio_right, 0) / 0.2) ** inv_power + base
            except:
                weight_left = 1
                weight_right = 1
        else:
            weight_left = 1
            weight_right = 1

        return images_left, images_right, mask_left, mask_right, mask_depress_left, mask_depress_right,\
               tag_left, tag_right, weight_left, weight_right, hu_left, hu_right, region_id, case_name

