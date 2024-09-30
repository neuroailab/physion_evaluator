import glob

import h5py
from torch.utils.data import Dataset
from PIL import Image

from . import data_utils  # Disable for TPU training

buggy_stims = "pilot-containment-cone-plate_0017 \
pilot-containment-cone-plate_0022 \
pilot-containment-cone-plate_0029 \
pilot-containment-cone-plate_0034 \
pilot-containment-multi-bowl_0042 \
pilot-containment-multi-bowl_0048 \
pilot-containment-vase_torus_0031 \
pilot_dominoes_SJ020_d3chairs_o1plants_tdwroom_0005 \
pilot_it2_collision_non-sphere_box_0002 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0004 \
pilot_it2_collision_non-sphere_tdw_1_dis_1_occ_0007 \
pilot_it2_drop_simple_box_0000 \
pilot_it2_drop_simple_box_0042 \
pilot_it2_drop_simple_tdw_1_dis_1_occ_0003 \
pilot_it2_rollingSliding_simple_collision_box_0008 \
pilot_it2_rollingSliding_simple_collision_box_large_force_0009 \
pilot_it2_rollingSliding_simple_collision_tdw_1_dis_1_occ_0002 \
pilot_it2_rollingSliding_simple_ledge_tdw_1_dis_1_occ_sphere_small_zone_0022 \
pilot_it2_rollingSliding_simple_ramp_box_small_zone_0006 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0004 \
pilot_it2_rollingSliding_simple_ramp_tdw_1_dis_1_occ_small_zone_0017 \
pilot_linking_nl1-8_mg000_aCyl_bCyl_tdwroom1_long_a_0022 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1_0012 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0006 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0010 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0029 \
pilot_linking_nl1-8_mg000_aCylcap_bCyl_tdwroom_small_rings_0036 \
pilot_linking_nl6_aNone_bCone_occ1_dis1_boxroom_0028 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0000 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0002 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0003 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0010 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0013 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0017 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0018 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0032 \
pilot_towers_nb4_fr015_SJ000_gr000sph_mono1_dis0_occ0_tdwroom_stable_0036 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0021 \
pilot_towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable_0041 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0006 \
pilot_towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable_0009".split(' ')

# pilot_linking_nl1-8_mg000_aNone_bCyl_tdwroom_small_rings_0033
import numpy as np


def get_object_masks(seg_imgs):
    try:
        seg_colors = np.unique(seg_imgs)[1:]
        obj_masks = []
        for scol in seg_colors:
            mask = (seg_imgs == scol)
            # filter out small masks
            if mask.sum() < 200:
                continue
            obj_masks.append(mask)

        obj_masks = np.stack(obj_masks, 0)
    except:
        seg_colors = np.unique(seg_imgs)
        obj_masks = []
        for scol in seg_colors:
            mask = (seg_imgs == scol)
            if mask.sum() < 200:
                continue
            obj_masks.append(mask)
        obj_masks = np.stack(obj_masks, 0)
    return obj_masks


def get_label(f):
    with h5py.File(f) as h5file:
        for key in h5file['frames'].keys():
            lbl = np.array(h5file['frames'][key]['labels']['target_contacting_zone']).item()
            if lbl:
                return int(key), True

        ind = len(h5file['frames'].keys()) // 2

        return ind, False


from torch.utils.data._utils.collate import default_collate


def custom_collate(batch):
    keys = batch[0].keys()

    collated_batch = {}
    for key in keys:
        # Check if the current key corresponds to the list of tensors
        if isinstance(batch[0][key], list):
            # If so, just concatenate the lists from all samples
            collated_batch[key] = [item for sample in batch for item in sample[key]]
        else:
            # For all other keys, use the default collate function
            collated_batch[key] = default_collate([sample[key] for sample in batch])

    return collated_batch


import json
import imageio


class Physion(Dataset):

    def __init__(self, frame_gap=150, num_frames=4, transform=None, mp4_paths='', labels_path='', task='ocp',
                 mode='train'):

        if mode == 'test':
            self.buggy_stims = buggy_stims
        else:
            self.buggy_stims = []

        self.task = task

        self.all_mp4s = glob.glob(mp4_paths)

        with open(labels_path, 'r') as file:
            self.labels = json.load(file)

        blacklisted_inds = []

        for ct, f in enumerate(self.all_mp4s):
            if str(f).split('/')[-1].split('.')[0] not in self.buggy_stims:
                blacklisted_inds.append(f)

        self.all_mp4s = blacklisted_inds

        self.get_label = get_label

        self.transform = transform

        self.background = True

        self.frame_gap = frame_gap

        self.req_video_len = max(num_frames, 450 // self.frame_gap + 1)

        print(f"Number of videos: {len(self.all_mp4s)}")

    def __len__(self):

        return len(self.all_mp4s)

    def __getitem__(self, idx):

        filename = self.all_mp4s[idx]

        video_reader = imageio.get_reader(filename, 'ffmpeg')

        labels = self.labels[filename.split('/')[-1].split('.')[0]]

        num_frames = video_reader.count_frames()

        frame_label = labels['contact_frame_label']
        ret = {}
        ret['label'] = labels['label']
        ret['frame_label'] = frame_label

        if self.task == 'ocd':
            indices = np.arange(frame_label + 15, (frame_label - 31), -self.frame_gap // 10).clip(0, num_frames - 1)[:-1]
            # if size of indices if less than self.num_frames, then repeat the first frame
            if len(indices) < self.req_video_len:
                indices = np.concatenate([np.array([indices[0]] * (self.req_video_len - len(indices))), indices])
        elif self.task == 'ocp':

            if 'collision' in filename and 'roll' not in filename:
                max_frame = 15
            else:
                max_frame = 45

            indices = np.arange(max_frame, -1, -self.frame_gap // 10).clip(0, num_frames - 1)[::-1].copy()

            if len(indices) < self.req_video_len:
                indices = np.concatenate([np.array([indices[0]] * (self.req_video_len - len(indices))), indices])

        image_list = [Image.fromarray(video_reader.get_data(ind)) for ind in indices]

        video_reader.close()

        if self.transform is not None:
            image_tensor = self.transform(image_list)
            image_tensor = image_tensor.view(len(image_list),  # num_frames
                                             3,  # num_channels
                                             *image_tensor.shape[-2:]).contiguous()
        else:
            image_tensor = np.stack(image_list)

        ret['video'] = image_tensor
        ret['filename'] = filename
        ret['indices'] = indices

        return ret
