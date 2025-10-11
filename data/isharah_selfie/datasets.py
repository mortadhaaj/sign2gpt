from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time
from vidaug import augmentors as va
from augmentation import *

from loguru import logger
from hpman.m import _
import av
import zipfile

# global definition
from definition import *

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        if isinstance(Image, PIL.Image.Image):
            Image = np.asarray(Image, dtype=np.uint8)
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class SomeOf(object):
    """
    Selects one augmentation from a list.
    Args:
        transforms (list of "Augmentor" objects): The list of augmentations to compose.
    """

    def __init__(self, transforms1, transforms2):
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, clip):
        select = random.choice([0, 1, 2])
        if select == 0:
            return clip
        elif select == 1:
            if random.random() > 0.5:
                return self.transforms1(clip)
            else:
                return self.transforms2(clip)
        else:
            clip = self.transforms1(clip)
            clip = self.transforms2(clip)
            return clip

class S2T_Dataset(Dataset.Dataset):
    def __init__(self,path,tokenizer,config,args,phase, training_refurbish=False):
        # if phase == 'train':
        #     phase = 'val'  # use val set for training
        self.config = config
        self.args = args
        self.training_refurbish = training_refurbish
        
        self.raw_data = utils.load_dataset_file(path)
        self.tokenizer = tokenizer
        # self.img_path = config['data']['img_path']
        self.phase = phase
        self.max_length = config['data']['max_length']
        self.list = []
        for vid_pth in self.raw_data[self.raw_data['split']==phase].video_pth:
            self.list.append(f'dataset_256/{vid_pth}')
        #self.list = [key for key,value in self.raw_data.items()]
        # self._zip = zipfile.ZipFile(config['data']['zip_path'], 'r', allowZip64=True)

        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.1)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10)),

            # sometimes(Brightness(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),

        ])
        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),
            # sometimes(Sharpness(min=0.1, max=2.))
        ])
        # self.seq = SomeOf(self.seq_geo, self.seq_color)

    def __len__(self):
        return len(self.list)
    
    # def _ensure_zip(self):
    #     if self._zip is None:
    #         # allowZip64 for big archives; keep handle open per worker
    #         self._zip = zipfile.ZipFile(self.config['data']['zip_path'], 'r', allowZip64=True)
    
    def __getitem__(self, index):
        key = self.list[index]
        _, _, tgt_sample, length  = self.raw_data[self.raw_data['video_pth']==key.split('/', 1)[1]].values[0]
        # tgt_sample = sample['sentence']
        # length = sample['length']
        
        name_sample = key
        # name_sample = sample['name']

        # img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']])
        img_sample = self.load_imgs(name_sample, length)
        
        return name_sample,img_sample,tgt_sample
    
    def load_imgs(self, name_sample, num_frames):

        data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])
        # self._ensure_zip()
        with zipfile.ZipFile(self.config['data']['zip_path'], 'r', allowZip64=True) as _zip:
            with _zip.open(name_sample, 'r') as f:
                file_bytes = f.read()

        # Open from memory (BytesIO) with PyAV
        bio = io.BytesIO(file_bytes)
        try:
            container = av.open(bio, mode='r', options={"scan_all_pmts": "1"})
        except:
            raise ValueError(f"Failed to open video file in zip: {name_sample}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.skip_frame = "DEFAULT"  # decode all frames (change to 'DEFAULT' if needed)

        if num_frames > self.max_length:
            tmp = set(sorted(random.sample(range(num_frames), k=self.max_length)))
        else:
            tmp = None
        frames = []
        for idx, frame in enumerate(container.decode(stream)):
            if tmp is not None and idx not in tmp:
                continue
            img = frame.to_rgb().to_ndarray()  # H,W,3 uint8
            # ten = torch.from_numpy(img).permute(2,0,1).float() / 255.0  # (3,H,W) float
            # ten = torch.from_numpy(img).permute(2,0,1).contiguous()
            img = Image.fromarray(img)
            frames.append(img)
        container.close()
        # return frames
    
        imgs = torch.zeros(num_frames,3, self.args.input_size,self.args.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize), crop_size=self.args.input_size, is_train=(self.phase=='train'))

        if self.phase == 'train':
            batch_image = self.seq(frames)
        else:
            batch_image = frames

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i,:,:,:] = img[:,:,crop_rect[1]:crop_rect[3],crop_rect[0]:crop_rect[2]]
            # imgs[i,:,:,:] = img[:,:,:,:]
        
        return imgs

    def collate_fn(self,batch):
        
        tgt_batch,img_tmp,src_length_batch,name_batch = [],[],[],[]

        for name_sample, img_sample, tgt_sample in batch:

            name_batch.append(name_sample)

            img_tmp.append(img_sample)

            tgt_batch.append(tgt_sample)

        max_len = max([len(vid) for vid in img_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
            )
            , dim=0)
            for vid in img_tmp]
        
        img_tmp = [padded_video[i][0:video_length[i],:,:,:] for i in range(len(padded_video))]
        
        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
        src_length_batch = torch.tensor(src_length_batch)
        
        img_batch = torch.cat(img_tmp,0)

        new_src_lengths = (((src_length_batch-5+1) / 2)-5+1)/2
        new_src_lengths = new_src_lengths.long()
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX,batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt",padding = True,  truncation=True)

        src_input = {}
        src_input['input_ids'] = img_batch
        src_input['attention_mask'] = img_padding_mask
        src_input['name_batch'] = name_batch

        src_input['src_length_batch'] = src_length_batch
        src_input['new_src_length_batch'] = new_src_lengths
        
        if self.training_refurbish:
            masked_tgt = utils.NoiseInjecting(tgt_batch, self.args.noise_rate, noise_type=self.args.noise_type, random_shuffle=self.args.random_shuffle, is_train=(self.phase=='train'))
            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
            return src_input, tgt_input, masked_tgt_input
        return src_input, tgt_input

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'







