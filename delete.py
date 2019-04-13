import h5py
import tqdm
import numpy as np
import PIL
import torch
import json
import os
from tensorboardX import SummaryWriter
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.model import Sg2ImModel
from sg2im.data.vg_for_vqa import VgSceneGraphDataset, vg_collate_fn

VG_DIR = '/home/shared/vg/'
vocab_json = os.path.join(VG_DIR, 'vocab.json')
with open(vocab_json, 'r') as f:
    vocab = json.load(f)
dset_kwargs = {
    'vocab': vocab,
    'h5_path': os.path.join(VG_DIR, 'test.h5'),
    'image_dir': os.path.join(VG_DIR, 'images'),
    'image_size': (64,64),
    'max_objects': 10,
    'use_orphaned_objects': True,
    'include_relationships': True,
    'normalize_images': False,
  }
train_dset = VgSceneGraphDataset(**dset_kwargs)
'''
loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': True,
    'collate_fn': collate_fn,
  }
'''
#train_loader = DataLoader(train_dset, **loader_kwargs)
#batch = 
#batch = [tensor.cuda() for tensor in batch]
image, objs, boxes, triples, img_id = train_dset.__getitem__(10)
image = image.numpy().transpose(1,2,0) #64,64
print(img_id)

writer = SummaryWriter(log_dir='tensorboard_output/test1/runs/a4')
writer.add_image('img10', image)
writer.close()