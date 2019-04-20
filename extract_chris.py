import argparse
import os
import time
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import pickle
from PIL import Image

import sys
sys.path.append('/scratch/jiadeng_fluxoe/cnris/tailpose/sg2im/vqa_pytorch')

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import vqa.models.convnets as convnets
import vqa.datasets as datasets
from vqa.lib.dataloader import DataLoader
from vqa.lib.logger import AvgMeter

import scipy

parser = argparse.ArgumentParser(description='Extract')
parser.add_argument('--dir_data', default='data/theirs')
parser.add_argument('--arch', '-a', default='resnet18')
parser.add_argument('--mode', default='att', type=str)
parser.add_argument('--size', default=448, type=int)

def main():
    global args
    args = parser.parse_args()

    print("=> using pre-trained model '{}'".format(args.arch))
    model = convnets.factory({'arch':'resnet50'}, cuda=True, data_parallel=True)

    extract_name = 'arch,{}_size,{}'.format(args.arch, args.size)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform=transforms.Compose([
                transforms.Scale(args.size),
                transforms.ToTensor(),
                normalize,
    ])
    '''
    if args.dataset == 'coco':
        if 'coco' not in args.dir_data:
            raise ValueError('"coco" string not in dir_data')
        dataset = datasets.COCOImages(args.data_split, dict(dir=args.dir_data),
            transform=transforms.Compose([
                transforms.Scale(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.dataset == 'vgenome':
        if args.data_split != 'train':
            raise ValueError('train split is required for vgenome')
        if 'vgenome' not in args.dir_data:
            raise ValueError('"vgenome" string not in dir_data')
        dataset = datasets.VisualGenomeImages(args.data_split, dict(dir=args.dir_data),
            transform=transforms.Compose([
                transforms.Scale(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                normalize,
            ]))

    data_loader = DataLoader(dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    batch_size = 1
    dir_extract = os.path.join(args.dir_data, 'extract', extract_name)
    path_file = os.path.join(dir_extract, 'set')

    extract(transform, model, path_file, args.mode)


def extract(transform, model, path_file, mode):
    path_hdf5 = path_file + '.hdf5'
    path_txt = path_file + '.txt'
    #hdf5_file = h5py.File(path_hdf5, 'w')

    # estimate output shapes
    output = model(Variable(torch.ones(1, 3, args.size, args.size),
                            volatile=True))

    nb_images = 1000
    '''
    if mode == 'both' or mode == 'att':
        shape_att = (nb_images, output.size(1), output.size(2), output.size(3))
        print('Warning: shape_att={}'.format(shape_att))
        #hdf5_att = hdf5_file.create_dataset('att', shape_att,
        #                                    dtype='f')#, compression='gzip')
    if mode == 'both' or mode == 'noatt':
        shape_noatt = (nb_images, output.size(1))
        print('Warning: shape_noatt={}'.format(shape_noatt))
        #hdf5_noatt = hdf5_file.create_dataset('noatt', shape_noatt,
        #                                      dtype='f')#, compression='gzip')
    '''
    model.eval()

    batch_time = AvgMeter()
    data_time  = AvgMeter()
    begin = time.time()
    end = time.time()

    idx = 0
    #dl = data_loader.__iter__()
    with open('their_gen_imgs.pickle', 'rb') as handle:
        img_set = pickle.load(handle)
    
    features = {}
    
    for i in range(nb_images):
        inp, ct = img_set[i]
        #inp[0] = scipy.misc.imresize(inp[0],(32,32)) #448
        #print('b4',np.shape(inp))
        #print(np.amax(inp), np.amin(inp))
        #print(type(inp))
        img = np.uint8(inp[0])
        #print(type(inp))
        img = Image.fromarray(img)
        img = transform(img)
        img = np.expand_dims(img, axis=0)
        input_var = Variable(torch.from_numpy(img), volatile=True) # should be size 1,3,448,448
        output_att = model(input_var)

        #nb_regions = output_att.size(2) * output_att.size(3)
        #output_noatt = output_att.sum(3).sum(2).div(nb_regions).view(-1, 2048)

        #batch_size = output_att.size(0)
        output = output_att.data.cpu().numpy() #1, 2048, 14, 14
        features[ct] = output
        
        #if mode == 'both' or mode == 'att':
        #    hdf5_att[idx:idx+batch_size]   = output_att.data.cpu().numpy()
        #if mode == 'both' or mode == 'noatt':
        #    #print(np.shape(output_noatt.data.cpu().numpy()))
        #    #print('2',np.shape(hdf5_noatt[idx:idx+batch_size]))
        #    hdf5_noatt[idx:idx+batch_size] = output_noatt.data.cpu().numpy()
        idx += 1
        print(ct)

        '''
        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 1 == 0:
            print('Extract: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                   i, 1,
                   batch_time=batch_time,
                   data_time=data_time,))
        '''
    with open('their_gen_features.pickle', 'wb') as handle:
        pickle.dump(features, handle)
    
    #hdf5_file.close()

    # Saving image names in the same order than extraction
    #with open(path_txt, 'w') as handle:
    #    for name in data_loader.dataset.dataset.imgs:
    #        handle.write(name + '\n')

    end = time.time() - begin
    print('Finished in {}m and {}s'.format(int(end/60), int(end%60)))


if __name__ == '__main__':
    main()
