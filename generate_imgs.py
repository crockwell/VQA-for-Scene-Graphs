import argparse
import h5py
import tqdm
import numpy as np
import pickle
import PIL
import yaml
import torch
from torch import nn
from torch.autograd import Variable
import json
import os
import sys
from tensorboardX import SummaryWriter
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.model import Sg2ImModel
from sg2im.data.vg_for_vqa import VgSceneGraphDataset, vg_collate_fn
import vqa_pytorch.vqa.datasets as datasets
from vqa_pytorch.vqa.models.att import MutanAtt # vqa model 
#import vqa.models as models

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path_opt', default='vqa_pytorch/options/vqa/mutan_att_trainval.yaml', type=str, 
    help='path to a yaml options file')

#args = parser.parse_args()
#with open(args.path_opt, 'r') as handle:
#    options = yaml.load(handle)
#    #options = utils.update_values(options, options_yaml)

#trainset = datasets.factory_VQA('trainval', opt_vgenome=options['vgenome'], opt=options['vqa'], opt_coco=options['coco'])
#LEN_VQA = 334554 # so we can access only visual genome items

def generate_img(objs, triples, model):
    '''
    takes scene graph, returns generated img
    TODO: get working with obj & triple
    obj_to_img will be all 0s like this I think
    '''
    O = objs.size(0)
    obj_to_img = torch.LongTensor(O).fill_(0)
    obj_to_img = obj_to_img.cuda()
    objs = objs.cuda()
    triples = triples.cuda()
    with torch.no_grad():
        model_out = model(objs, triples, obj_to_img)#, boxes_gt=model_boxes, masks_gt=model_masks)
    imgs, boxes_pred, masks_pred, predicate_scores = model_out
    '''
    with torch.no_grad():
        imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graph)
    '''
    imgs = imagenet_deprocess_batch(imgs)
    return imgs

def get_info(num_eval):
    '''
    gets all gt img, question, answer, scene graph
    '''
    VG_DIR = '/scratch/jiadeng_fluxoe/shared/vg'
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
    dset = VgSceneGraphDataset(**dset_kwargs)
    with open(os.path.join(VG_DIR, 'question_answers.json')) as data_file:
        data = json.load(data_file)
    
    #with open('qa_from_qid_master.pickle', 'rb') as handle:
    with open('idx_from_qid_master.pickle', 'rb') as handle:
        qa_from_qid = pickle.load(handle)
    
    #tr = tqdm.tqdm( range(0, num_eval), total = num_eval )
    gotten = 0
    for i in range(190,10000): #above 360 there are 0!
        if i % 10 == 0:
            print(i,gotten)
        #questions = []
        #answers = []
        #question_tensors = []
        #answer_tensors = []
        gt_img, objs, __, triples, img_id = dset.__getitem__(i)
        gt_img = gt_img.numpy().transpose(1,2,0) #64,64
        #feature_tensor = None
        c = 0
        for j in data:
            if j['id'] == img_id:
                added = False
                for k in j['qas']:
                    qid = k['qa_id']
                    try:
                        l = qa_from_qid[qid]#[0]
                    except:
                        continue
                    c += 1
                    added = True
                    #questions.append(k['question'])
                    #answers.append(k['answer'])
                    #print(k['question']) #where are the cpus being stored?
                    #print(k['answer']) #under the desk.
                    #print('shape visual',np.shape(k['visual']))
                    #print('img id', img_id, 'i', i, 'qa_id', k['qa_id'], 'vqa id', l) #id is 10, i is 0, qa id is 988197 (others sampled > 900k)
                    #item = trainset.__getitem__(l+LEN_VQA)
                    #question_tensors.append(item['question'].numpy())
                    #answer_tensors.append(item['answer'])
                    if c > 1:
                        break
                #if added:
                #    feature_tensor = item['visual']
            if c > 1:
                break
        if c > 1:
            gotten += 1
            yield objs, triples, i, gt_img
        if gotten >= num_eval:
            print('had to go through ',i,' to get ', gotten)
            break

def main():
    '''
    calls fcns to load info, answer questions, evaluate
    '''
    # Load the model, with a bit of care in case there are no GPUs
    print('loading scene gen model...')
    device = torch.device('cuda:0')
    map_location = 'cpu' if device == torch.device('cpu') else None
    checkpoint_theirs = torch.load('sg2im-models/vg64.pt', map_location=map_location)
    their_model = Sg2ImModel(**checkpoint_theirs['model_kwargs'])
    their_model.load_state_dict(checkpoint_theirs['model_state'])
    their_model.eval()
    their_model.to(device)
    
    
    checkpoint_mine = torch.load('vg_only.pt', map_location=map_location)
    my_model = Sg2ImModel(**checkpoint_mine['model_kwargs'])
    my_model.load_state_dict(checkpoint_mine['model_state'])
    my_model.eval()
    my_model.to(device)
    

    num_eval = 1000
    print('getting', str(num_eval), ' q, a, images...')
    images_theirs = {}
    images_mine = {}
    ct = 0
    for objs, triples, i, img in get_info(num_eval):
        print(ct)
        gen_img_theirs = generate_img(objs, triples, their_model)
        gen_img_mine = generate_img(objs, triples, my_model)
        gen_img_theirs = np.array(gen_img_theirs)
        gen_img_mine = np.array(gen_img_mine)
        #print(np.shape(gen_img_theirs))
        gen_img_theirs = np.transpose(gen_img_theirs, (0, 2, 3, 1))
        gen_img_mine = np.transpose(gen_img_mine, (0, 2, 3, 1))
        #print('their img',np.amax(gen_img_theirs), np.amin(gen_img_theirs))
        images_theirs[ct] = (gen_img_theirs, i) # should be of size 1,64,64,3 I think
        images_mine[ct] = (gen_img_mine, i)
        ct += 1
    
    with open('their_gen_imgs.pickle', 'wb') as handle:
        pickle.dump(images_theirs, handle)
        
    with open('my_gen_imgs.pickle', 'wb') as handle:
        pickle.dump(images_mine, handle)
    
if __name__ == '__main__':
    main()    
    