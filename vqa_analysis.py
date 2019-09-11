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
from datetime import datetime
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.model import Sg2ImModel
from sg2im.data.vg_for_vqa import VgSceneGraphDataset, vg_collate_fn
import vqa_pytorch.vqa.datasets as datasets
from vqa_pytorch.vqa.models.att import MutanAtt # vqa model 
#import vqa.models as models
# when i left was running python vqa_pytorch/extract.py --dataset vgenome --dir_data data/vgenome --data_split train
# on flux.

'''
HOW TO RUN:
for all, choose say 500 images with max of 30 questions on each image
1. python generate_imgs.py with our model & their model
2. python extract_chris.py with our model & their model
3. python eval_vqa.py (need to add our model in as well).
* Might want to see which are val / test imgs etc.
'''

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path_opt', default='vqa_pytorch/options/vqa/mutan_att_trainval.yaml', type=str, 
    help='path to a yaml options file')

args = parser.parse_args()
with open(args.path_opt, 'r') as handle:
    options = yaml.load(handle)
    #options = utils.update_values(options, options_yaml)

trainset = datasets.factory_VQA('trainval', opt_vgenome=options['vgenome'], opt=options['vqa'], opt_coco=options['coco'])
LEN_VQA = 334554 # so we can access only visual genome items

# answer questions
def inference(vqa_model, image, question_set):
    '''
    need >1 question in question_set per image.
    no cuda. need our input image to be 
    '''
    answer_set = []
    size_q = len(question_set)
    image = np.tile(image,(size_q,1,1,1))
    input_visual = torch.autograd.Variable(torch.from_numpy(image), requires_grad=False)
    
    question_set = np.array(question_set)
    input_question = torch.autograd.Variable(torch.from_numpy(question_set), requires_grad=False)
    output = vqa_model(input_visual, input_question)
    _, pred = output.data.cpu().max(1)
    pred.squeeze_()
    return pred

def vg_eval(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers, i_s, questions, gt_imgs, my_imgs, their_imgs, objs, triples):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(log_dir='pictures'+'/runs/_'+current_time)
    
    gt_corr = {'one': 0, 'two': 0, 'three': 0, 'four': 0}
    gen_corr_theirs = {'one': 0, 'two': 0, 'three': 0, 'four': 0}
    gen_corr_mine = {'one': 0, 'two': 0, 'three': 0, 'four': 0}
    count = {'one': 0, 'two': 0, 'three': 0, 'four': 0}
    img_scores = []
    for answer_tensor, answer_gt, ans_gen_theirs, ans_gen_mine, answer, i, gt_img, my_img, their_img, question, obj_set, triple_set in zip(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers, i_s, gt_imgs, my_imgs, their_imgs, questions, objs, triples):
        img_score = [0, 0, 0, i, 0]
        print('i: ', str(i))
        print('triple,ob: ', obj_set, triple_set)
        for ans, agt, a, theirs, mine, q in zip(answer_tensor, answer_gt, answer, ans_gen_theirs, ans_gen_mine, question):
            word_ct = 'one'
            ct = a.count(' ')
            if ct == 1:
                word_ct = 'two'
            elif ct == 2:
                word_ct = 'three'
            elif ct > 2:
                word_ct = 'four'
            print(q, a, ans, agt.item(), theirs.item(), mine.item()) #.item() 
            if ans == agt.item(): #.item() 
                gt_corr[word_ct] += 1
                img_score[0] += 1 
            if ans == theirs.item():
                gen_corr_theirs[word_ct] += 1
                img_score[1] += 1 
            if ans == mine.item():
                gen_corr_mine[word_ct] += 1
                img_score[2] += 1 
            count[word_ct] += 1
            img_score[4] += 1
        img_scores.append(img_score)
        writer.add_image('gt '+str(i) + ' score ' + str(img_score[0]) + ' out of ' + str(img_score[4]), gt_img)
        writer.add_image('mine '+str(i) + ' score ' + str(img_score[2]) + ' out of ' + str(img_score[4]), my_img[0])
        writer.add_image('theirs '+str(i) + ' score ' + str(img_score[1]) + ' out of ' + str(img_score[4]), their_img[0])
    
    for ct in gt_corr.keys():
        print('accuracy, answers from gt, ', ct, ' word: ', round(gt_corr[ct]/max(count[ct],1),3), 'count', count[ct])
        print('accuracy, answers from generated (theirs), ', ct, ' word: ', round(gen_corr_theirs[ct]/max(count[ct],1),3), 'count', count[ct])
        print('accuracy, answers from generated (mine), ', ct, ' word: ', round(gen_corr_mine[ct]/max(count[ct],1),3), 'count', count[ct])
    total_ct = count['one'] + count['two'] + count['three'] + count['four']
    total_corr = gt_corr['one'] + gt_corr['two'] + gt_corr['three'] + gt_corr['four']
    their_corr = gen_corr_theirs['one'] + gen_corr_theirs['two'] + gen_corr_theirs['three'] + gen_corr_theirs['four']
    my_corr = gen_corr_mine['one'] + gen_corr_mine['two'] + gen_corr_mine['three'] + gen_corr_mine['four']
    print('accuracy, answers from gt, total: ', round(total_corr/max(total_ct,1),3), 'count', total_ct)
    print('accuracy, answers from generated (theirs), total: ', round(their_corr/max(total_ct,1),3), 'count', total_ct)
    print('accuracy, answers from generated (mine), total: ', round(my_corr/max(total_ct,1),3), 'count', total_ct)
    
    img_scores = np.array(img_scores)
    diffs = img_scores[:,2] - img_scores[:,1] # high is good
    diffs_gt = img_scores[:,2] - img_scores[:,0] # high is good
    
    print('median difference in score vs. theirs', np.median(diffs), ', vs. gt: ', np.median(diffs_gt))
    best_vs_them = diffs.argsort()[-5:][::-1]
    print('our best elements vs. them', best_vs_them, ', scores: ', img_scores[best_vs_them,:])
    best_vs_gt = diffs_gt.argsort()[-5:][::-1]
    print('our best elements vs. gt', best_vs_gt, ', scores: ', img_scores[best_vs_gt,:])
    worst_vs_them = diffs.argsort()[:5]
    print('our worst elements vs. them', worst_vs_them, ', scores: ', img_scores[worst_vs_them,:])
    worst_vs_gt = diffs_gt.argsort()[:5]
    print('our worst elements vs. gt', worst_vs_gt, ', scores: ', img_scores[worst_vs_gt,:])
    
    writer.close()
    
def generate_img(objs, triples, model):
    '''
    takes scene graph, returns generated img
    '''
    O = objs.size(0)
    obj_to_img = torch.LongTensor(O).fill_(0)
    obj_to_img = obj_to_img.cuda()
    objs = objs.cuda()
    triples = triples.cuda()
    with torch.no_grad():
        model_out = model(objs, triples, obj_to_img)#, boxes_gt=model_boxes, masks_gt=model_masks)
    imgs, boxes_pred, masks_pred, predicate_scores = model_out
    imgs = imagenet_deprocess_batch(imgs)
    return imgs

def get_info(is_to_check, their_model):
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
    
    with open('idx_from_qid_master.pickle', 'rb') as handle:
        qa_from_qid = pickle.load(handle)
        
    with open('their_gen_features.pickle', 'rb') as handle:
        their_features = pickle.load(handle)
        
    with open('my_gen_features.pickle', 'rb') as handle:
        my_features = pickle.load(handle)
    
    for i in is_to_check:
        questions = []
        answers = []
        question_tensors = []
        answer_tensors = []
        gt_img, objs, __, triples, img_id = dset.__getitem__(i)
        gt_img = gt_img.numpy().transpose(1,2,0) #64,64
        feature_tensor = None
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
                    questions.append(k['question'])
                    answers.append(k['answer'])
                    item = trainset.__getitem__(l+LEN_VQA)
                    question_tensors.append(item['question'].numpy())
                    answer_tensors.append(item['answer'])
                    if c > 29:
                        break
                if c > 1: # must have at least 2 questions!
                    feature_tensor = item['visual']
        their_feats = their_features[i]
        my_feats = my_features[i]
        print('i: ', str(i), 'triple,ob: ', objs, triples)
        print('printing objects')
        for object_ in objs:
            print(their_model.vocab['object_idx_to_name'][object_])
        for trip in triples:
            print(trip[0], their_model.vocab['pred_idx_to_name'][trip[1]], trip[2])
        yield gt_img, questions, answers, objs, triples, question_tensors, answer_tensors, feature_tensor, their_feats, my_feats

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
    
    gt_imgs = []
    #gen_imgs = []
    questions = []
    answers = []
    question_tensors = []
    answer_tensors = []
    vqa_gt = []
    vqa_gen_theirs = []
    vqa_gen_mine = []
    objs = []
    triples = []
    feature_tensors = []
    their_feat_tensors = []
    my_feat_tensors = []
    my_imgs = []
    their_imgs = []

    print('getting q, a, images...')
    #i_s = [225, 227, 209, 1556] # good vs theirs
    #i_s = [204, 207, 1715, 675, 225] # good vs gt
    #i_s = [198, 221, 547, 207, 859] # bad vs them
    #i_s = [209, 215, 214, 1144, 217] # bad vs gt
    #i_s = [200, 218, 203, 226, 651, 526, 216, 483, 543, 837, 635, 1003]
    i_s = [651, 226, 218, 209, 1003, 837]
    for gt_img, question_set, answer_set, obj, triple, question_tensor_set, answer_tensor_set, feature_tensor, their_feats, my_feats in get_info(i_s, their_model):
        gt_imgs.append(gt_img)
        questions.append(question_set)
        answers.append(answer_set)
        objs.append(obj)
        triples.append(triple)
        question_tensors.append(question_tensor_set)
        answer_tensors.append(answer_tensor_set)
        feature_tensors.append(feature_tensor)
        their_feat_tensors.append(their_feats)
        my_feat_tensors.append(my_feats)

    print('loading vqa model...')
    vqa_model = MutanAtt(options['model'], trainset.vocab_words(), trainset.vocab_answers())
    
    path_ckpt_model = 'vqa_pytorch/vqa/mutan_att_trainval/ckpt_model.pth.tar'
    model_state = torch.load(path_ckpt_model)
    vqa_model.load_state_dict(model_state)
    vqa_model.eval()
        
    for i in range(len(gt_imgs)):
        gt_img = gt_imgs[i]
        question_set_tensors = question_tensors[i]
        obj = objs[i]
        feature_tensor = feature_tensors[i]
        triple = triples[i]
        their_feats = their_feat_tensors[i]
        my_feats = my_feat_tensors[i]
        
        # answer, gt
        vqa_answer_from_gt = inference(vqa_model, feature_tensor, question_set_tensors)
        vqa_gt.append(vqa_answer_from_gt)
        
        vqa_theirs = inference(vqa_model, their_feats, question_set_tensors)
        vqa_gen_theirs.append(vqa_theirs)
        
        vqa_mine = inference(vqa_model, my_feats, question_set_tensors)
        vqa_gen_mine.append(vqa_mine)
        
        # generating imgs
        gen_img_theirs = generate_img(obj, triple, their_model)
        gen_img_mine = generate_img(obj, triple, my_model)
        gen_img_theirs = np.array(gen_img_theirs)
        gen_img_mine = np.array(gen_img_mine)
        my_imgs.append(gen_img_theirs)
        their_imgs.append(gen_img_mine)

    vg_eval(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers, i_s, questions, gt_imgs, my_imgs, their_imgs, objs, triples)
    
if __name__ == '__main__':
    main()    
    