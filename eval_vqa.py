import argparse
import h5py
import tqdm
import numpy as np
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
# when i left was running python vqa_pytorch/extract.py --dataset vgenome --dir_data data/vgenome --data_split train
# on flux.
    
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vg', choices=['vg', 'coco'])

# Optimization hyperparameters
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_iterations', default=1000000, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)

# Switch the generator to eval mode after this many iterations
parser.add_argument('--eval_mode_after', default=100000, type=int)

# Dataset options common to both VG and COCO
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=1024, type=int)
parser.add_argument('--shuffle_val', default=True, type=bool_flag)
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)

# VG-specific options
parser.add_argument('--vg_image_dir', default=os.path.join(VG_DIR, 'images'))
parser.add_argument('--train_h5', default=os.path.join(VG_DIR, 'train.h5'))
parser.add_argument('--val_h5', default=os.path.join(VG_DIR, 'val.h5'))
parser.add_argument('--vocab_json', default=os.path.join(VG_DIR, 'vocab.json'))
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)

# COCO-specific options
parser.add_argument('--coco_train_image_dir',
         default=os.path.join(COCO_DIR, 'images/train2017'))
parser.add_argument('--coco_val_image_dir',
         default=os.path.join(COCO_DIR, 'images/val2017'))
parser.add_argument('--coco_train_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_train2017.json'))
parser.add_argument('--coco_train_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_train2017.json'))
parser.add_argument('--coco_val_instances_json',
         default=os.path.join(COCO_DIR, 'annotations/instances_val2017.json'))
parser.add_argument('--coco_val_stuff_json',
         default=os.path.join(COCO_DIR, 'annotations/stuff_val2017.json'))
parser.add_argument('--instance_whitelist', default=None, type=str_tuple)
parser.add_argument('--stuff_whitelist', default=None, type=str_tuple)
parser.add_argument('--coco_include_other', default=False, type=bool_flag)
parser.add_argument('--min_object_size', default=0.02, type=float)
parser.add_argument('--min_objects_per_image', default=3, type=int)
parser.add_argument('--coco_stuff_only', default=True, type=bool_flag)

# Generator options
parser.add_argument('--mask_size', default=16, type=int) # Set this to 0 to use no masks
parser.add_argument('--embedding_dim', default=128, type=int)
parser.add_argument('--gconv_dim', default=128, type=int)
parser.add_argument('--gconv_hidden_dim', default=512, type=int)
parser.add_argument('--gconv_num_layers', default=5, type=int)
parser.add_argument('--mlp_normalization', default='none', type=str)
parser.add_argument('--refinement_network_dims', default='1024,512,256,128,64', type=int_tuple)
parser.add_argument('--normalization', default='batch')
parser.add_argument('--activation', default='leakyrelu-0.2')
parser.add_argument('--layout_noise_dim', default=32, type=int)
parser.add_argument('--use_boxes_pred_after', default=-1, type=int)

# Generator losses
parser.add_argument('--mask_loss_weight', default=0, type=float)
parser.add_argument('--l1_pixel_loss_weight', default=1.0, type=float)
parser.add_argument('--bbox_pred_loss_weight', default=10, type=float)
parser.add_argument('--predicate_pred_loss_weight', default=0, type=float) # DEPRECATED

# Generic discriminator options
parser.add_argument('--discriminator_loss_weight', default=0.01, type=float)
parser.add_argument('--gan_loss_type', default='gan')
parser.add_argument('--d_clip', default=None, type=float)
parser.add_argument('--d_normalization', default='batch')
parser.add_argument('--d_padding', default='valid')
parser.add_argument('--d_activation', default='leakyrelu-0.2')

# Object discriminator
parser.add_argument('--d_obj_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--crop_size', default=32, type=int)
parser.add_argument('--d_obj_weight', default=1.0, type=float) # multiplied by d_loss_weight 
parser.add_argument('--ac_loss_weight', default=0.1, type=float)

# Image discriminator
parser.add_argument('--d_img_arch',
    default='C4-64-2,C4-128-2,C4-256-2')
parser.add_argument('--d_img_weight', default=1.0, type=float) # multiplied by d_loss_weight

# Output options
parser.add_argument('--print_every', default=10, type=int)
parser.add_argument('--timing', default=False, type=bool_flag)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)
    
args = parser.parse_args()
    
dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.val_h5,
    'image_dir': args.vg_image_dir,
    'image_size': args.image_size,
    'max_samples': args.num_train_samples,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
}
val_dset = VgSceneGraphDataset(**dset_kwargs)

loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': collate_fn,
}

val_loader = DataLoader(val_dset, **loader_kwargs)
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
    answer_set = []
    for q in question_set:
        input_visual = torch.autograd.Variable(torch.from_numpy(image), requires_grad=False).cuda()#Variable(image.cuda(async=True), volatile=True)
        input_question = torch.autograd.Variable(torch.from_numpy(q), requires_grad=False).cuda()#Variable(q.cuda(async=True), volatile=True)
        output = vqa_model(input_visual, input_question)
        print(output)
        _, pred = output.data.cpu().max(1)
        print(pred)
        pred.squeeze_()
        print(pred)
        answer_set.append(pred) #output 
    return answer_set

def vg_eval(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers):
    '''
    must match answer exactly. However, only use questions with 2 or 1 word answers (relatively easy). 
    May also try 3 word answers
    '''
    gt_corr = {'one': 0, 'two': 0, 'three': 0}
    gen_corr_theirs = {'one': 0, 'two': 0, 'three': 0}
    gen_corr_mine = {'one': 0, 'two': 0, 'three': 0}
    count = {'one': 0, 'two': 0, 'three': 0}
    for answer_tensor, answer_gt, ans_gen_theirs, ans_gen_mine, answer in zip(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers):
        word_ct = 'one'
        ct = answer.count(' ')
        if ct == 1:
            word_ct = 'two'
        elif ct == 2:
            word_ct = 'three'
        elif ct > 2:
            print('longer than 3 words: ', answer)
            continue
        if answer_tensor == answer_gt:
            gt_corr[word_ct] += 1
        if answer_tensor == ans_gen_theirs:
            gen_corr_theirs[word_ct] += 1
        if answer_tensor == ans_gen_mine:
            gen_corr_mine[word_ct] += 1
        count[word_ct] += 1
    
    for ct in gt_corr.keys():
        print('accuracy, answers from gt, ', ct, ' word: ', round(gt_corr[ct]/count[ct]))
        print('accuracy, answers from generated (theirs), ', ct, ' word: ', round(gen_corr_theirs[ct]/count[ct]))
        print('accuracy, answers from generated (mine), ', ct, ' word: ', round(gen_corr_mine[ct]/count[ct]))
    print('accuracy, answers from gt, total: ', round(gt_corr[ct]/count[ct]))
    print('accuracy, answers from generated (theirs), total: ', round(gen_corr_theirs[ct]/count[ct]))
    print('accuracy, answers from generated (mine), total: ', round(gen_corr_mine[ct]/count[ct]))
    
def generate_img(obj, triple, model):
    '''
    takes scene graph, returns generated img
    TODO: get working with obj & triple
    obj_to_img will be all 0s like this I think
    '''
    O = objs.size(0)
    obj_to_img = torch.LongTensor(O).fill_(0)
    with torch.no_grad():
        model_out = model(objs, triples, obj_to_img, boxes_gt=model_boxes, masks_gt=model_masks)
    imgs, boxes_pred, masks_pred, predicate_scores = model_out
    print(np.shape(imgs))
    '''
    with torch.no_grad():
        imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graph)
    '''
    #imgs = imagenet_deprocess_batch(imgs)
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
    
    with open('qa_from_qid.pickle', 'rb') as handle:
        qa_from_qid = pickle.load(handle)
    
    tr = tqdm.tqdm( range(0, num_eval), total = num_eval )
    for i in tr:
        questions = []
        answers = []
        question_tensors = []
        answer_tensors = []
        gt_img, objs, __, triples, img_id = dset.__getitem__(i)
        gt_img = gt_img.numpy().transpose(1,2,0) #64,64
        for j in data:
            if j['id'] == img_id:
                for k in j['qas']:
                    questions.append(k['question'])
                    answers.append(k['answer'])
                    qid = k['qa_id']
                    print(k['question']) #where are the cpus being stored?
                    print(k['answer']) #under the desk.
                    print('img id', img_id, 'i', i, 'qa_id', k['qa_id']) #id is 10, i is 0, qa id is 988197 (others sampled > 900k)
                    question_tensors.append(qa_from_qid[qid][0])
                    answer_tensors.append(qa_from_qid[qid][1])
                    return ya
        yield gt_img, questions, answers, objs, triples, question_tensors, answer_tensors
    
    '''
    tr = tqdm.tqdm( range(0, num_eval), total = num_eval )
    val_f = h5py.File('/home/shared/vg/qa_challenge_chris.h5', 'r')
    VG_DIR = '/home/shared/vg/images/VG_100K'
    imgs = set()
    for i in num_eval:
        question = '%s' % (val_f['question'][i].decode('UTF-8')) 
        answer = '%s' % (val_f['answer'][i].decode('UTF-8')) 
        img_id = val_f['image_id'][i]
        img_path = os.path.join(VG_DIR, str(img_id)+'.jpg')
        with open(img_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                gt_img = self.transform(image.convert('RGB'))
        yield gt_img, question, answer, img_path, scene_graph
    '''

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
    
    # load vqa model
    
    '''
    options = {
        'model': {
            'arch': 'MutanNoAtt',
            'seq2vec': {
                'type': args.st_type,
                'dropout': args.st_dropout,
                'fixed_emb': args.st_fixed_emb
            }
        }
    }
    '''
    #train_loader = trainset.data_loader(batch_size=1,num_workers=args.workers,shuffle=False) 
    
    #vqa_model = getattr(sys.modules[__name__], options['arch'])(options, trainset.vocab_words(), trainset.vocab_answers())
    #vocab_words = h5py.File('vqa/data/vocab_words.h5','r')
    #vocab_answers = h5py.File('vqa/data/vocab_answers.h5','r')

    #args.start_epoch, best_acc1, exp_logger = load_checkpoint(model.module, optimizer,
    #        os.path.join(options['logs']['dir_logs'], args.resume))
    #test(test_loader, model, exp_logger, start_epoch, print_freq)
    
    
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
    #img_paths = []

    print('getting q, a, images...')
    for gt_img, question_set, answer_set, obj, triple, question_tensor_set, answer_tensor_set in get_info(num_eval=2):
        gt_imgs.append(gt_img)
        questions.append(question_set)
        answers.append(answer_set)
        objs.append(obj)
        triples.append(triple)
        question_tensors.append(question_tensor_set)
        answer_tensors.append(answer_tensor_set)
    '''    
    vocab_answers = []
    vocab_words = []
    for i in range(len(questions)):
        for j in range(len(questions[i])):
            vocab_answers.append(answers[i][j])
            words = questions[i][j].split()
            for word in words:
                vocab_words.append(word)
    print(len(vocab_answers), len(vocab_words))        
    vqa_model = MutanAtt(options['model'], vocab_words, vocab_answers)#trainset.vocab_words(), trainset.vocab_answers())
    '''
    print('loading vqa model...')
    vqa_model = MutanAtt(options['model'], trainset.vocab_words(), trainset.vocab_answers())
    #vqa_model = nn.DataParallel(vqa_model).cuda()
    #vqa_model.cuda()
    
    path_ckpt_model = 'vqa_pytorch/vqa/mutan_att_trainval/ckpt_model.pth.tar'
    model_state = torch.load(path_ckpt_model)
    vqa_model.load_state_dict(model_state)
    vqa_model.eval()
        
    for i in range(len(gt_imgs)):
        gt_img = gt_imgs[i]
        question_set = questions[i]
        question_set_tensors = question_tensors[i]
        obj = objs[i]
        triple = triples[i]
        #img_paths.append(img_path)
        
        # answer, gt
        vqa_answer_from_gt = inference(vqa_model, gt_img, question_set_tensors)
        vqa_gt.append(vqa_answer_from_gt)
        
        # answer, theirs
        gen_img_theirs = generate_img(obj, triple, their_model)
        #gen_imgs.append(gen_img)
        vqa_answer_from_gen_theirs = inference(vqa_model, gen_img_theirs, question_set_tensors)
        vqa_gen_theirs.append(vqa_answer_from_gen_theirs)
        
        # answer, mine
        gen_img_mine = generate_img(obj, triple, my_model)
        #gen_imgs.append(gen_img)
        vqa_answer_from_gen_mine = inference(vqa_model, gen_img_mine, question_set_tensors)
        vqa_gen_mine.append(vqa_answer_from_gen_mine)

    vg_eval(answer_tensors, vqa_gt, vqa_gen_theirs, vqa_gen_mine, answers)
    
if __name__ == '__main__':
    main()    
    