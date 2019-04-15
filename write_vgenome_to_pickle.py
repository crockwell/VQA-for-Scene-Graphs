import pickle
import vqa_pytorch.vqa.datasets as datasets
import argparse
import tqdm
import numpy as np
import yaml
import torch

# getting h5py
#q_file = h5py.File('qa_from_qid.h5','w')

# getting loader
parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path_opt', default='vqa_pytorch/options/vqa/mutan_att_trainval.yaml', type=str, 
    help='path to a yaml options file')

args = parser.parse_args()
with open(args.path_opt, 'r') as handle:
    options = yaml.load(handle)

LEN_VQA = 334554
trainset = datasets.factory_VQA('trainval', opt_vgenome=options['vgenome'], opt=options['vqa'], opt_coco=options['coco'])

# run through all to make file
print('load & convert files...')
tot = 660692
tenth = 16517
for file in range(23,40):
    print(file,'of 40')
    question_hash = {}
    tr = tqdm.tqdm(range(tenth*file, tenth*(file+1)), total = tenth)
    if file == 39:
        tr = tqdm.tqdm(range(tenth*file, 660692), total = tenth)
    for l in tr:
        item = trainset.__getitem__(l+LEN_VQA)
        question_hash[item['question_id']] = (item['question'].numpy(), item['answer'])

    with open('qa_from_qid'+str(file)+'.pickle', 'wb') as f:
        pickle.dump(question_hash, f)

