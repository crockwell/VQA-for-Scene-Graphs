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
train_loader = trainset.data_loader(batch_size=256,num_workers=4,shuffle=False) 
#warning - this also loads VQA -- HOPEFULLY its okay to have these here too.

length = len(train_loader)
ct = 0
for i, sample in enumerate(train_loader):
    if i * 100 > length * ct:
        print(ct, "% done")
        ct += 1
    

# run through all to make file
print('load & convert files...')
tot = 660692
question_hash = {}
tr = tqdm.tqdm(range(tot), total = tot)
for l in tr:
    item = trainset.__getitem__(l+LEN_VQA)
    question_hash[item['question_id']] = (item['question'].numpy(), item['answer'])

with open('qa_from_qid.pickle', 'wb') as f:
    pickle.dump(question_hash, f)

