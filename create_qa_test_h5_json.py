import h5py
f = h5py.File('test.h5','r')
f2 = h5py.File('qa_challenge_chris.h5', 'w')
#for i in f:
#    print(i)
    
import json
with open('question_answers.json') as data_file:
    data = json.load(data_file)
    
    
questions = []
answers = []
indices = []
    
idxs = [10, 16, 87]
for idx in idxs:
    for i in data:
        if i['id'] == idx:
            for j in i['qas']:
                questions.append(j['question'])
                answers.append(j['answer'])
                indices.append(idx)

questions = [n.encode("ascii", "ignore") for n in questions]
answers = [n.encode("ascii", "ignore") for n in answers]
                
f2['question'] = questions
f2['answer'] = answers
f2['image_id'] = indices
f2.close()