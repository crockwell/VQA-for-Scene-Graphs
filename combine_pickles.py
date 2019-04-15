import pickle

master = {}

for file in range(20):
    print(file)
    with open('qa_from_qid'+str(file)+'.pickle', 'rb') as f:
        new = pickle.load(f)
        master = {**master, **new}

with open('qa_from_qid_master.pickle', 'wb') as f:
    pickle.dump(master, f)
