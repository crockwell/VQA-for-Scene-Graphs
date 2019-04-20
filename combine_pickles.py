import pickle

master = {}

for file in range(37):
    print(file)
    with open('idx_from_qid'+str(file)+'.pickle', 'rb') as f:
        new = pickle.load(f)
        master = {**master, **new}

with open('idx_from_qid_master.pickle', 'wb') as f:
    pickle.dump(master, f)
