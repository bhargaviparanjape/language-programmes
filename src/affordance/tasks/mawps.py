from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0*correct)/count

data = datasets.load_dataset('omarxadel/MaWPS-ar', 'train', cache_dir=cache_dir)
inputs = [list(d.values())[0] for d in data['validation']]
labels = []
ans_list = []
for d in data['validation']:
    try:
        ans = eval(list(d.values())[1].split("=")[-1].strip())
        labels.append(list(d.values())[1].split("=")[-1].strip())
        if isinstance(ans, int):
            ans_list.append(ans)
        elif (ans).is_integer():
            ans_list.append(int(ans))
        else:
            ans_list.append(float("%.2f" % ans))
        
    except:
        ans = eval(list(d.values())[1].split("=")[0].strip())
        labels.append(list(d.values())[1].split("=")[-1].strip())
        if isinstance(ans, int):
            ans_list.append(ans)
        elif (ans).is_integer():
            ans_list.append(int(ans))
        else:
            ans_list.append(float("%.2f" % ans))
    
def mawps():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Amanda wants to diverge 608 Books among 28 friends. How many would each friend acquire?
608 / 28
----
Connie had 141 plum. Carrie took 139 from him. Now How many plum Connie have left?
141 - 139
----
Michael had 280 watermelon. Jasmine gripped some watermelon. Now Michael has 3  watermelon. How many did Jasmine grippeds?
280 - 3
----
Katherine acquire 9 bags of nectarine . how many nectarine in each bag? If total 89 nectarine Katherine  acquire.
89 / 9
----
 A company invited 24 people to a luncheon, but 10 of them didn't show up. If the tables they had held 7 people each, how many tables do they need? 
((24.0-10.0)/7.0)
----
%s
"""% x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        evaluated_predictions = []
        for p in preds:
            try:
                evaluated_predictions.append(eval(p))
            except:
                evaluated_predictions.append(eval(re.search("[0-9,.]+", p).group(0).replace(",", "")))
        evaluated_labels = [] #[eval(l) for l in labels]
        for l in labels:
            try:
                evaluated_labels.append(eval(l))
            except:
                pdb.set_trace()
        # sometimes the model actually predicts the final answer
        pdb.set_trace()
        perf_array.append(np.mean([1 if p==l else 0 for p,l in zip(evaluated_predictions, evaluated_labels)]))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

mawps()