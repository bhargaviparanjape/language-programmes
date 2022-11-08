from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'navigate', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

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

def navigate():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 4 steps forward. Take 5 steps right. Take 7 steps left.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Turn right. Turn around. Take 2 steps. Take 6 steps. Take 4 steps.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Take 7 steps. Turn right. Take 8 steps. Turn around.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Take 6 steps. Take 7 steps. Turn right. Take 3 steps. Turn right. Turn right. Take 4 steps. Take 4 steps.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 2 steps backward. Take 8 steps forward. Take 10 steps forward. Take 7 steps left. Take 6 steps forward. Take 5 steps left. Take 5 steps left.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Take 1 step. Turn around. Take 2 steps. Turn left. Take 1 step. Take 3 steps. Turn left. Take 1 step.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Take 2 steps. Take 3 steps. Take 1 step. Take 6 steps. Turn left. Take 1 step.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 2 steps right. Take 2 steps right.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 7 steps backward. Take 5 steps left. Take 3 steps right.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 3 steps backward. Take 5 steps backward. Take 3 steps backward. Take 9 steps forward. Take 9 steps right.
A:
False
----
%s
""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


navigate()