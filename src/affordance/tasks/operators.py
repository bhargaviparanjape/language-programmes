from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'operators', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']
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

def operators():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Given the definition of the op operator, compute the result.
i op j subtracts j from i.
7 op 3 =
4
----
Given the definition of the op operator, compute the result.
i op j subtracts j from i.
176 op 23 =
153
----
Given the definition of the op operator, compute the result.
i op j keeps the j first digits of i.
125690 op 3 =
125
----
Given the definition of the op operator, compute the result.
i op j suppresses the j first digits of i.
125690 op 2 =
5690
----
Given the definition of the op operator, compute the result.
op n1 n2 ... nn selects the smallest from the n listed numbers.
op 16 0 10 6 =
0
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


operators()