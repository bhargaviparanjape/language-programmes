from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'unit_interpretation', cache_dir=cache_dir)
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

def unit_interpretation():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Please select the option that best replaces '()' in each text input given the chocies presented.

Q: 27 eggs divided by 9 eggs per hen per day is ().
  choice: 243 hen-days
  choice: 3 hen-days
  choice: 3 days
  choice: 243 eggs
  choice: 3 eggs
A:
3 hen-days
----
Please select the option that best replaces '()' in each text input given the chocies presented.

Q: A newsstand sells 5 newspapers every 2 weeks. It will sell () newspapers in 60 weeks.
  choice: 120
  choice: 300
  choice: 150
  choice: 12
  choice: 24
A:
150
----
Please select the option that best replaces '()' in each text input given the chocies presented.

Q: A bell rings one time every 5 hours. It will ring 20 times in () hours.
  choice: 20
  choice: 50
  choice: 25
  choice: 100
  choice: 4
A:
100
----
Please select the option that best replaces '()' in each text input given the chocies presented.

Q: It takes 2 sausages and 5 buns to make 40 hotdogs. It will take 10 sausages and () buns to make 200 hotdogs.
  choice: 25
  choice: 50
  choice: 4
  choice: 10
  choice: 5
A:
25
----
Please select the option that best replaces '()' in each text input given the chocies presented.

Q: It takes 3 men and 3 hours to cut down 9 trees. It will take 2 men and 4 hours to cut down () trees.
  choice: 18
  choice: 8
  choice: 3
  choice: 12
  choice: 6
A:
8
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

unit_interpretation()