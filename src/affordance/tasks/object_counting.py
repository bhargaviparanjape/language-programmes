from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'object_counting', cache_dir=cache_dir)
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

def object_counting():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Q: I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano. How many musical instruments do I have?
A:
thirteen
----
Q: I have a mouse, a goat, and two fish. How many animals do I have?
A:
four
----
Q: I have a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. How many fruits do I have?
A:
ten
----
Q: I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone. How many musical instruments do I have?
A:
eight
----
Q: I have three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin. How many musical instruments do I have?
A:
five
----
Q: I have five stoves, a fridge, a lamp, a microwave, a table, and an oven. How many objects do I have?
A:
ten
----
Q: I have a cabbage, a stalk of celery, a carrot, a lettuce head, a yam, three heads of broccoli, three potatoes, and two cauliflowers. How many vegetables do I have?
A:
thirteen
----
Q: I have a donkey, a cat, two bears, a pig, and a goat. How many animals do I have?
A:
six
----
Q: I have a frog, a duck, a bear, two donkeys, a fish, a goat, and a rabbit. How many animals do I have?
A:
eight
----
Q: I have a duck, a donkey, a cow, a bear, a dog, and a mouse. How many animals do I have?
A:
six
----
%s""" % x for x in chunk]
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


object_counting()