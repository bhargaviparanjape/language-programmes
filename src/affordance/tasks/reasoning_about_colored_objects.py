from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'reasoning_about_colored_objects', cache_dir=cache_dir)
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

def reasoning_about_colored_objects():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?
A:
pink
----
Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?
A:
yes
----
Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?
A:
fuchsia
----
Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?
A:
3
----
Q: On the table, there is a red dog leash, a brown teddy bear, a silver pencil, and a teal paperclip. What color is the paperclip?
A:
teal
----
Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?
A:
teal
----
Q: On the floor, there is one yellow textbook, one teal necklace, three yellow puzzles, three teal puzzles, two purple textbooks, two magenta pencils, one yellow pencil, two yellow necklaces, and one purple necklace. If I remove all the magenta things from the floor, how many puzzles remain on it?
A:
6
----
Q: On the desk, you see several things arranged in a row: a blue textbook, a red jug, a green mug, and a grey keychain. What is the color of the thing directly to the left of the grey thing?
A:
green
----
Q: On the nightstand, you see one gold plate, three orange puzzles, three turquoise pairs of sunglasses, three orange pairs of sunglasses, two gold pairs of sunglasses, two gold dog leashes, two orange dog leashes, one silver dog leash, one turquoise dog leash, one orange plate, and one turquoise plate. If I remove all the plates from the nightstand, how many gold objects remain on it?
A:
4
----
Q: On the floor, there are two silver dog leashes, two red dog leashes, one silver bracelet, one silver keychain, three pink dog leashes, three red bracelets, one red cat toy, and one red keychain. If I remove all the silver objects from the floor, how many cat toys remain on it?
A:
1
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


reasoning_about_colored_objects()