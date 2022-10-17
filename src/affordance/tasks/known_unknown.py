from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

# Startegies for unknown questions
# predictions about future times from present day
# Things that ARE HARD TO count or find an exact cnt for
# Things that ARE HARD TO measure or find an exact measure for
# Things that ARE HARD TO DATE or find an exact date for
# Facts that are not in public record because personal details are often unknown


# load data
d = datasets.load_dataset('bigbench', 'known_unknowns', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
labels = d['train']['targets']  + d['validation']['targets']
labels = [l[0] for l in labels]

# Instructions that Flipped learning comes up with (not known)

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

def known_unknown():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Q: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
A:
Ronnie Ronalde
----
Q: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
A:
Florida, Missouri
----
Q: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
A:
Bowler hat
----
Q: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
A:
Unknown
----
Q: How many legs do ants have?
  choice: Unknown
  choice: 6
A:
6
----
Q: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours
A:
Between 7 and 9 hours
----
Q: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
A:
Unknown
----
Q: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings
A:
Unknown
----
Q: What will be the name of the great-great-granddaughter of Diana Ross?
  choice: Unknown
  choice: Hazel Ross
A:
Unknown
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
    print("Std. Dev", np.mean(perf_array))


known_unknown()

def automatic_decomposition():
    decomp_prompt = "The task is to answer the question or figure out if the answer is unknown. I want to break this task into individual steps."
    io_pairs = """Input: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
Output: Ronnie Ronalde
----
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Output: Florida, Missouri
----
Input: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
Output: Bowler hat
----
Input: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
Output: Unknown
----
Input: How many legs do ants have?
  choice: Unknown
  choice: 6
Output: 6
----
Input: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours
Output: Between 7 and 9 hours
----
Input: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
Output: Unknown
----
Input: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings
Output: Unknown
----
Input: What will be the name of the great-great-granddaughter of Diana Ross?
  choice: Unknown
  choice: Hazel Ross
Output: Unknown"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_list_reversal_fn(decomposition, batch_size=10):
        decomposition = '1.'+ decomposition
        last_n = int(re.findall(r'(\d+)\.', decomposition)[-1])
    #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = ['''The task is to answer the question or figure out if the answer is unknown. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise. Either provide the answer or say "Unknown".''' % (decomposition, x) for x in chunk]
                out.extend(gpt3(prompts))
            return out
        return decomposition_fn


    labs, subset = get_subset(inputs, labels=labels, n=len(inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print('Decomposition', z)
        fn = get_list_reversal_fn(decomposition, batch_size=20)
        this_preds = fn(subset)
    #     pp = np.array([1 if 'contains an anachronism' in x.lower() else 0 for x in this_preds])
        pdb.set_trace()
        # pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        pp = [pred.strip().lower().endswith('unknown') or pred.strip().lower().endswith('known') for pred in this_preds]
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
    print(accs)
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.mean(accs))

automatic_decomposition()