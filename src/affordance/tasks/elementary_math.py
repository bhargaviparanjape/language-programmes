from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'elementary_math_qa', cache_dir=cache_dir)
inputs = d['validation']['inputs'][:500]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets'][:500]
labels = [l[0] for l in labels]
task_description = "Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer."

io_pairs = [
("""What is the result of the following arithmetic operations?:divide(multiply(12, multiply(1.6, 1000)), 48)
 choice:200 m
 choice:600 m
 choice:300 m
 choice:100 m
 choice:400 m""",
"400 m"),
("""What is the result of the following arithmetic operations?:multiply(divide(divide(400, 4), 8), 7)
 choice:87.1
 choice:87.5
 choice:87.2
 choice:87.0
 choice:87.4""",
"87.5"),
("""What is the answer to the following math word problem?:a sporting goods store sold 64 frisbees in one week , some for $ 3 and the rest for $ 4 each . if receipts from frisbee sales for the week totaled $ 196 what is the fewest number of $ 4 frisbees that could have been sold ?
 choice:2
 choice:8
 choice:24
 choice:12
 choice:4""",
"4"),
("""What is the answer to the following math word problem, with the given hint?:what decimal fraction is 20 ml of a litre ?
divide 20 by 1000,
 choice:. 05
 choice:none of these
 choice:. 2
 choice:. 02
 choice:0.002""",
". 02"),
("""What is the answer to the following math word problem, with the given hint?:united telephone charges a base rate of $ 6.00 for service , plus an additional charge of $ 0.25 per minute . atlantic call charges a base rate of $ 12.00 for service , plus an additional charge of $ 0.20 per minute . for what number of minutes would the bills for each telephone company be the same ?
divide(subtract(12.00, 6.00), subtract(0.25, 0.20))
 choice:120 minutes
 choice:20 minutes
 choice:140 minutes
 choice:110 minutes
 choice:160 minutes""",
"120 minutes"),
]

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

def elementary_math_qa():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""What is the result of the following arithmetic operations?:divide(multiply(12, multiply(1.6, 1000)), 48)
 choice:200 m
 choice:600 m
 choice:300 m
 choice:100 m
 choice:400 m
A:
400 m
----
What is the result of the following arithmetic operations?:multiply(divide(divide(400, 4), 8), 7)
 choice:87.1
 choice:87.5
 choice:87.2
 choice:87.0
 choice:87.4
A:
87.5
----
What is the answer to the following math word problem?:a sporting goods store sold 64 frisbees in one week , some for $ 3 and the rest for $ 4 each . if receipts from frisbee sales for the week totaled $ 196 what is the fewest number of $ 4 frisbees that could have been sold ?
 choice:2
 choice:8
 choice:24
 choice:12
 choice:4
A:
4
----
What is the answer to the following math word problem, with the given hint?:what decimal fraction is 20 ml of a litre ?
divide 20 by 1000,
 choice:. 05
 choice:none of these
 choice:. 2
 choice:. 02
 choice:0.002
A:
. 02
----
What is the answer to the following math word problem, with the given hint?:united telephone charges a base rate of $ 6.00 for service , plus an additional charge of $ 0.25 per minute . atlantic call charges a base rate of $ 12.00 for service , plus an additional charge of $ 0.20 per minute . for what number of minutes would the bills for each telephone company be the same ?
divide(subtract(12.00, 6.00), subtract(0.25, 0.20))
 choice:120 minutes
 choice:20 minutes
 choice:140 minutes
 choice:110 minutes
 choice:160 minutes
A:
120 minutes
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

# elementary_math_qa()

def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer is one of the five choices.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is one of the five choices.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 20
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Auto-CoT Performance:")
    print("Perf Array", perf_array)
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

auto_cot(temperature=0.3)