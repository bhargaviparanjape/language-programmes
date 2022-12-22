import argparse
import ast
import json
import pdb
import re
import time
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import urllib.request
from collections import Counter

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks,
                            few_shot_retrieval_prompt, few_shot_code_prompt, 
                            few_shot_arithmetic_prompt, few_shot_string_prompt)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta

d = datasets.load_dataset('bigbench', 'temporal_sequences', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = """Temporal sequences: Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity."""

io_pairs = [("""Today, Elizabeth went to the construction site. Between what times could they have gone?
We know that: 
Elizabeth woke up at 5am.
Lisa saw Elizabeth buying a phone at the electronics store from 5am to 10am.
Mary saw Elizabeth waiting at the train station from 10am to 11am.
Hannah saw Elizabeth waiting at the airport from 1pm to 3pm.
William saw Elizabeth reading at the library from 3pm to 9pm.
The construction site was closed after 9pm.
Between what times could Elizabeth have gone to the construction site?
  choice: 11am to 1pm
  choice: 5am to 10am
  choice: 10am to 11am
  choice: 3pm to 9pm
Possible times:""",
"11am to 1pm"),
("""Today, Andrew went to the art show. Between what times could they have gone?
We know that: 
Andrew woke up at 6am.
Kimberly saw Andrew waiting at the airport from 10am to 12pm.
Anthony saw Andrew attending class at the school from 12pm to 7pm.
John saw Andrew working out at the gym from 7pm to 8pm.
The art show was closed after 8pm.
Between what times could Andrew have gone to the art show?
  choice: 12pm to 7pm
  choice: 7pm to 8pm
  choice: 10am to 12pm
  choice: 6am to 10am
Possible times:""",
"6am to 10am"),
("""Today, Susan went to the orchestra hall. Between what times could they have gone?
We know that: 
Susan woke up at 5am.
Thomas saw Susan buying cookies at a bakery from 5am to 9am.
Ashley saw Susan buying a phone at the electronics store from 9am to 10am.
Betty saw Susan driving to the water park from 10am to 12pm.
Linda saw Susan watching a movie at the theater from 1pm to 3pm.
Leslie saw Susan walking towards the Statue of Liberty from 3pm to 8pm.
The orchestra hall was closed after 8pm.
Between what times could Susan have gone to the orchestra hall?
  choice: 12pm to 1pm
  choice: 5am to 9am
  choice: 10am to 12pm
  choice: 3pm to 8pm
Possible times:""",
"12pm to 1pm"),
("""Today, Jennifer went to the art show. Between what times could they have gone?
We know that: 
Jennifer woke up at 5am.
Ashley saw Jennifer sitting on a rooftop from 7am to 3pm.
David saw Jennifer buying a bike at the bike shop from 3pm to 5pm.
Steven saw Jennifer playing tennis at the tennis court from 5pm to 6pm.
Susan saw Jennifer waiting at the airport from 6pm to 8pm.
Anthony saw Jennifer stretching at a yoga studio from 8pm to 10pm.
The art show was closed after 10pm.
Between what times could Jennifer have gone to the art show?
  choice: 5am to 7am
  choice: 6pm to 8pm
  choice: 3pm to 5pm
  choice: 5pm to 6pm
Possible times:""",
"5am to 7am")]

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

def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = ["""Today, Elizabeth went to the construction site. Between what times could they have gone?
We know that: 
Elizabeth woke up at 5am.
Lisa saw Elizabeth buying a phone at the electronics store from 5am to 10am.
Mary saw Elizabeth waiting at the train station from 10am to 11am.
Hannah saw Elizabeth waiting at the airport from 1pm to 3pm.
William saw Elizabeth reading at the library from 3pm to 9pm.
The construction site was closed after 9pm.
Between what times could Elizabeth have gone to the construction site?
  choice: 11am to 1pm
  choice: 5am to 10am
  choice: 10am to 11am
  choice: 3pm to 9pm
Possible times:
11am to 1pm
----
Today, Andrew went to the art show. Between what times could they have gone?
We know that: 
Andrew woke up at 6am.
Kimberly saw Andrew waiting at the airport from 10am to 12pm.
Anthony saw Andrew attending class at the school from 12pm to 7pm.
John saw Andrew working out at the gym from 7pm to 8pm.
The art show was closed after 8pm.
Between what times could Andrew have gone to the art show?
  choice: 12pm to 7pm
  choice: 7pm to 8pm
  choice: 10am to 12pm
  choice: 6am to 10am
Possible times:
6am to 10am
----
Today, Susan went to the orchestra hall. Between what times could they have gone?
We know that: 
Susan woke up at 5am.
Thomas saw Susan buying cookies at a bakery from 5am to 9am.
Ashley saw Susan buying a phone at the electronics store from 9am to 10am.
Betty saw Susan driving to the water park from 10am to 12pm.
Linda saw Susan watching a movie at the theater from 1pm to 3pm.
Leslie saw Susan walking towards the Statue of Liberty from 3pm to 8pm.
The orchestra hall was closed after 8pm.
Between what times could Susan have gone to the orchestra hall?
  choice: 12pm to 1pm
  choice: 5am to 9am
  choice: 10am to 12pm
  choice: 3pm to 8pm
Possible times:
12pm to 1pm
----
Today, Jennifer went to the art show. Between what times could they have gone?
We know that: 
Jennifer woke up at 5am.
Ashley saw Jennifer sitting on a rooftop from 7am to 3pm.
David saw Jennifer buying a bike at the bike shop from 3pm to 5pm.
Steven saw Jennifer playing tennis at the tennis court from 5pm to 6pm.
Susan saw Jennifer waiting at the airport from 6pm to 8pm.
Anthony saw Jennifer stretching at a yoga studio from 8pm to 10pm.
The art show was closed after 10pm.
Between what times could Jennifer have gone to the art show?
  choice: 5am to 7am
  choice: 6pm to 8pm
  choice: 3pm to 5pm
  choice: 5pm to 6pm
Possible times:
5am to 7am
----
Today, James went to the physics classroom. Between what times could they have gone?
We know that: 
James woke up at 8am.
Richard saw James working out at the gym from 10am to 11am.
Mark saw James sitting on a rooftop from 11am to 1pm.
Lisa saw James fixing their computer at the electronic store from 1pm to 3pm.
Mary saw James playing tennis at the tennis court from 3pm to 7pm.
The physics classroom was closed after 7pm.
Between what times could James have gone to the physics classroom?
  choice: 11am to 1pm
  choice: 8am to 10am
  choice: 10am to 11am
  choice: 3pm to 7pm
Possible times:
8am to 10am
----
Today, Leslie went to the movies. Between what times could they have gone?
We know that: 
Leslie woke up at 9am.
Betty saw Leslie working out at the gym from 9am to 10am.
Linda saw Leslie taking photos near the Leaning Tower of Pisa from 10am to 11am.
William saw Leslie walking towards the Statue of Liberty from 11am to 12pm.
The movies was closed after 9pm.
Between what times could Leslie have gone to the movies?
  choice: 11am to 12pm
  choice: 9am to 10am
  choice: 10am to 11am
  choice: 12pm to 9pm
Possible times:
12pm to 9pm
----
Today, Kimberly went to the construction site. Between what times could they have gone?
We know that: 
Kimberly woke up at 8am.
Richard saw Kimberly taking photos near the Eiffel Tower from 8am to 9am.
Sarah saw Kimberly watching a movie at the theater from 9am to 12pm.
Tiffany saw Kimberly reading at the library from 1pm to 5pm.
The construction site was closed after 5pm.
Between what times could Kimberly have gone to the construction site?
  choice: 1pm to 5pm
  choice: 9am to 12pm
  choice: 12pm to 1pm
  choice: 8am to 9am
Possible times:
12pm to 1pm
----
Today, Jennifer went to the park. Between what times could they have gone?
We know that: 
Jennifer woke up at 10am.
Susan saw Jennifer walking in the garden from 10am to 5pm.
Elizabeth saw Jennifer buying lunch at the deli from 5pm to 6pm.
Sean saw Jennifer playing tennis at the tennis court from 7pm to 9pm.
The park was closed after 9pm.
Between what times could Jennifer have gone to the park?
  choice: 5pm to 6pm
  choice: 7pm to 9pm
  choice: 10am to 5pm
  choice: 6pm to 7pm
Possible times:
6pm to 7pm
----
Today, William went to the soccer field. Between what times could they have gone?
We know that: 
William woke up at 1pm.
Tiffany saw William taking photos near the Leaning Tower of Pisa from 1pm to 2pm.
David saw William waiting at the airport from 2pm to 6pm.
Thomas saw William reading at the library from 6pm to 7pm.
The soccer field was closed after 8pm.
Between what times could William have gone to the soccer field?
  choice: 2pm to 6pm
  choice: 7pm to 8pm
  choice: 1pm to 2pm
  choice: 6pm to 7pm
Possible times:
7pm to 8pm
----
%s
""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used. 
# Description: 
# Input: 
# ```
# if x < 5:
# 	pass
# ```
# What error does this program surface?
# Q1: [execute] Execute the following python code snippet.
# if x < 5:
# 	pass
# #1:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'x' is not defined
# Q2: What is the final error message?
# #2: NameError: name 'x' is not defined
# Q3: [EOC]
# NameError: name 'x' is not defined
# ----
# Desciption:
# Input:
# ```
# x = set([1, 1, 2, 3])
# ```
# What is the value of x after this program executes?
# Q1: [code edit] Edit the following code to get the value of x
# x = set([1, 1, 2, 3])
# #1:
# x = set([1, 1, 2, 3])
# print(x)
# Q2: [execute] Execute the following python code snippet.
# x = set([1, 1, 2, 3])
# print(x)
# #2: {1, 2, 3}
# Q3: [EOC]
# {1, 2, 3}
# ----
# Description:
# Input:
# Python code:
# try:
#     n = int(input())
#     m = int(input())
#     integer_sum = int(n) + int(m)
#     print(integer_sum)
# except:
#     print('error')

#   choice: prints number between 5 and 6
#   choice: try input and except error
#   choice: inputs the string 'try'
#   choice: prints sum of two input numbers only if they are integers otherwise raises error
# Q1: [code generate] prints number between 5 and 6
# #1:
# import random

# print(random.uniform(5,6))
# Q2: [code generate] try input and except error
# #2
# try:
#     #open file
#     file = open(file_name, "r")
#     #read file
#     data = file.read()
#     #close file
#     file.close()
#     #split data
#     data = data.split("\n")
#     #remove empty string
# Q3: [code generate] inputs the string 'try'
# #3: print('try')
# Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
# #4:
# #!/usr/bin/python

# a=raw_input("enter first number: ")
# b=raw_input("enter second number: ")
# try:
#     sum=int(a)+int(b)
#     print "sum is: ",sum
# except:
#     print "enter integer values only"
# Q5: Which of the generated code snippets are most like the original one?
# #5: prints sum of two input numbers only if they are integers otherwise raises error
# Q6: [EOC]
# ----
# Desciption: %s
# Input: %s
# Q1: """

few_shot_cot_prompt = few_shot_arithmetic_prompt

auto_cot_corrected_prompt = """TASK: 
Today, Elizabeth went to the construction site. Between what times could they have gone?
We know that: 
Elizabeth woke up at 5am.
Lisa saw Elizabeth buying a phone at the electronics store from 5am to 10am.
Mary saw Elizabeth waiting at the train station from 10am to 11am.
Hannah saw Elizabeth waiting at the airport from 1pm to 3pm.
William saw Elizabeth reading at the library from 3pm to 9pm.
The construction site was closed after 9pm.
Between what times could Elizabeth have gone to the construction site?
  choice: 11am to 1pm
  choice: 5am to 10am
  choice: 10am to 11am
  choice: 3pm to 9pm
Possible times:
A: Let's think step-by-step.

First, we need to find out when Lisa saw Elizabeth. We know that Elizabeth woke up at 5am and Lisa saw Elizabeth from 5am to 10am, so we can eliminate any times before 10am.
Next, we need to find out when Mary saw Elizabeth. We know that Mary saw Elizabeth from 10am to 11am, so we can eliminate the option 10am to 11am.
We know that Hannah saw Elizabeth waiting at the airport from 1pm to 3pm and William saw Elizabeth reading at the library from 3pm to 9pm. So we can eliminate any times after 1pm.
That means the only possible time Elizabeth could have gone to the construction site is from 11am to 1pm.

The best answer choice is 11am to 1pm.
----
TASK: 
Today, Andrew went to the art show. Between what times could they have gone?
We know that: 
Andrew woke up at 6am.
Kimberly saw Andrew waiting at the airport from 10am to 12pm.
Anthony saw Andrew attending class at the school from 12pm to 7pm.
John saw Andrew working out at the gym from 7pm to 8pm.
The art show was closed after 8pm.
Between what times could Andrew have gone to the art show?
  choice: 12pm to 7pm
  choice: 7pm to 8pm
  choice: 10am to 12pm
  choice: 6am to 10am
Possible times:
A: Let's think step-by-step.

The art show was closed after 8pm. So Andrew can only go to the art show before 8pm. Andrew woke up at 6am.
Kimberly saw Andrew waiting at the airport from 10am to 12pm. Anthony saw Andrew attending class at the school from 12pm to 7pm. John saw Andrew working out at the gym from 7pm to 8pm.
So, we know that Andrew was at the airport from 10am to 12pm, at school from 12pm to 7pm, and at the gym from 7pm to 8pm. 
That means that the only time he could have gone to the art show was between the time he woke up and 10am.

The best answer choice is 6am to 10am.
----
TASK: 
Today, Susan went to the orchestra hall. Between what times could they have gone?
We know that: 
Susan woke up at 5am.
Thomas saw Susan buying cookies at a bakery from 5am to 9am.
Ashley saw Susan buying a phone at the electronics store from 9am to 10am.
Betty saw Susan driving to the water park from 10am to 12pm.
Linda saw Susan watching a movie at the theater from 1pm to 3pm.
Leslie saw Susan walking towards the Statue of Liberty from 3pm to 8pm.
The orchestra hall was closed after 8pm.
Between what times could Susan have gone to the orchestra hall?
  choice: 12pm to 1pm
  choice: 5am to 9am
  choice: 10am to 12pm
  choice: 3pm to 8pm
Possible times:
A: Let's think step-by-step.

We need to find out when Susan went to the orchestra hall. We know that the hall was closed after 8pm, so Susan must have gone before then. We also know that Leslie saw Susan walking towards the Statue of Liberty from 3pm to 8pm. This means that Susan must have gone to the orchestra hall before 3pm.
Now, we need to find out when Susan could have left the theater. We know that Linda saw Susan watching a movie at the theater from 1pm to 3pm. This means that Susan must have gone to the orchestra hall before 1pm.
Next, we need to find out when Susan could have gone to the water park. We know that Betty saw Susan driving to the water park from 10am to 12pm. This means that Susan must have gone to the water park from 10am to 12pm.
We know that Ashley saw Susan buying a phone at the electronics store from 9am to 10am. This means that Susan must have gone to the electronics store from 9am to 10am.
Putting all of this together, we can see that the earliest Susan could have gone to the orchestra hall is 12pm and the latest she could have gone is 1pm.

The best answer choice is 12pm to 1pm.
----
TASK: 
Today, Jennifer went to the art show. Between what times could they have gone?
We know that: 
Jennifer woke up at 5am.
Ashley saw Jennifer sitting on a rooftop from 7am to 3pm.
David saw Jennifer buying a bike at the bike shop from 3pm to 5pm.
Steven saw Jennifer playing tennis at the tennis court from 5pm to 6pm.
Susan saw Jennifer waiting at the airport from 6pm to 8pm.
Anthony saw Jennifer stretching at a yoga studio from 8pm to 10pm.
The art show was closed after 10pm.
Between what times could Jennifer have gone to the art show?
  choice: 5am to 7am
  choice: 6pm to 8pm
  choice: 3pm to 5pm
  choice: 5pm to 6pm
Possible times:
A: Let's think step-by-step.

Jennifer woke up at 5am. Since the art show was closed after 10pm, she could only go to the art show before 10pm.
Ashley saw Jennifer sitting on a rooftop from 7am to 3pm. David saw Jennifer buying a bike at the bike shop from 3pm to 5pm. Steven saw Jennifer playing tennis at the tennis court from 5pm to 6pm. Susan saw Jennifer waiting at the airport from 6pm to 8pm. Anthony saw Jennifer stretching at a yoga studio from 8pm to 10pm.
Jennifer was busy from 7am to 10pm. The only time she could have gone to the art show was before 7am.
So, Jennifer could have gone to the art show any time between 5am and 7am.

The best answer choice is 5am to 7am.
----
"""
def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    global auto_cot_corrected_prompt
    auto_cot_prompt = ""
    description = "TASK: "
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=0.2, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Anachronisms\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)
    

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=0.2, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nPossible times:", "") for ex in x]
            answers.extend(predict(x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character  = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers = predict("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x)
            pdb.set_trace()
            affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x))
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):

    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_description, io_pairs, N=6)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_pot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.

Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input:  In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:  A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_disntace = 600
original_speed = total_disntace / duration
reduced_speed = total_disntace / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2:
1.0
Q3: [compare] Which of the options among A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours is most similar to the answer? 
#3: A
Q3: [EOQ]
Ans: A
----
Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: M men agree to purchase a gift for Rs. D. If 3 men drop out how much more will each have to contribute towards the purchase of the gift?  A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans=simplify(cost_after_dropout - cost_before_dropout)
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 3*D/(M*(M - 3))
Q3: [compare] Which of the options among A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these is most similar to the answer? 
#3: D
Q4: [EOQ]
Ans: D
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [EOQ]
Ans:18
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 20
Q3: [EOQ]
Ans: 20
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now. 
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 22
Q3: [EOQ]
Ans: 22
----
Description: Solve the following problems on financial data in the form of text and tables, writing out intermediate calculations as python code. Store your result as a variable named 'ans'.
Input: american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $75.0 million and network location intangibles of approximately $72.7 million. the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years. For acquired customer-related and network location intangibles, what is the expected annual amortization expenses, in millions?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 7.385
Q3: [EOQ]
Ans: 7.385
----
Descripton: %s
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_pot_prompt
    task_name = "Temporal Sequences"
    task_description = """(Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity."""

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_pot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    task_description = "Temporal sequences: Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("""\nPossible times:""", "") for ex in x]
            answers.extend(predict(task_description, x))
            time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3, model_name = "text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_pot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    task_description = "Select the option that best replaces '()' in each text input given the chocies presented, by solving the arithmetic word problem in the text input. Solving the problem will require reasoning about units of measurement. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
            new_answers = []
            for ans in answers:
                try:
                    parsed_program  = parse_program("Input: dummy\nQ1:" + ans)
                    code_snippet = parsed_program.node_list[0].command_output
                    result = safe_execute(code_snippet)
                except:
                    result = None
                if result:
                    new_answers.append(result)
                else:
                    new_answers.append(ans[re.search('Ans:', ans).span(0)[0]:] if "Ans:" in ans else ans)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    
    global few_shot_cot_prompt
    task_name = "(Temporal sequences)"
    task_description = """(Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity."""

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_pot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    interpreter = TopDownVisitorBeta()

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nPossible times:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action='store_true')

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[:args.num_dev_examples]
    labels = labels[:args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(args.temperature, args.model_name, predict=True, use_corrected=False, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)