from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns



d = datasets.load_dataset('bigbench', 'temporal_sequences', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

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

def temporal_sequences():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
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
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


temporal_sequences()

few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used. 
Description: 
Input: 
```
if x < 5:
	pass
```
What error does this program surface?
Q1: [execute] Execute the following python code snippet.
if x < 5:
	pass
#1:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Q2: What is the final error message?
#2: NameError: name 'x' is not defined
Q3: [EOC]
NameError: name 'x' is not defined
----
Desciption:
Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to get the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOC]
{1, 2, 3}
----
Description:
Input:
Python code:
try:
    n = int(input())
    m = int(input())
    integer_sum = int(n) + int(m)
    print(integer_sum)
except:
    print('error')

  choice: prints number between 5 and 6
  choice: try input and except error
  choice: inputs the string 'try'
  choice: prints sum of two input numbers only if they are integers otherwise raises error
Q1: [code generate] prints number between 5 and 6
#1:
import random

print(random.uniform(5,6))
Q2: [code generate] try input and except error
#2
try:
    #open file
    file = open(file_name, "r")
    #read file
    data = file.read()
    #close file
    file.close()
    #split data
    data = data.split("\n")
    #remove empty string
Q3: [code generate] inputs the string 'try'
#3: print('try')
Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
#4:
#!/usr/bin/python

a=raw_input("enter first number: ")
b=raw_input("enter second number: ")
try:
    sum=int(a)+int(b)
    print "sum is: ",sum
except:
    print "enter integer values only"
Q5: Which of the generated code snippets are most like the original one?
#5: prints sum of two input numbers only if they are integers otherwise raises error
Q6: [EOC]
----
Desciption: %s
Input: %s
Q1: """


def auto_cot():
    auto_cot_prompt = ""
    description = "TASK: "
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
    print(auto_cot_prompt)
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

# auto_cot()


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character  = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
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

# affordance()



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
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# auto_decomp(10, 0.3)
# affordance(temperature=0.3)
dynamic_few_shot_cot(temperature=0.3, strategy="random")
# few_shot_cot()
# few_shot(N=5, temperature=0.3)
# auto_cot()
