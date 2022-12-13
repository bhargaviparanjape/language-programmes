from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match, get_answer

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns
import time
from collections import Counter

d = datasets.load_dataset('bigbench', 'code_line_description', cache_dir=cache_dir)
inputs =  d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels =  d['validation']['targets']
labels = [l[0] for l in labels]

io_pairs = [("""Python code:
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
  choice: prints sum of two input numbers only if they are integers otherwise raises error""",
"""prints sum of two input numbers only if they are integers otherwise raises error"""),
("""Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6""",
"""prints a list of numbers greater than 7 from numbers_list"""),
("""Python code:
sum_of_numbers = 0
num = 10
while(num > 0):
    sum_of_numbers += num
    num -= 1
print('The sum is', sum_of_numbers)

  choice: prints the factorial of numbers
  choice: prints the sum of first 10 natural numbers
  choice: prints number from 10 to 0 in ascending order
  choice: prints the numbers from 1 to 10""",
"""prints the sum of first 10 natural numbers"""),
("""Python code:
for i in range(23):
    print(i)

  choice: prints 'hello world' to the terminal
  choice: prints values from 1 to 10
  choice: computes first 10 prime numbers
  choice: prints values from 0 to 22""",
"""prints values from 0 to 22"""),
("""Python code:
def sum_two(a,b):
    return a+b

  choice: returns a and b
  choice: returns the sum of two numbers
  choice: returns 5 and 6
  choice: returns numbers between 5 and 6""", """returns the sum of two numbers""")]

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def code_description(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", temperature=temperature, max_length=1000, quote='---', n=1)
        prompts = ["""Python code:
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

English language description:
prints sum of two input numbers only if they are integers otherwise raises error
----
Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6

English language description:
prints a list of numbers greater than 7 from numbers_list
----
Python code:
sum_of_numbers = 0
num = 10
while(num > 0):
    sum_of_numbers += num
    num -= 1
print('The sum is', sum_of_numbers)

  choice: prints the factorial of numbers
  choice: prints the sum of first 10 natural numbers
  choice: prints number from 10 to 0 in ascending order
  choice: prints the numbers from 1 to 10

English language description:
prints the sum of first 10 natural numbers
----
Python code:
for i in range(23):
    print(i)

  choice: prints 'hello world' to the terminal
  choice: prints values from 1 to 10
  choice: computes first 10 prime numbers
  choice: prints values from 0 to 22

English language description:
prints values from 0 to 22
----
Python code:
def sum_two(a,b):
    return a+b

  choice: returns a and b
  choice: returns the sum of two numbers
  choice: returns 5 and 6
  choice: returns numbers between 5 and 6

English language description:
returns the sum of two numbers
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


few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used. 
Description: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
Input: 
```
if x < 5:
	pass
```
What error does this program surface?
Q1: [code execute] Execute the following python code snippet.
if x < 5:
	pass
#1:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Q2: [get error message] What is the final error message?
#2: NameError: name 'x' is not defined
Q3: [EOQ]
Ans: NameError: name 'x' is not defined
----
Desciption: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
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
Q2: [code execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOQ]
Ans: {1, 2, 3}
----
Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet. 
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
Q5: [compare] Which of the generated code snippets are most like the original one?
#5: prints sum of two input numbers only if they are integers otherwise raises error
Q6: [EOQ]
Ans: prints sum of two input numbers only if they are integers otherwise raises error
----
Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet. 
Input:
Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6
Q1: [code generate] prints lambda
#1:
print("lambda")
Q2: [code generate] returns a filtered list
#2
def filter_list(l):
  'return a new list with the strings filtered out'
  return [x for x in l if not isinstance(x, str)]
Q3: [code generate] prints a list of numbers greater than 7 from numbers_list
#3:
numbers_list = [1,2,3,4,5,6,7,8,9,10]
for number in numbers_list:
    if number > 7:
        print(number)
Q4: [code generate] prints numbers from 2 to 6
#4:
for i in range(2,7):
    print(i)
Q5: Which of the generated code snippets are most like the original one?
#5: prints a list of numbers greater than 7 from numbers_list
Q6: [EOQ]
Ans: prints a list of numbers greater than 7 from numbers_list
----
Desciption: %s
Input: %s
Q1:"""




def few_shot_cot(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
            answers.extend(predict("""(Code Description) Which of the following choices best describes the functionality of the given python code snippet.""", x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    global few_shot_cot_prompt
    task_name = "Code Description"
    task_description = "(Code Description) Which of the following choices best describes the functionality of the given python code snippet."

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

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 5
        batch_size = 10
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for ans in answer_set:
                    new_answer = interpreter.batch_visit(prompts, ans)
                    new_answer = [get_answer(program) for program in new_answer]
                    for enum, pred in enumerate(new_answer):
                        if pred is not None:
                            result_counter[enum].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            time.sleep(60)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))




nl_program(temperature=0.3)
# few_shot_cot(temperature=0.3)
# code_description()