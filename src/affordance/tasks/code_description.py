from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'code_line_description', cache_dir=cache_dir)
inputs =  d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels =  d['validation']['targets']
labels = [l[0] for l in labels]

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def code_description():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-001",  max_length=1000, quote='---', n=1)
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


code_description()