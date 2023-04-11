import argparse
from collections import Counter
import re

import datasets
import numpy as np
from prompt_library import (
    few_shot_code_prompt,
    llm_similar_tasks,
    random_tasks,
    similar_auto_breakdowns,
    similar_tasks,
)
from sequential_interpreter import TopDownVisitorBeta
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (
    OpenAIModel,
    cache_dir,
    chunks,
    get_answer,
    get_autocot_answer,
    get_few_shot_prompt,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "code_line_description", cache_dir=cache_dir)
inputs = d["validation"]["inputs"] + d["train"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"] + d["train"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "(Code Description) Which of the following choices best describes the functionality of the given python code snippet."

io_pairs = [
    (
        """Python code:
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
        """prints sum of two input numbers only if they are integers otherwise raises error""",
    ),
    (
        """Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6""",
        """prints a list of numbers greater than 7 from numbers_list""",
    ),
    (
        """Python code:
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
        """prints the sum of first 10 natural numbers""",
    ),
    (
        """Python code:
for i in range(23):
    print(i)

  choice: prints 'hello world' to the terminal
  choice: prints values from 1 to 10
  choice: computes first 10 prime numbers
  choice: prints values from 0 to 22""",
        """prints values from 0 to 22""",
    ),
    (
        """Python code:
def sum_two(a,b):
    return a+b

  choice: returns a and b
  choice: returns the sum of two numbers
  choice: returns 5 and 6
  choice: returns numbers between 5 and 6""",
        """returns the sum of two numbers""",
    ),
]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)["input_ids"]))

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """%s\
%s"""
            % (few_shot_prompt, x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def code_description(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=1000, quote="---", n=1
        )
        prompts = [
            """Python code:
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
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
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
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

First, the code tries to convert the inputs n and m into integers. If this is successful, it calculates the sum of the two integers and prints it out. Otherwise, it prints 'error'.
So, if the inputs are integers, the code will print the sum. If the inputs are not integers, the code will print 'error'.

The best answer choice is "prints sum of two input numbers only if they are integers otherwise raises error"
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

The first line creates a list called numbers_list.
The second line calls the filter function, with a lambda function as the first argument. This lambda function checks if a number is greater than 7.
The third line prints the result of the filter function, which is a filtered list.
So the code snippet prints a list of numbers greater than 7 from numbers_list.

The best answer choice is "prints a list of numbers greater than 7 from numbers_list"
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
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
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

The first two lines initialize our variables. sum_of_numbers starts at 0, and num starts at 10.
Next, we have a while loop. This loop will keep running as long as num is greater than 0.
Inside the loop, we add the value of num to sum_of_numbers, and then subtract 1 from num.
Finally, we print out the value of sum_of_numbers.
So what does the code do? It prints the sum of the first 10 natural numbers.

The best answer choice is "prints the sum of first 10 natural numbers".
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
for i in range(23):
    print(i)

  choice: prints 'hello world' to the terminal
  choice: prints values from 1 to 10
  choice: computes first 10 prime numbers
  choice: prints values from 0 to 22
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

The first line creates a for loop that iterates over the numbers 0 through 23.
The second line prints the value of i at each iteration.
The third line ends the for loop.
This code snippet will print values from 0 to 22.

The best answer choice is "prints values from 0 to 22".
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
def sum_two(a,b):
    return a+b

  choice: returns a and b
  choice: returns the sum of two numbers
  choice: returns 5 and 6
  choice: returns numbers between 5 and 6
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

First, the function is defined as sum_two, which takes in two parameters, a and b.
Next, the function returns the sum of a and b.
So in this case, the code snippet returns the sum of two numbers.

The best answer choice is "returns the sum of two numbers".
----
"""

auto_cot_cleaned_prompt = """(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
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
The final answer is one of the choices.
A: Let's think step-by-step.

First, we have a try statement. This means that the code inside the try block will be executed first.
Inside the try block, we have two variables, n and m. We are trying to convert them into integers using the int() function.
Then, we are adding the two integers and storing the result in a variable called integer_sum.
Finally, we are printing the value of integer_sum.
If the input values are not integers, then the code inside the except block will be executed. The except block will print the string 'error'.

The best answer choice is prints sum of two input numbers only if they are integers otherwise raises error
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6
The final answer is one of the choices.
A: Let's think step-by-step.

The first line creates a list called numbers_list.
The second line uses the filter function to filter out numbers that are greater than 7.
The third line prints the filtered_list.
Therefore, the code snippet prints a list of numbers greater than 7 from numbers_list.

The best answer choice is prints a list of numbers greater than 7 from numbers_list.
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
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
The final answer is one of the choices.
A: Let's think step-by-step.

sum_of_numbers = 0
num = 10
The code sets up a variable called sum_of_numbers, which starts at 0, and a variable called num, which starts at 10.

while(num > 0):
This while loop will keep going as long as num is greater than 0.

sum_of_numbers += num
This line adds the value of num to sum_of_numbers.

num -= 1
This line decreases the value of num by 1.

print('The sum is', sum_of_numbers)
Finally, the code prints the sum of the numbers from 1 to 10.

The best answer choice is prints the sum of the numbers from 1 to 10.
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
for i in range(23):
    print(i)

  choice: prints 'hello world' to the terminal
  choice: prints values from 1 to 10
  choice: computes first 10 prime numbers
  choice: prints values from 0 to 22
The final answer is one of the choices.
A: Let's think step-by-step.

The for loop iterates over a range of numbers from 0 to 23.
For each number in the range, the code prints the number to the screen.
So, the code will print values from 0 to 22 to the screen.

The best answer choice is print values from 0 to 22 to the screen.
----
(Code Description) Which of the following choices best describes the functionality of the given python code snippet.
Python code:
def sum_two(a,b):
    return a+b

  choice: returns a and b
  choice: returns the sum of two numbers
  choice: returns 5 and 6
  choice: returns numbers between 5 and 6
The final answer is one of the choices.
A: Let's think step-by-step.

def sum_two(a,b):
This defines a function that takes two parameters, a and b.
return a+b
This line tells the function to return the sum of a and b.

The best answer choice is returns the sum of two numbers.
----
"""


def auto_cot(
    temperature=0.3,
    model_name="text-davinci-002",
    predict=True,
    use_corrected=False,
    self_consistency=False,
):
    global auto_cot_corrected_prompt
    global auto_cot_cleaned_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Anachronisms\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=500, temperature=temperature, quote="---", n=1
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1] :])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is """, x).span(0)[-1] :]]
                            )
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 20)):
                x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [
                get_autocot_answer(x, answer_prompt="The best answer choice is ") for x in answers
            ]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used.
# Description: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
# Input:
# ```
# if x < 5:
# 	pass
# ```
# What error does this program surface?
# Q1: [code execute] Execute the following python code snippet.
# if x < 5:
# 	pass
# #1:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'x' is not defined
# Q2: [get error message] What is the final error message?
# #2: NameError: name 'x' is not defined
# Q3: [EOQ]
# Ans: NameError: name 'x' is not defined
# ----
# Desciption: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
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
# Q2: [code execute] Execute the following python code snippet.
# x = set([1, 1, 2, 3])
# print(x)
# #2: {1, 2, 3}
# Q3: [EOQ]
# Ans: {1, 2, 3}
# ----
# Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet.
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
# Q5: [compare] Which of the generated code snippets are most like the original one?
# #5: prints sum of two input numbers only if they are integers otherwise raises error
# Q6: [EOQ]
# Ans: prints sum of two input numbers only if they are integers otherwise raises error
# ----
# Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet.
# Input:
# Python code:
# numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
# filtered_list = list(filter(lambda num: (num > 7), numbers_list))
# print(filtered_list)

#   choice: prints lambda
#   choice: returns a filtered list
#   choice: prints a list of numbers greater than 7 from numbers_list
#   choice: prints numbers from 2 to 6
# Q1: [code generate] prints lambda
# #1:
# print("lambda")
# Q2: [code generate] returns a filtered list
# #2
# def filter_list(l):
#   'return a new list with the strings filtered out'
#   return [x for x in l if not isinstance(x, str)]
# Q3: [code generate] prints a list of numbers greater than 7 from numbers_list
# #3:
# numbers_list = [1,2,3,4,5,6,7,8,9,10]
# for number in numbers_list:
#     if number > 7:
#         print(number)
# Q4: [code generate] prints numbers from 2 to 6
# #4:
# for i in range(2,7):
#     print(i)
# Q5: Which of the generated code snippets are most like the original one?
# #5: prints a list of numbers greater than 7 from numbers_list
# Q6: [EOQ]
# Ans: prints a list of numbers greater than 7 from numbers_list
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_code_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Code Description"
    task_description = """(Code Description) Which of the following choices best describes the functionality of the given python code snippet."""

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
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

    def predict_complete(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        outputs = gpt3(prompts)
        completed_outputs = [
            interpreter.complete_program(prefix, output) for prefix, output in zip(prompts, outputs)
        ]
        return completed_outputs

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
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
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 5
        batch_size = 10
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
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
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\n\nEnglish language description:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            # time.sleep(60)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "text-davinci-002",
            "text-davinci-003",
            "code-davinci-002",
            "code-cushman-001",
            "davinci-codex-002-msft",
        ],
        default="text-davinci-002",
    )
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument(
        "--inference_strategy",
        type=str,
        choices=[
            "dummy",
            "few_shot",
            "auto_cot",
            "cot_rollout",
            "few_shot_cot",
            "nl_program",
        ],
        default="few_shot",
    )
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action="store_true")

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[: args.num_dev_examples]
    labels = labels[: args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)["input_ids"]))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(
            args.temperature,
            args.model_name,
            predict=True,
            use_corrected=False,
            self_consistency=False,
        )
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)
