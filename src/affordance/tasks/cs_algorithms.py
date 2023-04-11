import argparse
from collections import Counter
import json
import pdb
import re
import time

import datasets
import numpy as np
from prompt_library import (
    few_shot_arithmetic_prompt,
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
    substring_match_v2,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


task_name = "CS Algorithms"
task_description = """(CS Algorithms) Solve the following simple programming tasks using Python."""

io_pairs = [
    (
        """Given two strings, determine the length of the longest common subsequence.

Strings: VIRVRHRSTQBLLSYPZDVYCFPSQRXNA SPLYVHLWMLDJVYMQTOZMVOJF
Length of longest common subsequence:""",
        "8",
    ),
    (
        """Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:""",
        "0",
    ),
    (
        """Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:""",
        "4",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: } { ( [
Valid/Invalid?""",
        "Invalid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: [ [ [ { [ [ ] { { } ( ) } [ ] ] } ] ] ]
Valid/Invalid?""",
        "Valid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?""",
        "Valid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?""",
        "Invalid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?""",
        "Valid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?""",
        "Invalid",
    ),
    (
        """Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ] { } { [ ] [ ] } [ ] ( { ( ) } )
Valid/Invalid?""",
        "Valid",
    ),
]

d = datasets.load_dataset("bigbench", "cs_algorithms", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0 * correct) / count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=200, quote="---", n=1)
        prompts = [
            """Given two strings, determine the length of the longest common subsequence.

Strings: VIRVRHRSTQBLLSYPZDVYCFPSQRXNA SPLYVHLWMLDJVYMQTOZMVOJF
Length of longest common subsequence:
8
----
Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:
0
----
Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:
4
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: } { ( [
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ [ [ { [ [ ] { { } ( ) } [ ] ] } ] ] ]
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ] { } { [ ] [ ] } [ ] ( { ( ) } )
Valid/Invalid?
Valid
----
%s
"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            time.sleep(5)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
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

few_shot_cot_prompt = few_shot_arithmetic_prompt

auto_cot_corrected_prompt = """CS Algorithms: Solve the following simple programming tasks.
Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:
A: Let's think step-by-step.

1. What is the length of the longest common subsequence of the strings "SCZFZGCCQQLB" and "OJDXI"?
2. What is the length of the longest common subsequence of the strings "SCZFZGCCQQL" and "OJDXI"?
3. What is the length of the longest common subsequence of the strings "SCZFZGCCQQ" and "OJDXI"?
4. What is the length of the longest common subsequence of the strings "SCZFZGCCQ" and "OJDXI"?
5. What is the length of the longest common subsequence of the strings "SCZFZGC" and "OJDXI"?
6. What is the length of the longest common subsequence of the strings "SCZFZG" and "OJDXI"?
7. What is the length of the longest common subsequence of the strings "SCZF" and "OJDXI"?
8. What is the length of the longest common subsequence of the strings "SCZ" and "OJDXI"?
9. What is the length of the longest common subsequence of the strings "SC" and "OJDXI"?
10. What is the length of the longest common subsequence of the strings "S" and "OJDXI"?
The length of the longest common subsequence is 3.

The final answer is 3.
----
CS Algorithms: Solve the following simple programming tasks.
Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:
A: Let's think step-by-step.

1) We can see that the first letter of each string is 'R'. So, we can start our subsequence with 'R'.
2) The next letter in each string is 'L'. So, we can add 'L' to our subsequence.
3) The next letter in each string is 'X'. Since there is no common letter after 'X', we cannot add anything to our subsequence.
4) The next letter in each string is 'E'. So, we can add 'E' to our subsequence.
5) The next letter in each string is 'H'. So, we can add 'H' to our subsequence.
6) The next letter in each string is 'V'. Since there is no common letter after 'V', we cannot add anything to our subsequence.
7) The next letter in each string is 'G'. So, we can add 'G' to our subsequence.
8) The next letter in each string is 'P'. So, we can add 'P' to our subsequence.
9) The next letter in each string is 'C'. So, we can add 'C' to our subsequence.
10) The next letter in each string is 'L'. So, we can add 'L' to our subsequence.
11) The next letter in each string is 'D'. Since there is no common letter after 'D', we cannot add anything to our subsequence.
12) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
13) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
14) The next letter in each string is 'B'. So, we can add 'B' to our subsequence.
15) The next letter in each string is 'A'. So, we can add 'A' to our subsequence.
16) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
17) The next letter in each string is 'C'. So, we can add 'C' to our subsequence.
18) The next letter in each string is 'Q'. Since there is no common letter after 'Q', , we cannot add anything to our subsequence.

The final answer cannot be found.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?
A: Let's think step-by-step.

The first character is '[', which is an opening parenthesis. This is fine.
The next character is '{', which is also an opening parenthesis. This is fine.
The next character is '}', which is a closing parenthesis. This is fine.
The next character is ']', which is a closing parenthesis. This is fine.
So far, everything is matching up nicely. The opening parentheses are being closed in the correct order.
The sequence is valid.

The final answer is valid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?
A: Let's think step-by-step.

The first character is a closing parenthesis, so that's not good. The next two characters are a closing curly brace and bracket, respectively, so those are also not good. The next character is an opening parenthesis, so that's good. The next two characters are an opening curly brace and bracket, respectively, so those are also good. The next character is a closing parenthesis, so that's not good. The next two characters are a closing curly brace and bracket, respectively, so those are also not good. The next character is an opening parenthesis, so that's good. The next two characters are an opening curly brace and bracket, respectively, so those are also good. The next character is a closing parenthesis, so that's not good.
Therefore, the sequence is invalid.

The final answer is invalid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?
A: Let's think step-by-step.

The first opening parenthesis is '(', so the next character must be a ')'. The next character is a '[', so the next character must be a ']'. The next character is '(', so the next character must be a ')'. The next character is a '{', so the next character must be a '}'. The sequence is valid.

The final answer is valid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?
A: Let's think step-by-step.

The first character is an open square bracket, so we'll need to find a corresponding close square bracket later on. The next character is an open curly brace, so we'll need to find a corresponding close curly brace later on. So far, everything is fine.
The next character is a close curly brace, but we don't have an open curly brace that we can match it to! This means the sequence is invalid.

The final answer is invalid.
----
"""

auto_cot_cleaned_prompt = """CS Algorithms: Solve the following simple programming tasks.
Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:
A: Let's think step-by-step.

1. What is the length of the longest common subsequence of the strings "SCZFZGCCQQLB" and "OJDXI"?
2. What is the length of the longest common subsequence of the strings "SCZFZGCCQQL" and "OJDXI"?
3. What is the length of the longest common subsequence of the strings "SCZFZGCCQQ" and "OJDXI"?
4. What is the length of the longest common subsequence of the strings "SCZFZGCCQ" and "OJDXI"?
5. What is the length of the longest common subsequence of the strings "SCZFZGC" and "OJDXI"?
6. What is the length of the longest common subsequence of the strings "SCZFZG" and "OJDXI"?
7. What is the length of the longest common subsequence of the strings "SCZF" and "OJDXI"?
8. What is the length of the longest common subsequence of the strings "SCZ" and "OJDXI"?
9. What is the length of the longest common subsequence of the strings "SC" and "OJDXI"?
10. What is the length of the longest common subsequence of the strings "S" and "OJDXI"?
The length of the longest common subsequence is 3.

The final answer is 3.
----
CS Algorithms: Solve the following simple programming tasks.
Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:
A: Let's think step-by-step.

1) We can see that the first letter of each string is 'R'. So, we can start our subsequence with 'R'.
2) The next letter in each string is 'L'. So, we can add 'L' to our subsequence.
3) The next letter in each string is 'X'. Since there is no common letter after 'X', we cannot add anything to our subsequence.
4) The next letter in each string is 'E'. So, we can add 'E' to our subsequence.
5) The next letter in each string is 'H'. So, we can add 'H' to our subsequence.
6) The next letter in each string is 'V'. Since there is no common letter after 'V', we cannot add anything to our subsequence.
7) The next letter in each string is 'G'. So, we can add 'G' to our subsequence.
8) The next letter in each string is 'P'. So, we can add 'P' to our subsequence.
9) The next letter in each string is 'C'. So, we can add 'C' to our subsequence.
10) The next letter in each string is 'L'. So, we can add 'L' to our subsequence.
11) The next letter in each string is 'D'. Since there is no common letter after 'D', we cannot add anything to our subsequence.
12) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
13) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
14) The next letter in each string is 'B'. So, we can add 'B' to our subsequence.
15) The next letter in each string is 'A'. So, we can add 'A' to our subsequence.
16) The next letter in each string is 'O'. So, we can add 'O' to our subsequence.
17) The next letter in each string is 'C'. So, we can add 'C' to our subsequence.
18) The next letter in each string is 'Q'. Since there is no common letter after 'Q', , we cannot add anything to our subsequence.

The final answer cannot be found.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?
A: Let's think step-by-step.

The first character is '[', which is an opening parenthesis. This is fine.
The next character is '{', which is also an opening parenthesis. This is fine.
The next character is '}', which is a closing parenthesis. This is fine.
The next character is ']', which is a closing parenthesis. This is fine.
So far, everything is matching up nicely. The opening parentheses are being closed in the correct order.
The sequence is valid.

The final answer is valid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?
A: Let's think step-by-step.

The first character is a closing parenthesis, so that's not good. The next two characters are a closing curly brace and bracket, respectively, so those are also not good. The next character is an opening parenthesis, so that's good. The next two characters are an opening curly brace and bracket, respectively, so those are also good. The next character is a closing parenthesis, so that's not good. The next two characters are a closing curly brace and bracket, respectively, so those are also not good. The next character is an opening parenthesis, so that's good. The next two characters are an opening curly brace and bracket, respectively, so those are also good. The next character is a closing parenthesis, so that's not good.
Therefore, the sequence is invalid.

The final answer is invalid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?
A: Let's think step-by-step.

The first opening parenthesis is '(', so the next character must be a ')'. The next character is a '[', so the next character must be a ']'. The next character is '(', so the next character must be a ')'. The next character is a '{', so the next character must be a '}'. The sequence is valid.

The final answer is valid.
----
CS Algorithms: Solve the following simple programming tasks.
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?
A: Let's think step-by-step.

The first character is an open square bracket, so we'll need to find a corresponding close square bracket later on. The next character is an open curly brace, so we'll need to find a corresponding close curly brace later on. So far, everything is fine.
The next character is a close curly brace, but we don't have an open curly brace that we can match it to! This means the sequence is invalid.

The final answer is invalid.
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
    description = "CS Algorithms: Solve the following simple programming tasks."
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=500, temperature=0.2, quote="---", n=1
        )
        prompt = """%s\n""" % description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_corrected_prompt

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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=1000, temperature=0.2, quote="---", n=1
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % description
            + """%s\nA: Let's think step-by-step.\n""" % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            # time.sleep(10)
        pdb.set_trace()
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=2048, temperature=0.4, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=2048, temperature=0.4, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers = predict(
                "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                x,
            )
            pdb.set_trace()
            affordance_inputs = [
                json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers
            ]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [
                ex + a[: re.search("#2: ", a).span(0)[1]] + json.dumps(o)
                for ex, a, o in zip(x, answers, affordance_outputs)
            ]
            new_answers.extend(
                predict_with_affordance(
                    "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                    x,
                )
            )
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "CS Algorithms"
    task_description = (
        """(CS Algorithms) Solve the following simple programming tasks using Python."""
    )

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
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
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
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "CS Algorithms"
    task_description = "(CS Algorithms) Solve the following simple programming tasks using Python, executing the code to get the final answer."

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

    interpreter = TopDownVisitorBeta(
        model_name=model_name,
        exclude_list=["[code generate]", "[generate python code]"],
    )

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=15):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 15)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in processed_answers:
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers[: len(inputs)]]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            label_dict = {"Invalid": "False", "Valid": "True"}
            new_labels = [label_dict.get(label, label) for label in labels]
            combined_labels = [[label, new_label] for label, new_label in zip(labels, new_labels)]
            for x in tqdm(chunks(inputs, 10)):
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match_v2(combined_labels, preds))
            print(perf_array)
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / len(interpreter.execution_details)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Description: (CS Algorithms) Solve the following simple programming tasks using Python.
Input: Given two strings, determine the length of the longest common subsequence.

Strings: IGONYTSVDC FFVQURTOILEOJZLQIJFPKCLOG
Q1: [code generate] Given two strings, determine the length of the longest common subsequence.
#1:
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "IGONYTSVDC"
str2 = "FFVQURTOILEOJZLQIJFPKCLOG"
ans = lcs(str1, str2)
print(ans)
Q2: [execute] Execute the following python code snippet.
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "IGONYTSVDC"
str2 = "FFVQURTOILEOJZLQIJFPKCLOG"
ans = lcs(str1, str2)
print(ans)
#2: 3
Q3: [EOQ]
Ans: 3
----
Description: (CS Algorithms) Solve the following simple programming tasks using Python.
Input: Given two strings, determine the length of the longest common subsequence.

Strings: ONOCPYDR WLDZYGSWVEOXOXCPFKRTTZJ
Q1: [code generate] Given two strings, determine the length of the longest common subsequence.
#1:
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "ONOCPYDR"
str2 = "WLDZYGSWVEOXOXCPFKRTTZJ"
ans = lcs(str1, str2)
print(ans)
Q2: [execute] Execute the following python code snippet.
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "ONOCPYDR"
str2 = "WLDZYGSWVEOXOXCPFKRTTZJ"
ans = lcs(str1, str2)
print(ans)
#2: 5
Q3: [EOQ]
Ans: 5
----
Description: (CS Algorithms) Solve the following simple programming tasks using Python.
Input: Given two strings, determine the length of the longest common subsequence.

Strings: GFPTAOVHYDTFQVXAXSGCAWCPCEC TVUDCTXTCRDSELUATDGAZLKCTD
Q1: [code generate] Given two strings, determine the length of the longest common subsequence.
#1:
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "GFPTAOVHYDTFQVXAXSGCAWCPCEC"
str2 = "TVUDCTXTCRDSELUATDGAZLKCTD"
ans = lcs(str1, str2)
print(ans)
Q2: [execute] Execute the following python code snippet.
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    return L[m][n]
str1 = "GFPTAOVHYDTFQVXAXSGCAWCPCEC"
str2 = "TVUDCTXTCRDSELUATDGAZLKCTD"
ans = lcs(str1, str2)
print(ans)
#2: 9
Q3: [EOQ]
Ans: 9
----
Description: (CS Algorithms) Solve the following simple programming tasks using Python.
Input: Determine whether the given sequence of parentheses is properly matched.

Sequence: { [ [ ( ( ) [ ] ( { } ) ) ] ] ( { } ) }
Q1: [code generate] Determine whether the given sequence of parentheses is properly matched.
#1:
def is_matched(expression):
    stack = []
    for i in expression:
        if i == '{' or i == '[' or i == '(':
            stack.append(i)
        elif i == '}' or i == ']' or i == ')':
            if len(stack) == 0:
                return False
            else:
                if i == '}' and stack[-1] == '{':
                    stack.pop()
                elif i == ']' and stack[-1] == '[':
                    stack.pop()
                elif i == ')' and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
    if len(stack) == 0:
        return "valid"
    else:
        return "invalid"
expression = "{ [ [ ( ( ) [ ] ( { } ) ) ] ] ( { } ) }"
ans = is_matched(expression)
print(ans)
Q2: [execute] Execute python code snippet.
def is_matched(expression):
    stack = []
    for i in expression:
        if i == '{' or i == '[' or i == '(':
            stack.append(i)
        elif i == '}' or i == ']' or i == ')':
            if len(stack) == 0:
                return False
            else:
                if i == '}' and stack[-1] == '{':
                    stack.pop()
                elif i == ']' and stack[-1] == '[':
                    stack.pop()
                elif i == ')' and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
    if len(stack) == 0:
        return "valid"
    else:
        return "invalid"
expression = "{ [ [ ( ( ) [ ] ( { } ) ) ] ] ( { } ) }"
ans = is_matched(expression)
print(ans)
#2: valid
Q3: [EOQ]
Ans: valid
----
Description: (CS Algorithms) Solve the following simple programming tasks using Python.
Input: Determine whether the given sequence of parentheses is properly matched.

Sequence: ( } ( ) ) ) ) ) [ [
Q1: [code generate] Determine whether the given sequence of parentheses is properly matched.
#1:
def is_matched(expression):
    stack = []
    for i in expression:
        if i == '{' or i == '[' or i == '(':
            stack.append(i)
        elif i == '}' or i == ']' or i == ')':
            if len(stack) == 0:
                return False
            else:
                if i == '}' and stack[-1] == '{':
                    stack.pop()
                elif i == ']' and stack[-1] == '[':
                    stack.pop()
                elif i == ')' and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
    if len(stack) == 0:
        return "valid"
    else:
        return "invalid"
expression = "( } ( ) ) ) ) ) [ ["
ans = is_matched(expression)
print(ans)
Q2: [execute] Execute python code snippet.
def is_matched(expression):
    stack = []
    for i in expression:
        if i == '{' or i == '[' or i == '(':
            stack.append(i)
        elif i == '}' or i == ']' or i == ')':
            if len(stack) == 0:
                return False
            else:
                if i == '}' and stack[-1] == '{':
                    stack.pop()
                elif i == ']' and stack[-1] == '[':
                    stack.pop()
                elif i == ')' and stack[-1] == '(':
                    stack.pop()
                else:
                    return False
    if len(stack) == 0:
        return "valid"
    else:
        return "invalid"
expression = "( } ( ) ) ) ) ) [ ["
ans = is_matched(expression)
print(ans)
#2: invalid
Q3: [EOQ]
Ans: invalid
----
Description: %s
Input: %s
Q1:"""


def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_human_prompt
    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=9):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nEdited:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 9)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in enumerate(processed_answers):
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0][1] for x in answers[: len(inputs)]]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("\nLength of longest common subsequence:", "") for ex in x]
                x = [ex.replace("\nValid/Invalid?", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))
            # Report on interpreter performance
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / (len(interpreter.execution_details) + 1e-6)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


# human_intervention(0.3, "davinci-codex-002-msft")


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
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"],
        default="fixed",
    )

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    # inputs = inputs[:args.num_dev_examples]
    # labels = labels[:args.num_dev_examples]
    selected = np.random.choice(len(inputs), size=args.num_dev_examples, replace=False)
    inputs = [inputs[idx] for idx in selected]
    labels = [labels[idx] for idx in selected]

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
        few_shot_cot(args.temperature, args.model_name, strategy=args.selection_strategy)
    elif args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            self_consistency=args.self_consistency,
            strategy=args.selection_strategy,
        )
