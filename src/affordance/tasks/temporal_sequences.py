import argparse
from collections import Counter
import json
import pdb
import re

import datasets
import numpy as np
from prompt_library import (
    few_shot_arithmetic_prompt,
    few_shot_free_prompt,
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

d = datasets.load_dataset("bigbench", "temporal_sequences", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = """(Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity."""

io_pairs = [
    (
        """Today, Elizabeth went to the construction site. Between what times could they have gone?
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
        "11am to 1pm",
    ),
    (
        """Today, Andrew went to the art show. Between what times could they have gone?
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
        "6am to 10am",
    ),
    (
        """Today, Susan went to the orchestra hall. Between what times could they have gone?
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
        "12pm to 1pm",
    ),
    (
        """Today, Jennifer went to the art show. Between what times could they have gone?
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
        "5am to 7am",
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


def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
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
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
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
# Q3: [EOQ]
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
# Q3: [EOQ]
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
# Q6: [EOQ]
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

auto_cot_cleaned_prompt = """TASK:
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

First, we need to find out when Lisa saw Elizabeth. We know that Lisa saw Elizabeth from 5am to 10am. This means that Elizabeth must have gone to the construction site sometime between 10am and 11am.
Next, we need to find out when Mary saw Elizabeth. We know that Mary saw Elizabeth from 10am to 11am. This means that Elizabeth must have gone to the construction site sometime between 11am and 1pm.
Lastly, we need to find out when William saw Elizabeth. We know that William saw Elizabeth from 3pm to 9pm. This means that Elizabeth must have gone to the construction site sometime between 3pm and 9pm.
Therefore, the possible times that Elizabeth could have gone to the construction site are from 11am to 1pm, from 5am to 10am, and from 3pm to 9pm.

The best answer choice is 5am to 10am.
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

First, we know that Andrew woke up at 6am. So, the earliest he could have gone to the art show would be 6am.
Next, we know that Kimberly saw Andrew waiting at the airport from 10am to 12pm. So, the latest he could have gone to the art show would be 10am.
That means that the only time that Andrew could have gone to the art show would be between 6am and 10am.

The best answer choice is 6am and 10am.
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

First, we need to determine when Susan woke up. We know that she woke up at 5am, so we'll put that down as our first possible time.
Next, we need to determine when Thomas saw her. We know that he saw her from 5am to 9am, so we'll put that down as our second possible time.
After that, we need to determine when Ashley saw her. We know that she saw her from 9am to 10am, so we'll put that down as our third possible time.
Now, we need to determine when Betty saw her. We know that she saw her from 10am to 12pm, so we'll put that down as our fourth possible time.
Finally, we need to determine when Linda saw her. We know that she saw her from 1pm to 3pm, so we'll put that down as our fifth possible time.
We also know that the orchestra hall was closed after 8pm, so we'll put that down as our sixth possible time.
That means that the possible times Susan could have gone to the orchestra hall are 5am, 9am, 10am, 12pm, 1pm, and 3pm.

The best answer choice is 5am to 9am.
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

1. Jennifer woke up at 5am. She could have gone to the art show at this time.
2. Ashley saw Jennifer sitting on a rooftop from 7am to 3pm. This means that the latest Jennifer could have gone to the art show is 3pm.
3. David saw Jennifer buying a bike at the bike shop from 3pm to 5pm. This means that the earliest Jennifer could have gone to the art show is 5pm.
4. Steven saw Jennifer playing tennis at the tennis court from 5pm to 6pm. This means that the latest Jennifer could have gone to the art show is 6pm.
5. Susan saw Jennifer waiting at the airport from 6pm to 8pm. This means that the earliest Jennifer could have gone to the art show is 8pm.
6. Anthony saw Jennifer stretching at a yoga studio from 8pm to 10pm. This means that the latest Jennifer could have gone to the art show is 10pm.
So, the possible times Jennifer could have gone to the art show are 5am to 3pm, 5pm to 6pm, and 8pm to 10pm.

The best answer choice is 5pm to 6pm.
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
    description = "TASK: "
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.2, quote="---", n=1)
        prompt = """%s\n""" % description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.2, quote="---", n=1)
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
            x = [ex.replace("\nPossible times:", "") for ex in x]
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [get_autocot_answer(x, answer_prompt="The best answer choice is") for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def human_decomp(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=1000,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [
            """Q: Today, Emily went to the museum. Between what times could they have gone?
We know that:
Emily woke up at 1pm.
Elizabeth saw Emily reading at the library from 2pm to 4pm.
Jessica saw Emily watching a movie at the theater from 4pm to 5pm.
Leslie saw Emily waiting at the airport from 5pm to 6pm.
William saw Emily buying clothes at the mall from 6pm to 7pm.
The museum was closed after 7pm.
Between what times could Emily have gone to the museum?
  choice: 1pm to 2pm
  choice: 6pm to 7pm
  choice: 5pm to 6pm
  choice: 2pm to 4pm
A: Let's think step by step.
Wake-up time: 1pm.
1pm-2pm: free.
2pm-4pm: reading at the library.
4pm-5pm: watching a movie at the theater.
5pm-6pm: waiting at the airport.
6pm-7pm: buying clothes at the mall.
The museum closure time: 7pm.
The only time when Emily could have gone to the museum was 1pm to 2pm. So the answer is 1pm to 2pm.
----
Q: Today, Elizabeth went to the amusement park. Between what times could they have gone?
We know that:
Elizabeth woke up at 7am.
David saw Elizabeth fixing their computer at the electronic store from 1pm to 2pm.
Sarah saw Elizabeth playing tennis at the tennis court from 2pm to 3pm.
Susan saw Elizabeth walking towards the Statue of Liberty from 3pm to 6pm.
Andrew saw Elizabeth taking photos near the Eiffel Tower from 6pm to 9pm.
Emily saw Elizabeth getting a coffee at the cafe from 9pm to 10pm.
The amusement park was closed after 10pm.
Between what times could Elizabeth have gone to the amusement park?
  choice: 7am to 1pm
  choice: 9pm to 10pm
  choice: 1pm to 2pm
  choice: 3pm to 6pm
A: Let's think step by step.
Wake-up time: 7am.
7am-1pm: free.
1pm-2pm: fixing their computer at the electronic store.
2pm-3pm: playing tennis at the tennis court.
3pm-6pm: walking towards the Statue of Liberty.
6pm-9pm: taking photos near the Eiffel Tower.
9pm-10pm: getting a coffee at the cafe.
The amusement park closure time: 10pm.
The only time when Elizabeth could have gone to the amusement park was 7am to 1pm. So the answer is 7am to 1pm.
----
Q: Today, Tiffany went to the beach. Between what times could they have gone?
We know that:
Tiffany woke up at 5am.
Betty saw Tiffany getting a coffee at the cafe from 5am to 6am.
Jessica saw Tiffany working at the office from 6am to 9am.
John saw Tiffany stretching at a yoga studio from 9am to 12pm.
Sean saw Tiffany sitting on a rooftop from 12pm to 2pm.
Sarah saw Tiffany playing tennis at the tennis court from 2pm to 3pm.
The beach was closed after 4pm.
Between what times could Tiffany have gone to the beach?
  choice: 9am to 12pm
  choice: 12pm to 2pm
  choice: 5am to 6am
  choice: 3pm to 4pm
A: Let's think step by step.
Wake-up time: 5am.
5am-6am: getting a coffee at the cafe.
6am-9am: working at the office.
9am-12pm: stretching at a yoga studio.
12pm-2pm: sitting on a rooftop.
2pm-3pm: playing tennis at the tennis court.
3pm-4pm: free.
The beach closure time: 4pm.
The only time when Tiffany could have gone to the beach was 3pm to 4pm. So the answer is 3pm to 4pm.
----
%s
A: Let's think step by step."""
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
            x = [ex.replace("\nPossible times:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
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
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
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


few_shot_pot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
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

few_shot_cot_prompt = few_shot_arithmetic_prompt


# few_shot_cot_prompt = few_shot_free_prompt
def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
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
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_pot_prompt % (description, x) for x in chunk]
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
    task_description = "Temporal sequences: Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        label_dict = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
        choices = [
            [ans.strip() for ans in inp.replace("\nPossible times:", "").split("choice: ")][1:]
            for inp in inputs
        ]
        new_labels = [label_dict[choice.index(label)] for label, choice in zip(labels, choices)]
        joint_labels = [[label, new_label] for label, new_label in zip(labels, new_labels)]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("""\nPossible times:""", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match_v2(joint_labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_pot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    task_description = "Select the option that best replaces '()' in each text input given the chocies presented, by solving the arithmetic word problem in the text input. Solving the problem will require reasoning about units of measurement. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
            new_answers = []
            for ans in answers:
                try:
                    parsed_program = parse_program("Input: dummy\nQ1:" + ans)
                    code_snippet = parsed_program.node_list[0].command_output
                    result = safe_execute(code_snippet)
                except:
                    result = None
                if result:
                    new_answers.append(result)
                else:
                    new_answers.append(
                        ans[re.search("Ans:", ans).span(0)[0] :] if "Ans:" in ans else ans
                    )
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use arithmetic and algebra functions in one or more of your substeps.
Description: Solve the following arithmetic problems on ratios and fractions, , writing out intermediate arithmetic calculations as needed.
Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number?
    choice: 29
    choice: 30
    choice: 31
    choice: 32
    choice: None
Q1: [algebra] Let the greater part be x. What is the lesser part?
#1: 49-x
Q2: [algebra] What is the greater part if increased by 6
#2: x+6
Q2: [algebra] What is the lesser part if decreased by 11
#2: 49-x-11
Q3: [algebra] What is the ratio of greater to lesser after the change?
#3: (x+6):(49-x-11)
Q4: [algebra] Write down the equation to solve for x
#4: (x+6):(49-x-11) = 9:2
Q5: [solve] Solve for x: (x+6):(49-x-11) = 9:2
#5: x = 30
Q6: [compare] Which option is closes to this answer?
#6: 30
Q7: [EOQ]
30
----
Description: Find the date based on provided information.
Input: Today is the last day of the first quarter of 2008. What is the date one week from today in MM/DD/YYYY?
Q1: [search] What is the first quarter of a year?
#1: The traditional calendar quarters that make up the year are: Dates for Q1: January 1 – March 31. Dates for Q2: April 1 – June 3. Dates for Q3: July 1 – September 30. Dates for Q4: October 1 – December 31.
Q2: [arithmetic] What is the last day of the first quarter?
#2: March 31
Q3: [arithmetic] What day is today?
#3: March 31, 2008
Q4: [string reformat] March 31, 2008
#4: 03/31/2008
Q5: [arithmetic] What is 1 week from today?
#5: 04/07/2008
Q6: [EOQ]
04/07/2008
----
Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
Q1: What is total number of toys that need to be made?
#1: 400
Q2: How many workers in total?
#2: 5
Q3: [arithmetic] How many toys do 2 of the workers produce?
#3: 2 workers working for 10 hours making 6 toys per hour produce 2*10*6 = 120 toys.
Q4: [arithmetic] How many toys do another 2 of workers produce?
#4: 2 workers working for 10 hours making 4 toys per hour produce 2*10*4 = 80 toys.
Q5: [arithmetic] How many toys did the last worker make?
#5: Out of 400 toys, 120+80=200 toys were made by the first 4 workers. The 5th worker didn't finish the job, since 20 toys were still remaining to be made. So they made 400-200-20=180 toys.
Q6: [arithmetic] How many toys per hour does the fifth worker produce if he worked for 10 hours?
#6: The 5th worker made 180/10 = 18 toys per hour.
Q7: [EOQ]
18
----
Description: What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as needed.
Input: add 70 to 90, subtract 20 from result, subtract result from 200.
 choice:130
 choice:80
 choice:150
 choice:100
 choice:60
Q1: [arithmetic] add 70 to 90
#1: 70+90=160
Q2: [arithmetic] subtract 20 from 160
#2: 160-20=140
Q3: [arithmetic] subtract result 140 from 200
#3: 200-140=60
Q4: [compare] Which option does the final answer match?
#4: 60
Q5: [EOQ]
60
----
Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
Q1: [arithmetic] How much bread does Viola have if he had 167 loafs before and Nancy too 137 of them?
#1: 167-137=30
Q2: [EOQ]
30
----
Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No"
Input: Take 7 steps. Turn right. Take 8 steps. Turn around.
Q1: Does this question require vector arithmetic?
#1: Yes.
Q2: [subquestion] Which way are you facing when you start? If unknown, assume you face forward?
#2: Face forward
Q3: [subquestion] What is the distance moved forward?
#3: 7 steps
Q4: [subquestion] What is the distance moved right?
#4: 8 steps
Q5: [subquestion] What is the distance moved backward?
#5: 0 steps
Q6: [subquestion] What is the distance moved left?
#6: 0 steps
Q7: [arithmetic] What is total distance moved from starting point?
#7: 7 steps vertically, 8 steps horizontally
Q8: [subquestion] Is the total distance moved, both vertically and horizontally, 0?
#8: No
Q9: [EOQ]
No
----
Description:
Input: If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?
Q1: [arithmetic] What is the speed of the second train?
#1: 60/2=30 miles an hour
Q2: [arithmetic] What is distance travelled by first train?
#2: 60*3=180 miles
Q3: [arithmetic] What is distance travelled by first train?
#3: 30*3=90 miles
Q4: [subquestion] Are the trains moving towards or away from each other?
#4: Away from each other.
Q5: [arithmetic] How far apart are the trains after 3 hours?
#5: 180+90=270 miles
Q6: [EOQ]
270 miles
----
Desciption: %s
Input: %s
Q1: """

few_shot_cot_prompt = few_shot_free_prompt


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_pot_prompt
    global few_shot_cot_prompt
    task_name = "(Temporal sequences)"
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

    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        # new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nPossible times:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            pdb.set_trace()
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / (len(interpreter.execution_details) + 1e-6)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Description: (Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity.
Input: Today, Tiffany went to the swimming pool. Between what times could they have gone?
We know that:
Tiffany woke up at 6am.
Jessica saw Tiffany waiting at the airport from 6am to 10am.
William saw Tiffany attending class at the school from 12pm to 5pm.
David saw Tiffany walking towards the Statue of Liberty from 5pm to 6pm.
James saw Tiffany reading at the library from 6pm to 8pm.
The swimming pool was closed after 8pm.
Between what times could Tiffany have gone to the swimming pool?
  choice: 12pm to 5pm
  choice: 6pm to 8pm
  choice: 10am to 12pm
  choice: 6am to 10am
Q1: [subquestion] Is Tiffany free 12pm to 5pm?
#1: No, she is attending classes at the school.
Q2: [subquestion] Is Tiffany free 6pm to 8pm?
#2: No, she is reading at the library.
Q3: [subquestion] Is Tiffany free 10am to 12pm?
#3: Yes, there is no other activity in her scehdule.
Q4: [subquestion] Is Tiffany free 6am to 10am?
#4: No, she is waiting at the airport.
Q5: [subquestion] Is the swimming pool open from 10am to 12pm?
#5: Yes, the swimming pool is open from 10am to 8pm.
Q6: [get answer] In conclusion, when should she go to the swimming pool?
#6: Tiffany should go to the swimming pool from 10am to 12pm.
Q7: [EOQ]
Ans: 10am to 12pm
----
Description: (Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity.
Input: Today, David went to the art studio. Between what times could they have gone?
We know that:
David woke up at 8am.
Leslie saw David working at the office from 8am to 9am.
David saw David fixing their computer at the electronic store from 9am to 10am.
Thomas saw David buying clothes at the mall from 10am to 11am.
Andrew saw David taking photos near the Leaning Tower of Pisa from 11am to 12pm.
Jennifer saw David walking towards the Statue of Liberty from 4pm to 6pm.
The art studio was closed after 6pm.
Between what times could David have gone to the art studio?
  choice: 12pm to 4pm
  choice: 9am to 10am
  choice: 11am to 12pm
  choice: 8am to 9am
Q1: [subquestion] Is David free 12pm to 4pm?
#1: Yes, there is no other activity in his scehdule.
Q2: [subquestion] Is David free 9am to 10am?
#2: No, he is fixing their computer at the electronic store.
Q3: [subquestion] Is David free 11am to 12pm?
#3: No, he is taking photos near the Leaning Tower of Pisa.
Q4: [subquestion] Is David free 8am to 9am?
#4: No, he is working at the office.
Q5: [subquestion] Is the art studio open from 12pm to 4pm?
#5: Yes, the art studio is open from 10am to 6pm.
Q6: [get answer] In conclusion, when should David go to the art studio?
#6: David should go to the art studio from 12pm to 4pm.
Q7: [EOQ]
Ans: 12pm to 4pm
----
Description: (Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity.
Input: Today, William went to the soccer field. Between what times could they have gone?
We know that:
William woke up at 5am.
Mark saw William walking towards the Statue of Liberty from 5am to 6am.
Jason saw William working at the office from 6am to 9am.
Andrew saw William playing tennis at the tennis court from 3pm to 5pm.
The soccer field was closed after 5pm.
Between what times could William have gone to the soccer field?
  choice: 5am to 6am
  choice: 9am to 3pm
  choice: 3pm to 5pm
  choice: 6am to 9am
Q1: [subquestion] Is William free 5am to 6am?
#1: No, he is walking towards the Statue of Liberty.
Q2: [subquestion] Is William free 9am to 3pm?
#2: Yes, there is no other activity in his scehdule.
Q3: [subquestion] Is William free 3pm to 5pm?
#3: No, he is playing tennis at the tennis court.
Q4: [subquestion] Is William free 6am to 9am?
#4: No, he is working at the office.
Q5: [subquestion] Is the soccer field open from 9am to 3pm?
#5: Yes, the soccer field is open from 9am to 5pm.
Q6: [get answer] In conclusion, when should William go to the soccer field?
#6: William should go to the soccer field from 9am to 3pm.
Q7: [EOQ]
Ans: 9am to 3pm
----
Description: (Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity.
Input: Today, Linda went to the market. Between what times could they have gone?
We know that:
Linda woke up at 5am.
William saw Linda watching a movie at the theater from 5am to 7am.
Richard saw Linda walking towards the Statue of Liberty from 7am to 8am.
Sean saw Linda waiting at the train station from 8am to 9am.
Leslie saw Linda getting a coffee at the cafe from 9am to 1pm.
The market was closed after 4pm.
Between what times could Linda have gone to the market?
  choice: 5am to 7am
  choice: 8am to 9am
  choice: 1pm to 4pm
  choice: 7am to 8am
Q1: [subquestion] Is Linda free 5am to 7am?
#1: No, she is watching a movie at the theater.
Q2: [subquestion] Is Linda free 8am to 9am?
#2: No, she is waiting at the train station.
Q3: [subquestion] Is Linda free 1pm to 4pm?
#3: Yes, there is no other activity in her scehdule.
Q4: [subquestion] Is Linda free 7am to 8am?
#4: No, she is walking towards the Statue of Liberty.
Q5: [subquestion] Is the market open from 1pm to 4pm?
#5: Yes, the market is open from 9am to 4pm.
Q6: [get answer] In conclusion, when should Linda go to the market?
#6: Linda should go to the market from 1pm to 4pm.
Q7: [EOQ]
Ans: 1pm to 4pm
----
Description: (Temporal sequences) Given the daily schedule of activities of a person and a final activity that they need to find time for, choose one of the four provided options when they could do the activity.
Input: Today, Richard went to the physics classroom. Between what times could they have gone?
We know that:
Richard woke up at 7am.
David saw Richard reading at the library from 10am to 11am.
Samantha saw Richard sitting on a rooftop from 11am to 2pm.
Ashley saw Richard working at the office from 2pm to 4pm.
Hannah saw Richard fixing their computer at the electronic store from 4pm to 8pm.
The physics classroom was closed after 8pm.
Between what times could Richard have gone to the physics classroom?
  choice: 2pm to 4pm
  choice: 7am to 10am
  choice: 11am to 2pm
  choice: 4pm to 8pm
Q1: [subquestion] Is Richard free 2pm to 4pm?
#1: No, he is working at the office.
Q2: [subquestion] Is Richard free 7am to 10am?
#2: Yes, there is no other activity in his scehdule.
Q3: [subquestion] Is Richard free 11am to 2pm?
#3: No, he is sitting on a rooftop.
Q4: [subquestion] Is Richard free 4pm to 8pm?
#4: No, he is fixing their computer at the electronic store.
Q5: [subquestion] Is the physics classroom open from 7am to 10am?
#5: Yes, the physics classroom is open from 8am to 8pm.
Q6: [get answer] In conclusion, when should Richard go to the physics classroom?
#6: Richard should go to the physics classroom from 7am to 10am.
Q7: [EOQ]
Ans: 7am to 10am
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
                x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
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
                x = [ex.replace("""\nPossible times:""", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
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


# human_intervention(0.3, "text-davinci-002")
# human_decomp()

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
            "auto_cot_corrected",
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
    elif args.inference_strategy == "auto_cot_corrected":
        auto_cot(
            args.temperature,
            args.model_name,
            predict=True,
            use_corrected=True,
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
