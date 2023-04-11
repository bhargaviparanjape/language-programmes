import argparse
from collections import Counter
import pdb
import re

import datasets
import numpy as np
from prompt_library import (
    few_shot_arithmetic_prompt,
    few_shot_string_prompt,
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

d = datasets.load_dataset("bigbench", "repeat_copy_logic", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]
# print(len(inputs))


train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

io_pairs = [
    (
        "Q: say pickup a pound of green beans twice, replacing a pound with a bunch for even times and a handful for odd",
        "pickup a handful of green beans pickup a bunch of green beans",
    ),
    (
        "Q: repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time",
        "a woodchuck chucks five pounds of wood a woodchuck chucks two tons of wood",
    ),
    (
        "Q: Repeat squiggly line twice after the phrase can you draw",
        "can you draw squiggly line squiggly line",
    ),
]

task_name = "Repeat with logic"
task_description = "(Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question."


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
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=200, quote="---", n=1
        )
        prompts = [
            """repeat with logic:

Q: say pickup a pound of green beans twice, replacing a pound with a bunch for even times and a handful for odd
A:
pickup a handful of green beans pickup a bunch of green beans
----
repeat with logic:

Q: repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time
A:
a woodchuck chucks five pounds of wood a woodchuck chucks two tons of wood
----
repeat with logic:

Q: Repeat squiggly line twice after the phrase can you draw
A:
can you draw squiggly line squiggly line
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
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: say pickup a pound of green beans twice, replacing a pound with a bunch for even times and a handful for odd
The final answer should be the repeated phrase.
A: Let's think step-by-step.

"Pick up a pound of green beans" is our starting phrase. This is repeated twice.

"Pick up a pound of green beans Pick up a pound of green beans"

For the first (odd) repetition, we replace "a pound" with "a handful", so our new phrase is "Pick up a handful of green beans".

For the second (even) repetition, we replace "a pound" with "a bunch", so our new phrase is "Pick up a bunch of green beans".

The final answer is 'Pick up a handful of green beans Pick up a bunch of green beans".
----
(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time
The final answer should be the repeated phrase.
A: Let's think step-by-step.

"A woodchuck chucks lots of wood" is our starting phrase. This is repeated twice.

"A woodchuck chucks lots of wood A woodchuck chucks lots of wood"

For the first repetition, we replace "lots" with "five pounds", so our new phrase is "A woodchuck chucks five pounds of wood"

For the second repetition, we replace "lots" with "two tons", so our new phrase is ""A woodchuck chucks two tons of wood"

The final answer is "A woodchuck chucks five pounds of wood A woodchuck chucks two tons of wood"
----
(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: Repeat squiggly line twice after the phrase can you draw
The final answer should be the repeated phrase.
A: Let's think step-by-step.

First, we have the phrase "can you draw." Next, we need to repeat the phrase "squiggly line" twice. This gives us the phrase "squiggly line squiggly line" This phrase is then appended to the first phrase.

The final answer is "can you draw squiggly line squiggly line"
----
"""

auto_cot_cleaned_prompt = """(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: say pickup a pound of green beans twice, replacing a pound with a bunch for even times and a handful for odd
The final answer should be the repeated phrase.
A: Let's think step-by-step.

Pick up a pound of green beans.

The final answer is
Pick up a bunch of green beans.
Pick up a handful of green beans.
Pick up a pound of green beans.
----
(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time
The final answer should be the repeated phrase.
A: Let's think step-by-step.

A woodchuck chucks lots of wood.

The final answer is
A woodchuck chucks five pounds of wood.
A woodchuck chucks two tons of wood.
----
(Repeat with logic) Repeat the given phrase following logical constraints provided in the question.
Q: Repeat squiggly line twice after the phrase can you draw
The final answer should be the repeated phrase.
A: Let's think step-by-step.

Squiggly line can you draw

The final answer is Squiggly line can you draw squiggly line
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
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer should be the repeated phrase.\nA: Let's think step-by-step.\n"""
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
    f.write("Language games\n\n")
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
            + """%s\nThe final answer should be in the repeated phrase.\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer should be in the repeated phrase.\nA: Let's think step-by-step.\n"""
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
                x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
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
                        if re.search("""The final answer is""", pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is""", x).span(0)[-1] :]]
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
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x, answer_prompt="The final answer is") for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
# Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
# Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
# Q1: [string reformat] first day of 2007 in MM/DD/YYYY
# #1: 01/01/2007
# Q2: [arithmetic] What date is one week from 01/01/2007?
# #2: 01/08/2007
# Q3: [EOQ]
# Ans: 01/08/2007
# ----
# Description: Translate English into Pig Latin.
# Input: (English) Sami made his way across the bar and hugged Layla.
# Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
# #1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
# Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
# #2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
# Q3: [string merge] Concatenate #2 into a full sentence.
# #3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
# Q4: [EOQ]
# Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
# ----
# Description: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
# Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
# Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
# #1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
# Q2: [string index] What is the third letter of words in the list in #1?
# #2: ["v", "e", "m", "t", "d"]
# Q3: [string merge] Concatenate #2 with spaces
# #3: "v e m t d"
# Q4: [EOQ]
# Ans: v e m t d
# ----
# Desciption: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
# Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
# Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
# #1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
# Q2: [string index] What is the third letter of words in the list in #1?
# #2: ["r", "a", "i", "o", "u"]
# Q3: [string merge] Concatenate #2 with spaces
# #3: "r a i u o"
# Q4: [EOQ]
# Ans: r a i u o
# ----
# Description: Translate English into Pig Latin.
# Input: (English) Tom is the most popular boy in school.
# Q1: [string split] What are the words in "Tom is the most popular boy in school."?
# #1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
# Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
# #2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
# Q3: [string merge] Concatenate #2 into a full sentence.
# #3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
# Q4: [EOQ]
# Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
# ----
# Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
# Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
# Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
# #1: 06/01/2021
# Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
# #2: Today is 04/01/2021
# Q3: [arithmetic] What date is 24 hours later than today?
# #3: 05/01/2021
# Q4: [EOQ]
# Ans: 05/31/2021
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_string_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Repeat with Logic"
    task_description = "(Repeat with logic) Repeat the given phrase following logical constraints provided in the question."

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
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):
    task_name = "Repeating text with logic"
    task_description = (
        "Repeat the given phrase following logical constraints provided in the question."
    )

    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=2048,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = few_shot_arithmetic_prompt


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    # task_name = "Repeating text with logic"
    # task_description = "Repeat the given phrase following logical constraints provided in the question, editing the phrase as required."

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

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


few_shot_notebook_prompt = """In [1]:
import os
import sys
import numpy as np
import re
from sympy.solvers import solve
from sympy import Symbol, Eq
import math
from sympy import simplify
import numpy as np
import cvxpy as cp
import statistics
from serpapi import GoogleSearch
from utils import string_compare

In [1]:
input_text = "Take the letters at position 3 of the words in 'Ibrahim Francois Pei Shu Ngo' and concatenate them using a space."
def get_sentence(text):
    text_start = re.search("words in '", text).span(0)[1]
    text_end = re.search("' and'", text).span(0)[0]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Ibrahim Francois Pei Shu Ngo"

In [2]:
input_text = "Ibrahim Francois Pei Shu Ngo"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]

In [3]:
input_tokens = ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
def get_nth_char_in_list(token_list, n):
    char_list = [token[n-1] for token in token_list]
    return char_list
get_nth_char_in_list(input_tokens, 3)

Out [3]:
["r", "a", "i", "o", "u"]

In [4]:
char_list = ["r", "a", "i", "o", "u"]
def merge_chars(char_list):
    return " ".join(char_list)
merge_chars(char_list)

Out [4]:
r a i u o

In [5]:
final_answer = r a i u o
print("Ans: ",final_answer)

Out [5]:
r a i u o

In [1]:
input_text = "Take the letters at position 3 of the words in 'Savita Saeed Ramos Sato Yadav' and concatenate them using a space."
def get_sentence(text):
    text_start = re.search("words in '", text).span(0)[1]
    text_end = re.search("' and'", text).span(0)[0]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Savita Saeed Ramos Sato Yadav"

In [2]:
input_text = "Savita Saeed Ramos Sato Yadav"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]

In [3]:
input_tokens = ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
def get_nth_char_in_list(token_list, n):
    char_list = [token[n-1] for token in token_list]
    return char_list
get_nth_char_in_list(input_tokens, 3)

Out [3]:
["v", "e", "m", "t", "d"]

In [4]:
char_list = ["v", "e", "m", "t", "d"]
def merge_chars(char_list):
    return " ".join(char_list)
merge_chars(char_list)

Out [4]:
v e m t d

In [5]:
final_answer = v e m t d
print("Ans: ",final_answer)

Out [5]:
Ans: v e m t d

In [1]:
input_text = "Translate English into Pig Latin. (English) Sami made his way across the bar and hugged Layla."

Out [1]:
def get_sentence(text):
    text_start = re.search("English into Pig Latin. \(English\) ", text).span(0)[1]
    text_end = re.search("\.", text).span(0)[1]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Sami made his way across the bar and hugged Layla"

In [2]:
input_text = "Sami made his way across the bar and hugged Layla"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Sami", "made", "his", "way", "across", "the", "bar", "and", "hugged", "Layla"]

In [3]:
input_tokens = ["Sami", "made", "his", "way", "across", "the", "bar", "and", "hugged", "Layla"]
def get_pig_latin(token_list):
    vowels = ["a", "e", "i", "o", "u", "y"]
    pig_latin_tokens = []
    for token in token_list:
        if token[0] in vowels:
            pig_latin_tokens.append(token + "yay")
        else:
            pig_latin_tokens.append(token[1:] + token[0] + "ay")
    return pig_latin_tokens
get_pig_latin(input_tokens)

Out [3]:
['Amisay', 'ademay', 'ishay', 'ayway', 'acrossyay', 'hetay', 'arbay', 'andyay', 'uggedhay', 'Aylalay']

In [4]:
pig_latin_tokens = ['amisyay', 'ademyay', 'ishyay', 'ayway', 'rossacay', 'ethyay', 'aryabay', 'andyay', 'uggedhay', 'aylaLay']
def merge_tokens(token_list):
    return " ".join(token_list)
merge_tokens(pig_latin_tokens)

Out [4]:
'Amisay ademay ishay ayway acrossyay hetay arbay andyay uggedhay Aylalay'

In [5]:
final_answer = 'amisyay ademyay ishyay ayway rossacay ethyay aryabay andyay uggedhay aylaLay'
print("Ans: ",final_answer)

Out [5]:
Ans: amisyay ademyay ishyay ayway rossacay ethyay aryabay andyay uggedhay aylaLay

In [1]:
input_text = "Translate English into Pig Latin. (English) Tom is the most popular boy in school."

Out [1]:
def get_sentence(text):
    text_start = re.search("(English) ", text).span(0)[1]
    text_end = re.search("\.", text).span(0)[1]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Tom is the most popular boy in school."

In [2]:
input_text = "Tom is the most popular boy in school."
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Tom", "is", "the", "most", "popular", "boy", "in", "school."]

In [3]:
input_tokens = ["Tom", "is", "the", "most", "popular", "boy", "in", "school."]
def isVowel(letter):
    if letter in ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'):
        return True
    else:
        return False
def translate(token_list):
    return ' '.join([token[1:] + token[0:1] + 'ay' if isVowel(token[0]) else token[2:] + token[0:2] + 'ay' for token in token_list])
translate(input_tokens)

Out [3]:
"Omtay isay ethay ostmay opularpay oybay inay oolschay."

In [4]:
final_answer = Omtay isay ethay ostmay opularpay oybay inay oolschay.
print("Ans: ",final_answer)

Out [4]:
"Ans: Omtay isay ethay ostmay opularpay oybay inay oolschay."

In [1]:
input_text = "%s"
"""


def notebook(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name,
            max_length=2048,
            temperature=temperature,
            quote="In [1]:",
            n=1,
        )
        prompts = [few_shot_notebook_prompt % (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [
                ex.replace(
                    "repeat with logic:\n\n",
                    "Repeat the given phrase following logical constraints provided in the question. ",
                )
                for ex in x
            ]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
            pdb.set_trace()
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_human_prompt = """Description: (Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question.
Input: A watermelon has seven seeds. Repeat they're delicious once for every seed
Q1: [code generate] A watermelon has seven seeds. Write python code to repeat they're delicious once for every seed
#1:
repeat = []
for i in range(7):
    repeat.append("they're delicious")
ans = " ".join(repeat)
print(ans)
Q2: [execute] Execute the following code snippet
repeat = []
for i in range(7):
    repeat.append("they're delicious")
ans = " ".join(repeat)
print(ans)
#2: they're delicious they're delicious they're delicious they're delicious they're delicious they're delicious they're delicious
Q3: [EOQ]
Ans: they're delicious they're delicious they're delicious they're delicious they're delicious they're delicious they're delicious
----
Description: (Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question.
Input: Repeat pickup a pound of green beans twice, replacing a pound with a bunch for even times and a handful for odd
Q1: [code generate] Python code to print "pickup a pound of green beans" twice, replacing a pound with a bunch for even times and a handful for odd.
#1:
repeat = []
for i in range(1,3):
    if i %% 2 == 0:
    	repeat.append("pickup a bunch of green beans")
    else:
    	repeat.append("pickup a handful of green beans")
ans =  ' '.join(repeat)
print(ans)
Q2: [execute] Execute the following code snippet
repeat = []
for i in range(1,3):
    if i %% 2 == 0:
    	repeat.append("pickup a bunch of green beans")
    else:
    	repeat.append("pickup a handful of green beans")
ans =  ' '.join(repeat)
print(ans)
#2: pickup a handful of green beans pickup a bunch of green beans
Q3: [EOQ]
Ans: pickup a handful of green beans pickup a bunch of green beans
----
Description: (Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question.
Input: Print logic eleven times, but after the 3rd and 8th also say true
Q1: [code generate] Python code to print logic eleven times, but after the 3rd and 8th also say true
#1:
repeat = []
for i in range(1,12):
    repeat.append("logic")
    if i == 3 or i == 8:
        repeat.append("true")
ans =  ' '.join(repeat)
print(ans)
Q2: [execute] Execute the following code snippet
repeat = []
for i in range(1,12):
    repeat.append("logic")
    if i == 3 or i == 8:
        repeat.append("true")
ans =  ' '.join(repeat)
print(ans)
#2: logic logic logic true logic logic logic logic logic true logic logic logic
Q3: [EOQ]
Ans: logic logic logic true logic logic logic logic logic true logic logic logic
----
Description: (Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question.
Input: Repeat hello world five times, but don't say world every even time
Q1: [code generate] Python code to repeat hello world five times, but don't say world every even time
#1:
repeat = []
for i in range(1,6):
    if i %% 2 == 0:
        repeat.append("Hello")
    else:
        repeat.append("Hello World")
ans = " ".join(repeat)
print(ans)
Q2: [execute] Execute the following code snippet
repeat = []
for i in range(1,6):
    if i %% 2 == 0:
        repeat.append("Hello")
    else:
        repeat.append("Hello World")
ans = " ".join(repeat)
print(ans)
#2: Hello World Hello Hello World Hello Hello World
Q3: [EOQ]
Ans: Hello World Hello Hello World Hello Hello World
----
Description: (Repeat with logic) Repeat the given phrase with spaces in between, following logical constraints provided in the question.
Input: Repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time
Q1: [code generate] repeat a woodchuck chucks lots of wood two times, but replace lots with five pounds the first time and two tons the second time
#1:
repeat = []
for i in range(1,3):
    if i == 1:
        repeat.append("A woodchuck chucks five pounds of wood")
    else:
        repeat.append("A woodchuck chucks two tons of wood")
ans = ' '.join(repeat)
print(ans)
Q2: [execute] Execute the following code snippet
repeat = []
for i in range(1,3):
    if i == 1:
        repeat.append("A woodchuck chucks five pounds of wood")
    else:
        repeat.append("A woodchuck chucks two tons of wood")
ans = ' '.join(repeat)
print(ans)
#2: A woodchuck chucks five pounds of wood A woodchuck chucks two tons of wood
Q3: [EOQ]
Ans: A woodchuck chucks five pounds of wood A woodchuck chucks two tons of wood
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
    interpreter = TopDownVisitorBeta(model_name=model_name)

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
                x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
                prompts, answer = predict(task_description, x)
                # new_answer  = interpreter.batch_visit(prompts, answer)
                answers.extend(answer)
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
            "auto_cot_corrected",
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
