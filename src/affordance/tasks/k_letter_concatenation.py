import argparse
from collections import Counter
import json
import pdb
import re
import time
import urllib.request

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
    chunks,
    get_answer,
    get_autocot_answer,
    get_few_shot_prompt,
    get_subset,
    propose_decomposition,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Get data

url = "https://raw.githubusercontent.com/allenai/DecomP/main/datasets/letter_cat/n5_eg100_pos2_space.json"
response = urllib.request.urlopen(url)
data = json.loads(response.read())
dev_inputs = [d["question"] for d in data["1"]["qa_pairs"]]
dev_labels = [d["answer"]["spans"][0] for d in data["1"]["qa_pairs"]]

url = "https://raw.githubusercontent.com/allenai/DecomP/main/datasets/letter_cat/n5_eg100_pos2_space.json"
response = urllib.request.urlopen(url)
data = json.loads(response.read())
train_inputs = [d["question"] for d in data["1"]["qa_pairs"]]
train_labels = [d["answer"]["spans"] for d in data["1"]["qa_pairs"]]

task_description = "(Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space."

io_pairs = [
    (
        """Take the letters at position 3 of the words in "Maya Sylvie Campbell Hussein Francisco" and concatenate them using a space.""",
        "y l m s a",
    ),
    (
        """Take the letters at position 3 of the words in "Beatriz Pauline Tang Murmu Idris" and concatenate them using a space.""",
        "a u n r r",
    ),
    (
        """Take the letters at position 3 of the words in "Yousef Emma Uddin Suleiman Zhan" and concatenate them using a space.""",
        "u m d l a",
    ),
    (
        """Take the letters at position 3 of the words in "Satish Enrique Sahu Rana Sih" and concatenate them using a space.""",
        "t r h n h",
    ),
    (
        """Take the letters at position 3 of the words in "Ali Cecilia Calderon Chakraborty Chai" and concatenate them using a space.""",
        "i c l a a",
    ),
    (
        """Take the letters at position 3 of the words in "Weiping Asma Nawaz Acosta Quispe" and concatenate them using a space.""",
        "i m w o i",
    ),
    (
        """Take the letters at position 3 of the words in "Frank Aicha Tan Manuel Patel" and concatenate them using a space.""",
        "a c n n t",
    ),
    (
        """Take the letters at position 3 of the words in "Hui Mahendra Munda Doan Mensah" and concatenate them using a space.""",
        "i h n a n",
    ),
    (
        """Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.""",
        "v e m t d",
    ),
    (
        """Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.""",
        "r a i u o",
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
    print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)["input_ids"]))

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
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def k_letter_concatenation(temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Take the letters at position 3 of the words in "Maya Sylvie Campbell Hussein Francisco" and concatenate them using a space.
y l m s a
----
Take the letters at position 3 of the words in "Beatriz Pauline Tang Murmu Idris" and concatenate them using a space.
a u n r r
----
Take the letters at position 3 of the words in "Yousef Emma Uddin Suleiman Zhan" and concatenate them using a space.
u m d l a
----
Take the letters at position 3 of the words in "Satish Enrique Sahu Rana Sih" and concatenate them using a space.
t r h n h
----
Take the letters at position 3 of the words in "Ali Cecilia Calderon Chakraborty Chai" and concatenate them using a space.
i c l a a
----
Take the letters at position 3 of the words in "Weiping Asma Nawaz Acosta Quispe" and concatenate them using a space.
i m w o i
----
Take the letters at position 3 of the words in "Frank Aicha Tan Manuel Patel" and concatenate them using a space.
a c n n t
----
Take the letters at position 3 of the words in "Hui Mahendra Munda Doan Mensah" and concatenate them using a space.
i h n a n
----
Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
v e m t d
----
Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
r a i u o
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
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def human_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """Take the letters at position 3 of the words in "Noel Jenny Duong Awad" and concatenate them using a space.
Word: Noel
3rd Letter: e
Word: Jenny
3rd Letter: n
Word: Duong
3rd Letter: o
Word: Awad
3rd Letter: a
Concatenation:
e n o a
----
Take the letters at position 3 of the words in "Kalpana Scott Guerrero Hamed" and concatenate them using a space.
Word: Kalpana
3rd Letter: l
Word: Scott
3rd Letter: o
Word: Guerrero
3rd Letter: e
Word: Hamed
3rd Letter: m
Concatenation:
l o e m
----
Take the letters at position 3 of the words in "Maryia Wendy Sangma Ayala" and concatenate them using a space.
Word: Maryia
3rd Letter: r
Word: Wendy
3rd Letter: n
Word: Sangma
3rd Letter: n
Word: Ayala
3rd Letter: a
Concatenation:
r n n a
----
Take the letters at position 3 of the words in "Eduardo Juan Hong Raut" and concatenate them using a space.
Word: Eduardo
3rd Letter: u
Word: Juan
3rd Letter: a
Word: Hong
3rd Letter: n
Word: Raut
3rd Letter: u
Concatenation:
u a n u
----
Take the letters at position 3 of the words in "Mark Jianhui Sekh Salas" and concatenate them using a space.
Word: Mark
3rd Letter: r
Word: Jianhui
3rd Letter: a
Word: Sekh
3rd Letter: k
Word: Salas
3rd Letter: l
Concatenation:
r a k l
----
Take the letters at position 3 of the words in "Liying Dan Jimenez Zin" and concatenate them using a space.
Word: Liying
3rd Letter: y
Word: Dan
3rd Letter: n
Word: Jimenez
3rd Letter: m
Word: Zin
3rd Letter: n
Concatenation:
y n m n
----
Take the letters at position 3 of the words in "Timothy Mona Nascimento Kaur" and concatenate them using a space.
Word: Timothy
3rd Letter: m
Word: Mona
3rd Letter: n
Word: Nascimento
3rd Letter: s
Word: Kaur
3rd Letter: u
Concatenation:
m n s u
----
Take the letters at position 3 of the words in "Eduardo Juan Hong Raut" and concatenate them using a space.
Word: Eduardo
3rd Letter: u
Word: Juan
3rd Letter: a
Word: Hong
3rd Letter: n
Word: Raut
3rd Letter: u
Concatenation:
u a n u
----
Take the letters at position 3 of the words in "Linh Radha Amadou Pires" and concatenate them using a space.
Word: Linh
3rd Letter: n
Word: Radha
3rd Letter: d
Word: Amadou
3rd Letter: a
Word: Pires
3rd Letter: r
Concatenation:
n d a r
----
Take the letters at position 3 of the words in "Dorothy Eric Mensah Moyo" and concatenate them using a space.
Word: Dorothy
3rd Letter: r
Word: Eric
3rd Letter: i
Word: Mensah
3rd Letter: n
Word: Moyo
3rd Letter: y
Concatenation:
r i n y
----
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
        preds = [x.split("\n")[-1].strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("Pattern based decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def cot_rollout(temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Take the last letters of the words in "Augusta Ada King" and concatenate them using a space.
Work: The words in "Augusta Ada King" are "Augusta", "Ada" and "King". The letters and their positions in "Augusta" are "[(A, 1), (u, 2), (g, 3), (u, 4), (s, 5), (t, 6), (a, 7)]". The last letter in this sequence is "a". The letters and their positions in "Ada" are "[(A, 1), (d, 2), (a, 3)]". The last letter in this sequence is "a". The letters and their positions in "King" are "[(K, 1), (i, 2), ( n, 3), (g, 4)]". The last letter in this sequence is "g". Concatenating "a", "a", "g" using a space leads to "a a g". So, "Augusta Ada King" outputs "a a g".
Answer: a a g
----
Take the letters at position 1 of the words in "Alan Mathison Turing" and concatenate them
using a space.
Work: The words in "Alan Mathison Turing" are "Alan", "Mathison", and "Turing". The letters and their positions in "Alan" are "[(A, 1), (l, 2), (a, 3), (n, 4)]". The letter at position 1 in this sequence is "A". The letters and their positions in "Mathison" are [(M, 1), (a, 2), (t, 3), (h, 4), (i, 5), (s, 6), (o, 7), (n, 8)]". The letter at position 1 in this sequence is "M". The letters and their positions in "Turing" are "[(T, 1), (u, 2), (r, 3), (i, 4), (n, 5), (g, 6)]". The letter at position 1 in this sequence is "T". Concatenating "A", "M", "T" using a space leads to "A M T". So, "Alan Mathison Turing" outputs "A M T".
Answer: A M T
----
Take the letters at position 4 of the words in "Herbert Alexander Simon" and concatenate
them using a space.
Work: The words in "Herbert Alexander Simon" are "Herbert", "Alexander", and "Simon". The letters and their positions in "Herbert" are "[(H, 1), (e, 2), (r, 3), (b, 4), (e, 5), (r, 6), (t, 7)]".The letter at position 4 in this sequence is "b". The letters and their positions in "Alexander" are "[(A, 1), (l, 2), (e, 3), (x, 4), (a, 5), (n, 6), (d, 7), (e, 8), (r, 9)]". The letter at position 4 in this sequence is "x". The letters and their positions in "Simon" are "[(S, 1), (i, 2), (m, 3), (o, 4), (n, 5)]". The letter at position 4 in this sequence is "o". Concatenating "b", "x", "o" using a space leads to "b x o". So, "Herbert Alexander Simon" outputs "b x o".
Answer: b x o
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
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
        preds = [x.split("\n")[-1].replace("Answer: ", "") for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def automatic_decomposition(N=10, temperature=0.3):
    # The other ways to do this is to ask what functions to use and given these functions, come up with required steps.
    decomp_prompt = (
        "%s I want to break this task into individual steps to make it easier to solve."
        % task_description
    )
    decompositions = propose_decomposition(decomp_prompt, io_pairs[:5], N)

    def get_decomp_func(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])
        decomposition += "\n%s. The final answer contains the letters separated by spaces." % (
            last_n + 1
        )

        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(
                model="text-davinci-002",
                max_length=1000,
                temperature=temperature,
                quote="---",
                n=1,
            )
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """%s Using the following steps will help.
Steps:
%s
----
%s
Also show how you used the steps provided to arrive at the answer."""
                    % (task_description, decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(dev_inputs, labels=dev_labels, n=len(dev_inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition:", z)
        print(decompositions[z])
        fn = get_decomp_func(decomposition, batch_size=20)
        subset = [ex.replace("\nA:", "") for ex in subset]
        this_preds = fn(subset)
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
        print(pp.mean())
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.std(accs))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
# Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
# Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
# Q1: [string reformat] first day of 2007 in MM/DD/YYYY
# #1: 01/01/2007
# Q2: [subquestion] What date is one week from 01/01/2007?
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
# Q2: [subquestion] 06/01/2021 is 2 days away from now. What date is today?
# #2: Today is 04/01/2021
# Q3: [subquestion] What date is 24 hours later than today?
# #3: 05/01/2021
# Q4: [EOQ]
# Ans: 05/31/2021
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_string_prompt
# few_shot_cot_prompt = few_shot_code_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Kth Letter concatenation"
    task_description = "(Kth Letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space."

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
        for x in tqdm(chunks(dev_inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
            # pdb.set_trace()
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(dev_labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Maya Sylvie Campbell Hussein Francisco" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.
1. Take the third letter of each word:
"Maya": y
"Sylvie": l
"Campbell": m
"Hussein": s
"Francisco": a

2. Concatenate them using a space:
y l m s a
The final answer is "y l m s a"
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Beatriz Pauline Tang Murmu Idris" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

1. Take the 3rd letter of "Beatriz": 'a'.

2. Take the 3rd letter of "Pauline": 'u'.

3. Take the 3rd letter of "Tang": 'n'.

4. Take the 3rd letter of "Murmu": 'r'.

5. Take the 3rd letter of "Idris": 'r'.

The final answer is "a u n r r"
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Yousef Emma Uddin Suleiman Zhan" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

Step 1: Take the letters at position 3 of each word:

Yousef: u
Emma: m
Uddin: d
Suleiman: l
Zhan: a

Step 2: Concatenate the letters with a space:

The final answer is "u m d l a"
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Satish Enrique Sahu Rana Sih" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

Step 1: Take the words in the list:
Satish
Enrique
Sahu
Rana
Sih

Step 2: Take the letter at position 3 of each word:
t r h n h

Step 3: Concatenate the letters using a space:
t r h n h

The final answer is "t r h n h"
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Ali Cecilia Calderon Chakraborty Chai" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

First, let's break down the list of words:

Ali
Cecilia
Calderon
Chakraborty
Chai

Now, let's take the letter at position 3 of each word:

i
c
l
a
a

Finally, we can concatenate the letters separated by spaces:

i c l a a
The final answer is "i c l a a"
----
"""

auto_cot_cleaned_prompt = """Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Maya Sylvie Campbell Hussein Francisco" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

1. We need to take the third letter from each word.
2. We need to concatenate them using a space.

So, for step 1, we can use the string indexing method to take the third letter from each word. For step 2, we can use the string concatenation operator (+) to join the letters together with a space in between.
Putting it all together, we get:
"Maya Sylvie Campbell Hussein Francisco".

3. We can use a for loop to iterate over the words in the string, and print out the third letter of each word.


def third_letter(string):
for word in string.split():
print(word[2])
 fourth_letter("Maya Sylvie Campbell Hussein Francisco")
c
l
m
s
F

The final answer is c
l
m
s
F
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Beatriz Pauline Tang Murmu Idris" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

Take the 3rd letter of each word in the list:

B: ea
P: u
T: r
M: i
I: s

C: Now, we concatenate them using a space:

D: ea u r i s

The final answer is ea u r i s
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Yousef Emma Uddin Suleiman Zhan" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

1) We need to take the letters at position 3 of each word in the list.

2) We need to concatenate them using a space.
In order to do this, we can use the string method .join().

3) We need to print the final answer.
Putting this all together, we have:
words = "Yousef Emma Uddin Suleiman Zhan"
letters = [word[3] for word in words.split()]
print(" ".join(letters))
This will give us the output:
f m d l n

The final answer is f m d l n
----
Kth letter concatenation: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Satish Enrique Sahu Rana Sih" and concatenate them using a space.
The final answer contains the letters separated by spaces.
A: Let's think step-by-step.

The first thing we need to do is to split the string into a list of words. We can do this by using the .split() method:

>>> "Satish Enrique Sahu Rana Sih".split()
['Satish', 'Enrique', 'Sahu', 'Rana', 'Sih']

Now that we have a list of words, we can use a for loop to iterate through each word and take the third letter of each word:

>>> for word in ["Satish", "Enrique", "Sahu", "Rana", "Sih"]:
...     print(word[2])
...
i
q
h
n
h

Notice that we use the print() function to print each letter on a new line.

Finally, we can concatenate the letters into a single string, separating them with a space:

>>> " ".join(["i", "q", "h", "n", "h"])
'i q h n h'

Therefore, the answer is:
'i q h n h'

The final answer is 'i q h n h'
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
            + """\nThe final answer contains the letters separated by spaces.\nA: Let's think step-by-step.\n"""
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
    f.write("Kth letter concatenation\n\n")
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
            + """%s\nThe final answer contains the letters separated by spaces.\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer contains the letters separated by spaces.\nA: Let's think step-by-step.\n"""
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
            for x in tqdm(chunks(dev_inputs, batch_size)):
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
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is """, x).span(0)[-1] :]]
                            )
                time.sleep(60)
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(dev_inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x, answer_prompt="The final answer is") for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3):
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

    def predict_self_consistency(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=2048,
            temperature=temperature,
            quote="---",
            n=10,
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        to_return = []
        results = gpt3(prompts)
        pdb.set_trace()
        for idx, g in enumerate(results["choices"]):
            text = g["text"]
            logprob = sum(g["logprobs"]["token_logprobs"])
            to_return.append((text, logprob))
        to_return = sorted(to_return, key=lambda tup: tup[1], reverse=True)
        to_return = [r[0] for r in to_return]
        return to_return

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
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
        new_answers = []
        for x in tqdm(chunks(dev_inputs, 10)):
            answers = predict_self_consistency(
                "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                x,
            )
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
        # preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        preds = [ans.split("Ans:", 1)[-1] for ans in new_answers]
        preds = [x.strip() for x in preds]
        perf_array.append(exact_match(dev_labels, preds))
        print(perf_array)
    print("Few-shot Affordance COT performance:")
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
    task_name = "Simple Text Editing"
    task_description = "(Simple Text Editing) Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."

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

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(dev_inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    new_answer = interpreter.batch_visit([prompts[chunk_no]] * 5, ans_chunk)
                    new_answer = [get_answer(program) for program in new_answer]
                    for enum, pred in enumerate(new_answer):
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                time.sleep(60)
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(dev_inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(dev_labels, preds))

            # Report on interpreter performance
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
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
        for x in tqdm(chunks(dev_inputs, 10)):
            x = [ex.replace("\nA:", "").replace('"', "'") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(dev_labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_human_prompt = """Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Musa Haiying Schmidt Robinson Afzal" and concatenate them using a space.
Q1: [generate python code] write down the code to solve the problem, storing the answer as 'ans'
#1:
words = "Musa Haiying Schmidt Robinson Afzal"
words_list = words.split(' ')
ans = ''
third_letters = [word[2] for word in words_list]
ans = ' '.join(third_letters)
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: s i h b z
Q3: [EOQ]
Ans: s i h b z
----
Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Gang Ping Mendez Cabrera Uddin" and concatenate them using a space.
Q1: [generate python code] write python code to solve the problem
#1:
words = "Gang Ping Mendez Cabrera Uddin"
words_list = words.split(' ')
third_letters = [word[2] for word in words_list]
ans = ' '.join(third_letters)
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: n n n b d
Q3: [EOQ]
Ans: n n n b d
----
Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Anne Shyam Watson Bux Malik" and concatenate them using a space.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
words = "Anne Shyam Watson Bux Malik"
words_list = words.split(' ')
third_letters = [word[2] for word in words_list]
ans = ' '.join(third_letters)
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: n y t x l
Q3: [EOQ]
Ans: n y t x l
----
Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Melissa Vicente Gao Teng Abdou" and concatenate them using a space.
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
words = "Melissa Vicente Gao Teng Abdou"
words_list = words.split(' ')
third_letters = [word[2] for word in words_list]
ans = ' '.join(third_letters)
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: l c o n d
Q3: [EOQ]
Ans: l c o n d
----
Descriptio: %s
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
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(dev_inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
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
cot_rollout()

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
    parser.add_argument("--num_dev_examples", type=int, default=len(dev_inputs))
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
    print("Dev examples:", len(dev_inputs))

    dev_inputs = dev_inputs[: args.num_dev_examples]
    dev_labels = dev_labels[: args.num_dev_examples]

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
    elif args.inference_strategy == "auto_cot_consistent":
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
