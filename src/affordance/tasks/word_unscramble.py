import argparse
from collections import Counter
import json
import pdb
import re
import time

import datasets
import numpy as np
from prompt_library import llm_similar_tasks, random_tasks, similar_auto_breakdowns, similar_tasks
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

io_pairs = [
    ("Q: The word untamqu is a scrambled version of the English word", "A: quantum"),
    ("Q: The word mnoeg is a scrambled version of the English word", "A: gnome"),
    ("Q: The word agknkbo is a scrambled version of the English word", "A: bangkok"),
    ("Q: The word mgiaein is a scrambled version of the English word", "A: imagine"),
    (
        "Q: The word eacperlfi is a scrambled version of the English word",
        "A: fireplace",
    ),
]

task_description = "Unscramble the given word into a word in English."


d = datasets.load_dataset("bigbench", "word_unscrambling", cache_dir=cache_dir)
inputs = d["validation"]["inputs"][:500]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"][:500]
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
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=200, quote="---", n=1
        )
        prompts = [
            """Q: The word untamqu is a scrambled version of the English word
A:
quantum
----
Q: The word mnoeg is a scrambled version of the English word
A:
gnome
----
Q: The word agknkbo is a scrambled version of the English word
A:
bangkok
----
Q: The word mgiaein is a scrambled version of the English word
A:
imagine
----
Q: The word eacperlfi is a scrambled version of the English word
A:
fireplace
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
            time.sleep(10)
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


few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007 in MM/DD/YYYY
#1: 01/01/2007
Q2: [arithmetic] What date is one week from 01/01/2007?
#2: 01/08/2007
Q3: [EOQ]
Ans: 01/08/2007
----
Description: Translate English into Pig Latin.
Input: (English) Sami made his way across the bar and hugged Layla.
Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
#1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
Q4: [EOQ]
Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
----
Description: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
#1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["v", "e", "m", "t", "d"]
Q3: [string merge] Concatenate #2 with spaces
#3: "v e m t d"
Q4: [EOQ]
Ans: v e m t d
----
Desciption: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
#1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["r", "a", "i", "o", "u"]
Q3: [string merge] Concatenate #2 with spaces
#3: "r a i u o"
Q4: [EOQ]
Ans: r a i u o
----
Description: Translate English into Pig Latin.
Input: (English) Tom is the most popular boy in school.
Q1: [string split] What are the words in "Tom is the most popular boy in school."?
#1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
Q4: [EOQ]
Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
----
Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
#1: 06/01/2021
Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: [arithmetic] What date is 24 hours later than today?
#3: 05/01/2021
Q4: [EOQ]
Ans: 05/31/2021
----
Desciption: %s
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Word Unscrambling"
    task_description = "(Word Unscrambling) Unscramble the given word into a word in English."

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
            model="text-davinci-002",
            max_length=2048,
            temperature=temperature,
            quote="---",
            n=1,
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
    runs = 2
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Unscramble the given word into a word in English.
Q: The word untamqu is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1) The first letter is 'u', so the word might start with 'un-' or 'u'.
2) The second letter is 'n', so the word might start with 'un-'.
3) The third letter is 't', so the word might start with 'un-t'.
4) The fourth letter is 'a', so the word might be 'un-ta-'.
5) The fifth letter is 'm', so the word might be 'un-ta-m'.
6) The sixth letter is 'q', so the word might be 'un-ta-mq'.
7) The seventh letter is 'u', so the word might be 'un-ta-mqu'.
The word is 'un-ta-mqu', so the unscrambled word is 'quantum'.

The final answer is 'quantum'
----
Unscramble the given word into a word in English.
Q: The word mnoeg is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1) The first letter is 'm', so the word might start with 'm'.
2) The second letter is 'n', so the word might start with 'n'.
3) The third letter is 'o', so the word might start with 'o' or have 'no','mo' in it. It could also have a 'nom'.
4) The fourth letter is 'e', so the word might have 'me', 'ne', 'em', 'en' in it. It could also have 'ome' or 'one' in it.
5) The last letter is 'g'. We could combine 'g' with 'nom' and 'ome' to give us 'gnome'.

The final answer is 'gnome'.
----
Unscramble the given word into a word in English.
Q: The word agknkbo is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1) The first letter is 'a', so the word might start with 'a'.
2) The second letter is 'g', so the word might have 'ag' or 'ga' in it.
3) The third letter is 'k', so the word might  have 'ak'. 'ka' in it.
4) The fourth letter is 'n', so the word might be 'gan', 'nag' in it.
5) The fifth letter is another 'k', maybe it has 'kang'.
6) The sixth letter is 'b', so the word might have 'bang'.
7) The seventh letter is 'o', so the word might contain 'ok' or 'kok'.
Combining 'bang' and 'kok' gives us 'bangkok'.

The final answer is 'bangkok'
----
Unscramble the given word into a word in English.
Q: The word mgiaein is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1. The first letter is 'm'. The word could start with 'm'.
2. The second letter is 'g'. The word could being with 'g'.
3. The third letter is 'i'. The word could be have a 'gi', 'mi', 'im' or 'ig'
4. The fourth letter is 'a'. The word could have 'aim', 'gam' or 'ima' in it.
5. The fifth letter is 'e'. The word could have 'game' or 'image' in it.
6. The sixth letter is 'n'. The word could have 'imagen' in it.
7. The seventh letter is 'i'. The word is definitely 'imagine'.

The final answer is 'imagine'
----
"""

auto_cot_cleaned_prompt = """Unscramble the given word into a word in English.
Q: The word untamqu is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1) The first letter is 'u', so the word might start with 'un-' or 'u'.
2) The second letter is 'n', so the word might start with 'un-'.
3) The third letter is 't', so the word might start with 'un-t'.
4) The fourth letter is 'a', so the word might be 'un-ta-'.
5) The fifth letter is 'm', so the word might be 'un-ta-m'.
6) The sixth letter is 'q', so the word might be 'un-ta-mq'.
7) The seventh letter is 'u', so the word might be 'un-ta-mqu'.
The word is 'un-ta-mqu', so the unscrambled word is 'quantum'.

The final answer is 'quantum'
----
Unscramble the given word into a word in English.
Q: The word mnoeg is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1. The word "mnoeg" is a scrambled version of the English word "menge."
2. The word "menge" is a German word meaning "crowd" or "group."
3. Therefore, the unscrambled English word is "crowd."

The final answer is 'crowd'
----
Unscramble the given word into a word in English.
Q: The word agknkbo is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1. The word agknkbo is a scrambled version of the English word
2. The final answer is the unscrambled English word.
3. The word agknkbo is a scrambled version of the English word "book".

The final answer is 'book'
----
Unscramble the given word into a word in English.
Q: The word mgiaein is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1. The first letter is 'm'. This could be the beginning of the word 'magic' or 'magician'.
2. The second letter is 'g'. This could be the beginning of the word 'game' or 'garden'.
3. The third letter is 'i'. This could be the beginning of the word 'is' or 'in'.
4. The fourth letter is 'a'. This could be the beginning of the word 'am' or 'are'.
5. The fifth letter is 'e'. This could be the beginning of the word 'eat' or 'end'.
6. The sixth letter is 'n'. This could be the beginning of the word 'now' or 'not'.
7. The seventh letter is 'i'. This could be the beginning of the word 'it' or 'is'.
The word is 'magician'.

The final answer is 'magician'
----
Unscramble the given word into a word in English.
Q: The word eacperlfi is a scrambled version of the English word
The final answer is the unscrambled English word.
A: Let's think step-by-step.

1. The word eacperlfi is a scrambled version of the English word
2. The final answer is the unscrambled English word.
3. Therefore, the word eacperlfi is a scrambled version of the English word "fireplace."

The final answer is 'fireplace'
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
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.2, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer is the unscrambled English word.\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Word unscrambling\n\n")
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
            + """%s\nThe final answer is the unscrambled English word.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.2, quote="---", n=1)
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer is the unscrambled English word.\nA: Let's think step-by-step.\n"""
            % x
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
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                time.sleep(10)
            preds = [get_autocot_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def affordance():
    best_decomp = """Q1: [string split] What are the letters in "%s"?
#1: %s
Q2: [string permutation] What are the possible permutations of the letters in #1?
#2:"""

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=1000, temperature=0.4, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def string_permutation(sequence):
        import itertools

        permutations = list(itertools.permutations(sequence))
        return ["".join(p) for p in permutations]

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=1000, temperature=0.4, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            words = [ex.split()[3] for ex in x]
            letters = [json.dumps([l for l in w]) for w in words]
            x = [
                ex.replace("\nA:", "\n") + best_decomp % (w, l)
                for w, l, ex in zip(words, letters, x)
            ]
            predict("Unscramble the given word into a word in English.", x)
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_permutation(json.loads(l)) for l in letters]
            x = [ex + json.dumps(o) + "\nQ3:" for ex, o in zip(x, affordance_outputs)]
            pdb.set_trace()
            new_answers.extend(
                predict_with_affordance("Unscramble the given word into a word in English.", x)
            )
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# affordance()
# auto_cot()


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):
    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_description, io_pairs, N=6)

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
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Word Unscrambling"
    task_description = "(Word Unscrambling) Unscramble the given word into a word in English."

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
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
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
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
                # time.sleep(10)
                pdb.set_trace()
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))

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
