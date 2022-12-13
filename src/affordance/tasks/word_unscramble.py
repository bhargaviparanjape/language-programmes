from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks

io_pairs = [("Q: The word untamqu is a scrambled version of the English word", 
"A: quantum"),
("Q: The word mnoeg is a scrambled version of the English word", 
"A: gnome"),
("Q: The word agknkbo is a scrambled version of the English word", 
"A: bangkok"),
("Q: The word mgiaein is a scrambled version of the English word", 
"A: imagine"),
("Q: The word eacperlfi is a scrambled version of the English word", 
"A: fireplace")]

task_description = "Unscramble the given word into a word in English."


d = datasets.load_dataset('bigbench', 'word_unscrambling', cache_dir=cache_dir)
inputs =  d['validation']['inputs'][:500]
# inputs = [x.split('\n')[0] for x in inputs]
labels =  d['validation']['targets'][:500]
labels = [l[0] for l in labels]

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

def word_unscrambling():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-001",  max_length=200, quote='---', n=1)
        prompts = ["""Q: The word untamqu is a scrambled version of the English word 
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
"""% x for x in chunk]
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
    

# word_unscrambling()

few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
Description: Find the required date in MM/DD/YYYY using the given date.
Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007
#1: 01/01/2007
Q2: [string replace]
#2: What date is one week from 01/01/2007? 
#2: 01/08/2007
Q3: [EOQ]
01/08/2007
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
Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
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
v e m t d
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
r a i u o
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
Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
----
Description: Find the required date in MM/DD/YYYY using the given date. 
Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021
#1: 06/01/2021
Q2: 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: What date is 24 hours later than today?  
#3: 05/01/2021
Q4: [EOQ]
05/31/2021
----
Desciption: %s
Input: %s
"""


def few_shot_cot():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 2
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict("Unscrambling the given word into a word in English.", x))
        preds = [[y.strip() for y in x.split("\n")] for x in answers]
        perf_array.append(token_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

# few_shot_cot()


def auto_cot():
    auto_cot_prompt = ""
    description = "TASK: Unscrambling the given word into a word in English."
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


def affordance():

    best_decomp="""Q1: [string split] What are the letters in "%s"?
#1: %s
Q2: [string permutation] What are the possible permutations of the letters in #1?
#2:"""

    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_permutation(sequence):
        import itertools      
        permutations = list(itertools.permutations(sequence))
        return [''.join(p) for p in permutations]

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            words = [ex.split()[3] for ex in x]
            letters = [json.dumps([l for l in w]) for w in words]
            x = [ex.replace("\nA:", "\n") + best_decomp %(w, l) for w, l, ex in zip(words, letters, x)]
            answers = predict("Unscramble the given word into a word in English.", x)
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_permutation(json.loads(l)) for l in letters]
            x = [ex + json.dumps(o) + "\nQ3:" for ex, o in zip(x, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Unscramble the given word into a word in English.", x))
            pdb.set_trace()
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
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
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
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='In [1]:', n=1)
        prompts=[few_shot_notebook_prompt% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
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

# repeat_copy_logic(temperature=0.3)
# auto_cot(temperature=0.3)
# few_shot_cot(temperature=0.3)
# dynamic_few_shot_cot(temperature=0.3, strategy="similar")
notebook(temperature=0.3, model_name="code-davinci-002")