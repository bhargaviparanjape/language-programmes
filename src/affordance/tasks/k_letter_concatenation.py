from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

# Get data
import urllib.request
url = 'https://raw.githubusercontent.com/allenai/DecomP/main/datasets/letter_cat/n5_eg100_pos2_space.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
dev_inputs = [d['question'] for d in data['1']['qa_pairs']]
dev_labels = [d['answer']['spans'][0] for d in data['1']['qa_pairs']]

url = 'https://raw.githubusercontent.com/allenai/DecomP/main/datasets/letter_cat/n4_eg100_pos2_space.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
train_inputs = [d['question'] for d in data['1']['qa_pairs']]
train_labels = [d['answer']['spans'][0] for d in data['1']['qa_pairs']]

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

def k_letter_concatenation():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Take the letters at position 3 of the words in "Maya Sylvie Campbell Hussein Francisco" and concatenate them using a space.
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
%s""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


def human_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Take the letters at position 3 of the words in "Noel Jenny Duong Awad" and concatenate them using a space.
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
%s""" % x for x in chunk]
        return gpt3(prompts)


    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.split('\n')[-1].strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("Pattern based decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


def cot_rollout():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
        prompts = ["""Take the last letters of the words in "Augusta Ada King" and concatenate them using a space. 
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
%s""" % x for x in chunk]
        return gpt3(prompts)


    perf_array = []
    runs = 2
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.split('\n')[-1].replace('Answer: ', '') for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


def automatic_decomposition():
    decomp_prompt = "I want to break down the task of taking the letters at position 3 of words in a list of words and concatenate them using a space. I want to break this task into individual steps."
    io_pairs = """Input: "Jin Wilson Ruiz Manna"
Output: n l i n
Input: "Raja Salim Haruna Qian"
Output: j l r a
Input: "Kirill Rajesh Ahmed Dutta"
Output: r j m t
Input: "Samir Joel Rodriguez Parker" 
Output: m e d r
Input: "Haji Abraham Barman White"
Output: j r r i"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_kletter_concatenation_fn(decomposition, batch_size=10):
        decomposition = '1.'+ decomposition
        last_n = int(re.findall(r'(\d+)\.', decomposition)[-1])
    #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = ['''Take the letters at position 3 of words in a list of words and concatenate them using a space. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise.''' % (decomposition, x) for x in chunk]
                out.extend(gpt3(prompts))
            return out
        return decomposition_fn


    labs, subset = get_subset(dev_inputs, labels=dev_labels, n=100)
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print('Decomposition', z)
        fn = get_kletter_concatenation_fn(decomposition, batch_size=20)
        this_preds = fn(subset)
    #     pp = np.array([1 if 'contains an anachronism' in x.lower() else 0 for x in this_preds])
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
    print(accs)
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.mean(accs))

cot_rollout()



