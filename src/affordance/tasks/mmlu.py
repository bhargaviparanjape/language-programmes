import ast
import json
import os
import pdb
import re
import time
import csv
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_few_shot_prompt,
                   get_subset, gpt3, propose_decomposition,
                   propose_instruction, search, substring_match)
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns
from sequential_interpreter import TopDownVisitor
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

dataset_name = "virology"
task_name = "College computer science"
task_description = "Answer the following multiple-choice questions about college-level Computer Science, which tests for specialized knowledge and concepts like Algorithms, systems, graphs, recursion, etc."

mmlu_namespace = os.path.join(cache_dir, 'mmlu', 'data')
data_files = {"train": os.path.join(mmlu_namespace, "dev/{0}_dev.csv".format(dataset_name)), \
"validation": os.path.join(mmlu_namespace, "test/{0}_test.csv".format(dataset_name))}
train_data = [row for row in csv.reader(open(data_files["train"]))]
train_inputs = [row[0] + "\n" + "\n".join(["%s) %s"%(letter, choice) for letter, choice in zip(["A", "B", "C", "D"], row[1:-1])]) for row in train_data]
train_labels = [row[-1] for row in train_data]

data = [row for row in csv.reader(open(data_files["validation"]))]
inputs = [row[0] + "\n" + "\n".join(["%s) %s"%(letter, choice) for letter, choice in zip(["A", "B", "C", "D"], row[1:-1])]) for row in data]
labels = [row[-1] for row in data]

io_pairs = [(inp, lab) for inp, lab in zip(train_inputs[:5], train_labels[:5])]
# inputs = d['train']['inputs'] + d['validation']['inputs']
# # inputs = [x.split('\n')[0] for x in inputs]
# labels = d['train']['targets'] + d['validation']['targets']
io_prompt = "\n----\n".join([(inp + "\nAns:\n" + lab) for (inp, lab) in io_pairs])

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def few_shot(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = [io_prompt + """\n----\n%s\nAns:""" % x for x in chunk]
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
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=5)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=5)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=5)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=5)

    interpreter = TopDownVisitor(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            pdb.set_trace()
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# nl_program(temperature=0.3, model_name="text-davinci-002", strategy="llm_similar")



few_shot(temperature=0.3)
