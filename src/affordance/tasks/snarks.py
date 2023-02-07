import argparse
import ast
import json
import pdb
import re
import time
import os
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_answer, get_autocot_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import urllib.request
from collections import Counter

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks,
                            few_shot_retrieval_prompt, few_shot_code_prompt, 
                            few_shot_arithmetic_prompt, few_shot_string_prompt)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta

d = datasets.load_dataset('bigbench', 'snarks', cache_dir=cache_dir)
inputs = d['validation']['inputs'] + d['train']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets'] + d['train']['targets']
labels = [l[0] for l in labels]
print(len(inputs))


train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = """Sarcasm is defined as language designed to attack, provide comic relief, or both using indirect semantic dependencies, exaggerations or subtext. Given two statements, identify which one is sarcastic."""

io_pairs = [
    ('Q: Which statement is sarcastic? (a) Soggy fries covered in bacon grease sounds unhealthy. (b) Soggy fries covered in bacon grease sounds delicious.\nA:', '(b)'),
    ('Q: Which statement is sarcastic? (a) The real tragedy here is that someone is buying a fraud (b) The real tragedy here is that someone is buying a Mustang\nA:', '(b)'),
    ('Q: Which statement is sarcastic? (a) Remind me, how do you print a legal-sized piece of paper? (b) Remind me, how do you print a blank piece of paper?\nA:', '(b)'),
    ('Q: Which statement is sarcastic? (a) But only 10000 economists said it will help the economy... (b) But only 2 economists said it will help the economy...\nA:', '(a)'),
    ('Q: Which statement is sarcastic? (a) how dare you use violence!!! (b) how dare you use logic!!!\nA:', '(b)')]

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)['input_ids']))

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts = ["""%s\
%s""" % (few_shot_prompt, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
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


auto_cot_corrected_prompt = """"""
auto_cot_cleaned_prompt = """"""

def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    global auto_cot_corrected_prompt
    global auto_cot_cleaned_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    
    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Snarks\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(dev_inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1]:])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update([pred[re.search("""The final answer is """, x).span(0)[-1]:]])
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
            print("Run %d"%run)
            answers = []
            for x in tqdm(chunks(dev_inputs, 10)):
                # x = [ex.replace("Q: Which statement is sarcastic? ", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x) for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action='store_true')
    parser.add_argument("--selection_strategy", type=str, choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"], default="fixed")

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    dev_inputs = inputs[:args.num_dev_examples]
    dev_labels = labels[:args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(args.temperature, args.model_name, predict=False, use_corrected=False, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name, strategy=args.selection_strategy)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency, strategy=args.selection_strategy)