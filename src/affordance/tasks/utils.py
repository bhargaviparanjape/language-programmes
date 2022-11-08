import datasets
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

import adatest
import pdb
import re
import json
import jsonlines
import seqio
import os
os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-bundle.crt"
from bigbench.bbseqio import tasks
vocabulary=seqio.SentencePieceVocabulary("/gscratch/zlab/bparan/projects/cascades/models/t5-spiece.model")
from sklearn.metrics import accuracy_score
from typing import List
import tqdm
import requests
subscription_key = '28dc412935f24fb9974d80c30915483a'
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}
from IPython.display import HTML

with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')
print(openai.api_key)
cache_dir = '/gscratch/zlab/bparan/projects/cascades/data'
import subprocess
import sys


class OpenAIModel(adatest.Model):
    def __init__(self, model="text-davinci-002", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1, logprobs=None):
        self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs
    def __call__(self, strings):
        resp = openai.Completion.create(
            model=self.model,
            prompt=strings,
            max_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.quote,
            logprobs = self.logprobs
        )
        return [x["text"] for x in resp['choices']]

gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='', n=1)

def propose_decomposition(decomp_prompt, io_pairs, n=20):
    # Default of 0.7 is being used.
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, quote='---', n=n)
    prompt = '''%s. Here are examples of input-output pairs for the task I'm trying to break down.
----
%s
----
Steps:
1.'''%(decomp_prompt, io_pairs)
    return gpt3(prompt)

def propose_instruction(instruct_prompt, io_pairs, n=20):
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=400, quote='---', n=n)
    prompt = '''%s. Here are examples of input-output pairs for this task.
----
%s
----
I can do this task by'''%(instruct_prompt, io_pairs)
    return gpt3(prompt)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_subset(inputs, labels, n=100):
    idxs = np.random.choice(len(inputs), n, replace=False)
    labs = np.array([labels[i] for i in idxs])
    subset = [inputs[i] for i in idxs]
    return labs, subset
    

def substring_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

import re
# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

def cleanhtml(raw_html):
  cleantext = re.sub(CLEANR, '', raw_html)
  return cleantext

def search(query, top_k=1):
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if "webPages" in search_results:
        snippets = [cleanhtml(v['name'] + " " + v['snippet']) for v in search_results['webPages']["value"]][:top_k]
    else:
        return ""
    return "\n".join(snippets)
    

def execute_code(code_snippet):
    # scope of variables to be defined here?
    # output = exec(code_snippet)
    result = subprocess.run([sys.executable, "-c", code_snippet], capture_output=True, text=True)
    return result

def generate_code(instruction):
    # Call Codex 002
    pass


def get_few_shot_prompt(inputs, labels, n=150):
    idx = np.random.randint(0, len(inputs), n)
    prompt_string = ""
    for id_, (inp, label) in enumerate(zip(inputs, labels)):
        if id_ not in idx:
            continue
        prompt_string += inp + "\n"
        prompt_string += label[0] + "\n"
        prompt_string += '----\n'
    return prompt_string

def edit_code(instructions, current_code):
    # Call Codex 001 (Edit) Model
    pass