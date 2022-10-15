import datasets
import os
import openai
import numpy as np
with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')

import adatest
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

with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')
print(openai.api_key)
cache_dir = '/gscratch/zlab/bparan/projects/cascades/data'


class OpenAIModel(adatest.Model):
    def __init__(self, model="text-davinci-002", quote="\"", temperature=0.7, top_p=1, max_length=30, n=1):
        self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
    def __call__(self, strings):
        resp = openai.Completion.create(
            model=self.model,
            prompt=strings,
            max_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.quote,
        )
        return [x["text"] for x in resp['choices']]

gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='', n=1)

def propose_decomposition(decomp_prompt, io_pairs, n=20):
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=400, quote='---', n=n)
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
