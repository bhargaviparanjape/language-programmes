from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset

import datasets
import numpy as np
from tqdm import tqdm
import json

# Get data
import urllib.request
url = 'https://raw.githubusercontent.com/allenai/DecomP/main/datasets/letter_cat/n5_eg100_pos2_space.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())
inputs = [d['question'] for d in data['1']['qa_pairs']]
labels = [d['answer']['spans'][0] for d in data['1']['qa_pairs']]
