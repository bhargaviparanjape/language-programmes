from  utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel
import datasets
import numpy as np
from tqdm import tqdm

# Get data
d = datasets.load_dataset('bigbench', 'anachronisms')
inputs = d['train']['inputs'] + d['validation']['inputs']
inputs = [x.split('\n')[0] for x in inputs]
labels = np.array([int(x[0] == 'Yes') for x in d['train']['targets'] + d['validation']['targets']])


# Human Decomp 
def anachronism(chunk):
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
    prompts = ['''Given a sentence and the time periods of each entity in it, tell me if it could have happened or not.
Sentence: I wrote about shakespeare
Entities and dates:
I -> 21st century
Shakespeare -> 16th century
Could the sentence be true based on the dates alone: Yes
----
Sentence: Shakespeare wrote about me

Entities and dates:
Shakespeare -> 16th century
I -> 21st century

Could the sentence be true based on the dates alone: No
----
Sentence: %s''' % x for x in chunk]
    return gpt3(prompts)

perf_array = []
runs = 2
for run in range(runs): 
    print("Run %d"%run)
    answers = []
    for x in tqdm(chunks(inputs, 20)):
        answers.extend(anachronism(x))
    preds = np.array([int(x.endswith('No')) for x in answers])
    perf_array.append((preds == labels).mean())
print("Human Performance:")
print("Mean", np.mean(perf_array))
print("Std. Dev", np.mean(perf_array))
