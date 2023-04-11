import datasets
import numpy as np
from tqdm import tqdm
from utils import OpenAIModel, cache_dir, chunks

d = datasets.load_dataset("bigbench", "novel_concepts", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]
# print(len(inputs))


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


def novel_concepts():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """Let's do some find-the-common-concept problems. In these problems, your goal is to identify the underlying concept or theme that relates the things listed. Make sure to answer carefully.
What do the following have in common? 1) rooks in chess 2) the Tower of London
A:
They are both castles.
----
Let's do some find-the-common-concept problems. In these problems, your goal is to identify the underlying concept or theme that relates the things listed. Make sure to answer carefully.
What do the following have in common? 1) bumble bees 2) 01010101 3) race cars
A:
They all have stripes.
----
Let's do some find-the-common-concept problems. In these problems, your goal is to identify the underlying concept or theme that relates the things listed. Make sure to answer carefully.
What do the following have in common? 1) coconut 2) paint 3) motor
A:
They all are related to oil.
----
Let's do some find-the-common-concept problems. In these problems, your goal is to identify the underlying concept or theme that relates the things listed. Make sure to answer carefully.
What do the following have in common? 1) bumble bees 2) cell phones 3) exciting news
A:
They all can make a buzz.
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
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


novel_concepts()
