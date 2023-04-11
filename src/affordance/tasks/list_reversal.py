import json
import re
import urllib.request

import numpy as np
from tqdm import tqdm
from utils import OpenAIModel, chunks, get_subset, propose_decomposition

url = "https://raw.githubusercontent.com/allenai/DecomP/main/datasets/reverse/test_10_normal_words.json"

response = urllib.request.urlopen(url)
data = json.loads(response.read())
dev_inputs = [d["question"] for d in data["alg_qa"]["qa_pairs"]]
dev_labels = [d["answer"]["spans"][0] for d in data["alg_qa"]["qa_pairs"]]

url = "https://raw.githubusercontent.com/allenai/DecomP/main/datasets/reverse/test_4_normal_words.json"

response = urllib.request.urlopen(url)
data = json.loads(response.read())
train_inputs = [d["question"] for d in data["alg_qa"]["qa_pairs"]]
train_labels = [d["answer"]["spans"][0] for d in data["alg_qa"]["qa_pairs"]]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def list_reversal():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """Reverse the sequence "case, laptop, file, bin".
bin, file, laptop, case
----
Reverse the sequence "stamp, lipstick, key, mobile phone".
mobile phone, key, lipstick, stamp
----
Reverse the sequence "identity card, painkiller, wallet, rubbish".
rubbish, wallet, painkiller, identity card
----
Reverse the sequence "case, laptop, file, bin".
bin, file, laptop, case
----
Reverse the sequence "toothbrush, battery, umbrella, alarm clock".
alarm clock, umbrella, battery, toothbrush
----
Reverse the sequence "key, player, pen, driving licence".
driving licence, pen, player, key
----
Reverse the sequence "toothbrush, umbrella, magazine, chewing gum".
chewing gum, magazine, umbrella, toothbrush
----
Reverse the sequence "phone card, battery, key, packet".
packet, key, battery, phone card
----
Reverse the sequence "identity card, laptop, player, dictionary".
dictionary, player, laptop, identity card
----
Reverse the sequence "newspaper, button, mobile phone, tissue".
tissue, mobile phone, button, newspaper
----
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


# list_reversal()


def automatic_decomposition():
    decomp_prompt = "I want to break down the task of reversing a list of words. I want to break this task into individual steps."
    io_pairs = """Input: "stamp, lipstick, key, mobile phone".
Output: mobile phone, key, lipstick, stamp
Input: "identity card, painkiller, wallet, rubbish".
Output: rubbish, wallet, painkiller, identity card
Input: "case, laptop, file, bin".
Output: bin, file, laptop, case
Input: "toothbrush, battery, umbrella, alarm clock".
alarm clock, umbrella, battery, toothbrush
Input: "key, player, pen, driving licence".
Output: driving licence, pen, player, key"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_list_reversal_fn(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])

        #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """Reverse the sequence of words. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise."""
                    % (decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(dev_inputs, labels=dev_labels, n=50)
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition", z)
        fn = get_list_reversal_fn(decomposition, batch_size=20)
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


automatic_decomposition()
