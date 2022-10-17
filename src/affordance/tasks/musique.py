import pdb
from  utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir
import datasets
import numpy as np
from tqdm import tqdm
import os
import re

# Musique
data_files = {split:os.path.join(cache_dir, 'musique', 'data', 'musique_full_v1.0_%s.jsonl'%split) for split in ['train', 'dev']}
d = datasets.load_dataset('json', data_files=data_files)
dev_inputs = [ex['question'] for ex in d['dev']][:500]
dev_labels = [ex['answer'] for ex in d['dev']][:500]

# Human Decomp 
def musique():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
        prompts = ["""Question: When did the country where Beterverwagting is located become a member of caricom?
Answer: 1 August 1973
----
Question: When did the roof gardens above the district where George Palao is born, open to the public?
Answer: 1980s
----
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Answer: Ming general Qu Neng
----
Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Answer: near Salem
----
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Answer: Hassan Gouled Aptidon
----
Question: What subject was studied in David Sassoon's birthplace?
Answer: Islamic mathematics
----
Question: Who is the spouse of the creator of The Neverwhere?
Answer: Amanda Palmer
----
Question: When did the torch arrive in the birthplace of Han Kum-Ok?
Answer: April 28
----
Question: Who sings Meet Me in Montana with the performer of I Only Wanted You?
Answer: Dan Seals
----
Question: When did the torch reach the mother country of Highs and Lows?
Answer: May 2
----
Question: %s
Answer: """ % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(preds, dev_labels)])
        perf_array.append(pp.mean())
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


# Human Decomp 
def subquestion_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
        prompts = ["""Question: When did the country where Beterverwagting is located become a member of caricom?
Follow up question: Beterverwagting >> country
Intermediate answer: Guyana
Follow up question: when did Guyana became a member of caricom
Intermediate answer: 1 August 1973
So the final answer is 1 August 1973
----
Question: When did the roof gardens above the district where George Palao is born, open to the public?
Follow up question: George Palao >> place of birth
Intermediate answer: Kensington
Follow up question: when did the roof gardens above Kensington open to the public
Intermediate answer: 1980s
So the final answer is 1980s
----
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Follow up question: Li Shuoxun >> place of birth
Intermediate answer: Sichuan
Follow up question: Who was ordered to force a Tibetan assault into Sichuan ?
Intermediate answer: Ming general Qu Neng
So the final answer is Ming general Qu Neng
----
Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Follow up question: Which place is A City Decides in?
Intermediate answer: St. Louis
Follow up question: Where is Washington University in St. Louis located?
Intermediate answer: Missouri
Follow up question: where does the merrimack river start in Missouri
Intermediate answer: near Salem
So the final answer is near Salem
----
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Follow up question: National Highway 6 >> country
Intermediate answer: Djibouti
Follow up question: Who was the first president of Djibouti ?
Intermediate answer: Hassan Gouled Aptidon
So the final answer is Hassan Gouled Aptidon
----
Question: What subject was studied in David Sassoon's birthplace?
Follow up question: David Sassoon >> place of birth
Intermediate answer: Baghdad
Follow up question: What subject was studied in Baghdad ?
Intermediate answer: Islamic mathematics
So the final answer is Islamic mathematics
----
Question: Who is the spouse of the creator of The Neverwhere?
Follow up question: The Neverwhere was made by whom?
Intermediate answer: Neil Gaiman
Follow up question: Neil Gaiman >> spouse
Intermediate answer: Amanda Palmer
So the final answer is Amanda Palmer
----
Question: When did the torch arrive in the birthplace of Han Kum-Ok?
Follow up question: Han Kum-Ok >> place of birth
Intermediate answer: Pyongyang
Follow up question: When did the torch arrive in Pyongyang ?
Intermediate answer: April 28
So the final answer is April 28
----
Question: Who sings Meet Me in Montana with the performer of I Only Wanted You?
Follow up question: I Only Wanted You >> performer
Intermediate answer: Marie Osmond
Follow up question: who sings meet me in montana with Marie Osmond
Intermediate answer: Dan Seals
So the final answer is Dan Seals
----
Question: When did the torch reach the mother country of Highs and Lows?
Follow up question: What is the country Highs and Lows is from?
Intermediate answer: Hong Kong
Follow up question: When did the torch arrive in Hong Kong ?
Intermediate answer: May 2
So the final answer is May 2
----
Question: %s""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(preds, dev_labels)])
        perf_array.append(pp.mean())
        # perf_array.append(exact_match(dev_labels, preds))
    print("Subquestion decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


# subquestion_decomposition()


def automatic_decomposition():
    decomp_prompt = "I want to break down the task of answering complex questions into individual steps."
    io_pairs = """Input: When did the country where Beterverwagting is located become a member of caricom?
Output: 1 August 1973
----
Input: When did the roof gardens above the district where George Palao is born, open to the public?
Output: 1980s
----
Input: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Output: Ming general Qu Neng
----
Input: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Output: near Salem
----
Input: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Output: Hassan Gouled Aptidon"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_musique_fn(decomposition, batch_size=10):
        decomposition = '1.'+ decomposition
        last_n = int(re.findall(r'(\d+)\.', decomposition)[-1])
    #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = ['''Answer the following complex questions. Using the following steps will help.
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
        fn = get_musique_fn(decomposition, batch_size=20)
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
