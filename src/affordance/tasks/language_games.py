import argparse
import ast
import json
import pdb
import re
import time
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import urllib.request
from collections import Counter

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta

d = datasets.load_dataset('bigbench', 'language_games', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

io_pairs = [("(Pig Latin) Apsnay ouryay ingersfay.\
(English)",
"Snap your fingers."),
("(Pig Latin) Omtay akedbay anyay appleyay iepay orfay usyay.\
(English)",
"Tom baked an apple pie for us."),
("(Pig Latin) Ithay ityay ithway ayay ickstay.\
(English)",
"Hit it with a stick."),
("(Pig Latin) Amisay adhay efensiveday oundsway onyay ishay armsyay.\
(English)",
"Sami had defensive wounds on his arms."),
("(Pig Latin) Iyay ishway otay aystay inyay Australiayay.\
(English)",
"I wish to stay in Australia."),
("(English) Fast food is unhealthy.\
(Pig Latin)",
"Astfay oodfay isyay unhealthyay."),
("(English) Tom was a crazy kid.\
(Pig Latin)",
"Omtay asway ayay azycray idkay."),
("(English) Who is Tom talking about?\
(Pig Latin)",
"Owhay isyay Omtay alkingtay aboutyay?"),
("(English) Mary unrolled her sleeping bag.\
(Pig Latin)",
"Arymay unrolledyay erhay eepingslay agbay."),
("(English) Never before have I seen such strength.\
(Pig Latin)",
"Evernay eforebay avehay Iyay eensay uchsay engthstray.")]

task_description = "Language Games: Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa."

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0*correct)/count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)['input_ids']))

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = ["""%s\
%s""" % (few_shot_prompt, x) for x in chunk]
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
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def language_games():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English.
 (Pig Latin) Apsnay ouryay ingersfay.
 (English)
Snap your fingers.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English.
 (Pig Latin) Omtay akedbay anyay appleyay iepay orfay usyay.
 (English)
Tom baked an apple pie for us.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English.
 (Pig Latin) Ithay ityay ithway ayay ickstay.
 (English)
Hit it with a stick.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English.
 (Pig Latin) Amisay adhay efensiveday oundsway onyay ishay armsyay.
 (English)
Sami had defensive wounds on his arms.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English.
 (Pig Latin) Iyay ishway otay aystay inyay Australiayay.
 (English)
I wish to stay in Australia.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate English into Pig Latin.
 (English) Fast food is unhealthy.
 (Pig Latin)
Astfay oodfay isyay unhealthyay.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate English into Pig Latin.
 (English) Tom was a crazy kid.
 (Pig Latin)
Omtay asway ayay azycray idkay.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate English into Pig Latin.
 (English) Who is Tom talking about?
 (Pig Latin)
Owhay isyay Omtay alkingtay aboutyay?
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate English into Pig Latin.
 (English) Mary unrolled her sleeping bag.
 (Pig Latin)
Arymay unrolledyay erhay eepingslay agbay.
----
Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate English into Pig Latin.
 (English) Never before have I seen such strength.
 (Pig Latin)
Evernay eforebay avehay Iyay eensay uchsay engthstray.
----
%s
""" % x for x in chunk]
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
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007 in MM/DD/YYYY 
#1: 01/01/2007
Q2: What date is one week from 01/01/2007? 
#2: 01/08/2007
Q3: [EOQ]
Ans: 01/08/2007
----
Description: Translate English into Pig Latin.
Input: (English) Sami made his way across the bar and hugged Layla.
Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
#1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
Q4: [EOQ]
Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
----
Description: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
#1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["v", "e", "m", "t", "d"]
Q3: [string merge] Concatenate #2 with spaces
#3: "v e m t d"
Q4: [EOQ]
Ans: v e m t d
----
Desciption: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
#1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["r", "a", "i", "o", "u"]
Q3: [string merge] Concatenate #2 with spaces
#3: "r a i u o"
Q4: [EOQ]
Ans: r a i u o
----
Description: Translate English into Pig Latin.
Input: (English) Tom is the most popular boy in school.
Q1: [string split] What are the words in "Tom is the most popular boy in school."?
#1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
Q4: [EOQ]
Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
----
Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
#1: 06/01/2021
Q2: 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: What date is 24 hours later than today?  
#3: 05/01/2021
Q4: [EOQ]
Ans: 05/31/2021
----
Desciption: %s
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):

    global few_shot_cot_prompt
    task_name = "Language games"
    task_description = "(Language Games) Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa."

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Language Games: Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa.
(Pig Latin) Iyay ishway otay aystay inyay Australiayay.(English)
The final answer contains the translated sequence.
A: Let's think step-by-step.

"Iyay" is "I" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "Iyay" is just "I".

"ishway" is "Ish" + "w" + "ay". In Pig Latin, "ay" is added to the end of words that start with a consonant. Therefore, "Ishway" is just "wish".

"otay" is "o" + "t" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "otay" is just "to".

"aystay" is "ay" + "st" + "ay". In Pig Latin, "ay" is added to words that start with a consonant cluster. Therefore, "aystay" is just "stay".

"inyay" is "in" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "inyay" is just "in".

"Australiayay" is "Australia" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "Australiayay" is just "Australia".

The final answer is "I with to stay in Australia."
----
Language Games: Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa.
(Pig Latin) Amisay adhay efensiveday oundsway onyay ishay armsyay.(English)
The final answer contains the translated sequence.
A: Let's think step-by-step.

"Amisay" is "Ami" + "s" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "Amisay" is just "Sami".

"adhay" is "ad" + "h" + "ay". In Pig Latin, "ay" is added to the end of words that start with a consonant. Therefore, "adhay" is just "had".

"efensiveday" is "efensive" + "d" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "efensiveday" is just "defensive".

"oundsway" is "ounds" + "w" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "oundsway" is just "wounds".

"onyay" is "on" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "onyay" is just "on".

"ishay" is "is" + "h" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "ishay" is just "his".

"armsyay" is "arms" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "armsyay" is just "arms".

The final answer is "Sami had defensive wounds on his arms."
----
Language Games: Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa.
(Pig Latin) Ithay ityay ithway ayay ickstay.(English)
The final answer contains the translated sequence.
A: Let's think step-by-step.

"Ithay" is "It" + "h" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "Ithay" is just "Hit".

"ityay" is "it" + "yay". In Pig Latin, "yay" is added to the end of words that start with a vowel. Therefore, "Ithay" is just "it".

"ithway" is "ith" + "w" + "ay". In Pig Latin, "ay" is added to words that start with a consonant. Therefore, "ithway" is just "with".

"ayay" is "a" + "yay". In Pig Latin, "yay" is added to words that start with a vowel. Therefore, "ayay" is just "a".

"ickstay" is "ick" + "st" + "ay". In Pig Latin, "ay" is added to words that start with a consonant cluster. Therefore, "ickstay" is just "stick".

The final answer is "Hit it with a stick"
----
Language Games: Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa.
(Pig Latin) Omtay akedbay anyay appleyay iepay orfay usyay.(English)
The final answer contains the translated sequence.
A: Let's think step-by-step.

1. "Omtay" is "Tom" in Pig Latin.

2. "akedbay" is "Baked" in Pig Latin.

3. "anyay" is "Any" in Pig Latin.

4. "appleyay" is "Apple" in Pig Latin.

5. "iepay" is "Pie" in Pig Latin.

6. "orfay" is "For" in Pig Latin.

7. "usyay" is "Us" in Pig Latin.

The final answer is "Tom baked an apple pie for us."
"""


def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    global auto_cot_corrected_prompt
    auto_cot_prompt = auto_cot_corrected_prompt
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer contains the translated sequence.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Language games\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer contains the translated sequence.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer contains the translated sequence.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
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
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 5
        for run in range(runs): 
            print("Run %d"%run)
            answers = []
            for x in tqdm(chunks(inputs, 20)):
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character  = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers = predict("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x)
            affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x))
        # preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        preds = [x.strip() for x in new_answers]
        perf_array.append(substring_match(labels, preds))
        time.sleep(60)
        print(perf_array)
    print("Few-shot Affordance COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    global few_shot_cot_prompt
    task_name = "Language games"
    task_description = "(Language Games) Pig Latin looks just like English, but if an English word starts with a vowel you add -yay at the end of the word (ice -> iceyay); if it starts with consonants, you move them to end of the word and add -ay (prank -> ankpray); if it is capitalized, the Pig Latin word is capitalized as well (Maggie -> Aggiemay). Translate Pig Latin into English or viceversa."

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    new_answer = interpreter.batch_visit([prompts[chunk_no]]*5, ans_chunk)
                    new_answer = [get_answer(program) for program in new_answer]
                    for enum, pred in enumerate(new_answer):
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                time.sleep(60)
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 1
        for run in range(runs): 
            print("Run %d"%run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer  = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
                time.sleep(30)
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))

            # Report on interpreter performance
            positive_calls = [int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details]
            positive_rate = sum(positive_calls)/len(interpreter.execution_details)
            
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action='store_true')

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[:args.num_dev_examples]
    labels = labels[:args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(args.temperature, args.model_name, use_corrected=True, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)

    # few_shot_cot(temperature=0.4)
    # few_shot(N=20)
    # auto_cot(temperature=0.3)