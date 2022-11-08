from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
import time
from utils import get_few_shot_prompt
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

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


def few_shot(N=10, temperature=0.3):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)['input_ids']))

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, temperature=temperature, quote='---', n=1)
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
01/08/2007
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
Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
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
v e m t d
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
r a i u o
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
Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
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
05/31/2021
----
Desciption: %s
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
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
        pdb.set_trace()
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer contains the translated sequence.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer contains the translated sequence.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
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


few_shot_cot(temperature=0.4)
# few_shot(N=20)
# auto_cot(temperature=0.3)