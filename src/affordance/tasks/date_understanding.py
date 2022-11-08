from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
from utils import get_few_shot_prompt
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# international_phonetic_alphabet_transliterate too 
d = datasets.load_dataset('bigbench', 'date_understanding', cache_dir=cache_dir)
inputs = d['validation']['inputs']
labels = d['validation']['targets']
labels = [l[0] for l in labels]

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = "Date understanding task: Find the required date in MM/DD/YYYY using information about related events and dates in the input."

io_pairs = [("""Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?""",
"""A:
05/01/2021"""),
("""Q: Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date today in MM/DD/YYYY?""",
"""A:
10/15/1924"""),
("""Q: Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?""",
"""A:
02/16/2010"""),
("""Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date a month ago in MM/DD/YYYY?""",
"""A:
04/06/2002"""),
("""Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date a month ago in MM/DD/YYYY?""",
"""A:
12/07/2018""")]

def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count

def token_match(labels, predictions):
    # Not all tokens are equal, it makes sense to only compare tokens that are generated at the end of the sequence. 
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower().split():
            correct += 1
        count += 1
    return (1.0*correct)/count

def substring_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower():
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


def human_decomp(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts = ["""Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
Q1: Yesterday was April 30, 2021. Reformat this into MM/DD/YYYY.
#1: 04/30/2021
Q2: How can you find today's day?
#2: Today is one day after yesterday.
Q2: What day is today in MM/DD/YYYY?
#2: 05/30/2021
Q3: [EOQ]
05/01/2021
----
Q: Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date today in MM/DD/YYYY?
Q1: What day is tomorrow?
#1: 10/16/1924
Q2: How can you find today's day?
#2: Today is one day before tomorrow.
Q2: What day is today in MM/DD/YYYY?
#2: 10/15/1924
Q3: [EOQ]
10/15/1924
----
Q: Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?
Q1: What was the date of Jane's first visit to the bookstore?
#1: 16th October of 2009.
Q2: Reformat this into MM/DD/YYYY.
#2: 10/16/2009
Q3: Which month will be Jane's 5th visit? 
#3: Jane's 5th visit will be 4 months from the first visit.
Q4: Jane's 5th visit in MM/DD/YYYY?
#4: 02/16/2010
Q5: [EOQ]
02/16/2010
----
Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date a month ago in MM/DD/YYYY?
Q1: What is the date 10 years ago in MM/DD/YYYY?
#1: 05/06/1992
Q2: What is the date today in MM/DD/YYYY?
#2: 05/06/2002
Q3: What day is the date a month ago in MM/DD/YYYY?
#3: 04/06/2002
Q4: [EOQ]
04/06/2002
----
Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date a month ago in MM/DD/YYYY?
Q1: What is the first day of 2009 in MM/DD/YYYY?
#1: 01/01/2019
Q2: What date will the first Monday of 2019 fall on if 01/01/2009 is a Tuesday.
#2: 01/07/2019
Q3: What date is one month before today (06/01/2009)?
#3: 12/07/2018
Q4: [EOQ]
12/07/2018
----
%s
Q1:""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(token_match(labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_decomp(N=10, temperature=0.3):
    # The other ways to do this is to ask what functions to use and given these functions, come up with required steps. 
    decomp_prompt = "%s I want to break this task into individual steps to make it easier to solve." % task_description
    decompositions = propose_decomposition(decomp_prompt, io_pairs, N)
    def get_decomp_func(decomposition, batch_size=10):
            decomposition = '1.'+ decomposition
            last_n = int(re.findall(r'(\d+)\.', decomposition)[-1])
            decomposition += '\n%s. The final answer should be in the MM/DD/YYYY format.' % (last_n + 1)
            def decomposition_fn(sentences):
                gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=temperature, quote='---', n=1)
                out = []
                for chunk in chunks(sentences, batch_size):
                    prompts = ['''%s Using the following steps will help.
Steps:
%s
----
%s
Also show how you used the steps provided to arrive at the answer.''' % (task_description, decomposition, x) for x in chunk]
                    out.extend(gpt3(prompts))
                return out
            return decomposition_fn
        
    labs, subset = get_subset(inputs, labels=labels, n=len(inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print('Decomposition:', z)
        print(decompositions[z])
        fn = get_decomp_func(decomposition, batch_size=20)
        subset = [ex.replace("\nA:", "") for ex in subset]
        this_preds = fn(subset)
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
        print(pp.mean())
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.std(accs))


def auto_cot():
    auto_cot_prompt = ""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer should be in the MM/DD/YYYY format.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.3, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer should be in the MM/DD/YYYY format.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

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


def few_shot_cot(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=["""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
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
Q1:"""% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict("Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.", x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot(N=5, temperature=0.7)
# human_decomp(temperature=0.7)
auto_decomp(N=20, temperature=0.7)
# auto_cot()
# few_shot_cot(temperature=0.3)