from enum import auto
from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
import time
import openai
import sys
import subprocess

d = datasets.load_dataset('bigbench', 'simple_text_editing', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

task_description="Simple text editing: Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."

io_pairs = [("""Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: add 'bcd' after each 'a' in the entire text""",
"""Turing wabcds one of the people who worked on the first computers. He creabcdted the theoreticabcdl Turing mabcdchine in 1936. The mabcdchine wabcds imabcdginabcdry, but it included the ideabcd of abcd computer prograbcdm."""),
("""Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: replace all occurrencies of the character 'o' with 'a' in the entire text""",
"""Turing was ane af the peaple wha warked an the first camputers. He created the thearetical Turing machine in 1936. The machine was imaginary, but it included the idea af a camputer pragram."""),
("""Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: append the sentence 'SENTENCE TO BE APPENDED.' to the text""",
"""Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program. SENTENCE TO BE APPENDED.""")]

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


def simple_text_editing(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = ["""Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: add 'bcd' after each 'a' in the entire text
Edited:
Turing wabcds one of the people who worked on the first computers. He creabcdted the theoreticabcdl Turing mabcdchine in 1936. The mabcdchine wabcds imabcdginabcdry, but it included the ideabcd of abcd computer prograbcdm.
----
Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: replace all occurrencies of the character 'o' with 'a' in the entire text
Edited:
Turing was ane af the peaple wha warked an the first camputers. He created the thearetical Turing machine in 1936. The machine was imaginary, but it included the idea af a camputer pragram.
----
Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: append the sentence 'SENTENCE TO BE APPENDED.' to the text
Edited:
Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program. SENTENCE TO BE APPENDED.
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


def human_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, quote='---', n=1)
        prompts = ["""Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: add 'bcd' after each 'a' in the entire text
Step1: Split the text into a list of words.
#1: ['Turing', 'was', 'one', 'of', 'the', 'people', 'who', 'worked', 'on', 'the', 'first', 'computers.', 'He', 'created', 'the', 'theoretical', 'Turing', 'machine', 'in', '1936.', 'The', 'machine', 'was', 'imaginary,', 'but', 'it', 'included', 'the', 'idea', 'of', 'a', 'computer', 'program.']
Step2: For each word, add "bcd" after every occurrence of "a"
#2: [Turing', 'wabcds', 'one', 'of', 'the', 'people', 'who', 'worked', 'on', 'the', 'first', 'computers.', 'He', 'creabcdted', 'the', 'theoreticabcdl', 'Turing', 'mabcdchine', 'in', '1936.', 'The', 'mabcdchine', 'wabcds', 'imabcdginabcdry,', 'but', 'it', 'included', 'the', 'ideabcd', 'of', 'abcd', 'computer', 'prograbcdm.]
Step3: Merge words back into a sentence with spaces.
#3: Turing wabcds one of the people who worked on the first computers. He creabcdted the theoreticabcdl Turing mabcdchine in 1936. The mabcdchine wabcds imabcdginabcdry, but it included the ideabcd of abcd computer prograbcdm.
Edited: 
Turing wabcds one of the people who worked on the first computers. He creabcdted the theoreticabcdl Turing mabcdchine in 1936. The mabcdchine wabcds imabcdginabcdry, but it included the ideabcd of abcd computer prograbcdm.
----
Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: replace all occurrencies of the character 'o' with 'a' in the entire text
Step1: Split the text into a list of words.
#1: ['Turing', 'was', 'one', 'of', 'the', 'people', 'who', 'worked', 'on', 'the', 'first', 'computers.', 'He', 'created', 'the', 'theoretical', 'Turing', 'machine', 'in', '1936.', 'The', 'machine', 'was', 'imaginary,', 'but', 'it', 'included', 'the', 'idea', 'of', 'a', 'computer', 'program.']
Step2: For each word, replace all occurrences of "o" with "a".
#2: ['Turing', 'was', 'ane', 'af', 'the', 'peaple', 'wha', 'warked', 'an', 'the', 'first', 'camputers.', 'He', 'created', 'the', 'thearetical', 'Turing', 'machine', 'in', '1936.', 'The', 'machine', 'was', 'imaginary,', 'but', 'it', 'included', 'the', 'idea', 'af', 'a', 'camputer', 'pragram.']
Step3: Merge words back into a sentence with spaces.
#3: Turing was ane af the peaple wha warked an the first camputers. He created the thearetical Turing machine in 1936. The machine was imaginary, but it included the idea af a camputer pragram.
Edited: 
Turing was ane af the peaple wha warked an the first camputers. He created the thearetical Turing machine in 1936. The machine was imaginary, but it included the idea af a camputer pragram.
----
Edit the text 'Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program.' with the following edit operation and the scope of the edit and replicate the remaining text as is: append the sentence 'SENTENCE TO BE APPENDED.' to the text
Step1: Split the text into a list of words.
#1: ['Turing', 'was', 'one', 'of', 'the', 'people', 'who', 'worked', 'on', 'the', 'first', 'computers.', 'He', 'created', 'the', 'theoretical', 'Turing', 'machine', 'in', '1936.', 'The', 'machine', 'was', 'imaginary,', 'but', 'it', 'included', 'the', 'idea', 'of', 'a', 'computer', 'program.']
Step2: List words in the phrase 'SENTENCE TO BE APPENDED.'
#2: ['SENTENCE', 'TO', 'BE', 'APPENDED.']
Step3: Append the second list to the first.
#3: ['Turing', 'was', 'one', 'of', 'the', 'people', 'who', 'worked', 'on', 'the', 'first', 'computers.', 'He', 'created', 'the', 'theoretical', 'Turing', 'machine', 'in', '1936.', 'The', 'machine', 'was', 'imaginary,', 'but', 'it', 'included', 'the', 'idea', 'of', 'a', 'computer', 'program.', 'SENTENCE', 'TO', 'BE', 'APPENDED.']
Step4: Merge words back into a sentence with spaces.
Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program. SENTENCE TO BE APPENDED.
Edited:
Turing was one of the people who worked on the first computers. He created the theoretical Turing machine in 1936. The machine was imaginary, but it included the idea of a computer program. SENTENCE TO BE APPENDED.
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
        preds = [x.strip().split("\n") for x in answers]
        perf_array.append(token_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_decomp(N=10, temperature=0.3):
    # The other ways to do this is to ask what functions to use and given these functions, come up with required steps. 
    decomp_prompt = "%s I want to break this task into individual steps to make it easier to solve." % task_description
    decompositions = propose_decomposition(decomp_prompt, io_pairs, N)
    def get_decomp_func(decomposition, batch_size=10):
            decomposition = '1.'+ decomposition
            last_n = int(re.findall(r'(\d+)\.', decomposition)[-1])
            decomposition += '\n%s. The final answer should be the edited text.' % (last_n + 1)
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
    subset = [ex.replace("\nEdited:", "") for ex in subset]
    for z, decomposition in enumerate(decompositions):
        print('Decomposition:', z)
        print(decompositions[z])
        fn = get_decomp_func(decomposition, batch_size=20)
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
    runs = 2
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            answers.extend(predict(task_description, x))
            # pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer should be the edited text.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer should be the edited text.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nEdited:", "") for ex in x]
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

    def string_edit(sequence, edit_operation):
        # Use Codex code generation and execution
        # Sequencce is either a list of sentences or words.
        edit_op = "Python program to " + edit_operation.strip()
        code_input = "x = " + sequence
        response = openai.Edit.create(model="code-davinci-edit-001", input=code_input, instruction=edit_op, temperature=0, top_p=1)
        #program_output = exec(response['choices'][0]['text'])
        code = response['choices'][0]['text']
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return result.stdout

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
            x = [ex.replace("\nEdited:", "") for ex in x] 
            answers = predict(task_description, x)
            # Two programs that work, #1 string_split followed by string_edit #2 direct string edit.
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            new_x = []
            for i, a in enumerate(answers):
                a = a.strip()
                affordance_outputs = []
                if ("string split" in a) and ("string edit" in a):
                    # Only split into sentences and ignore words.
                    editing_input = a[re.search("#1: ", a).span(0)[1]:re.search("Q2: ", a).span(0)[0]].strip()
                    affordance_output = string_edit(editing_input, x[i].split(":")[1])
                    affordance_outputs.append(affordance_output)
                    ex = x[i] + "\nQ1:" + a[:re.search("#2: ", a).span(0)[1]] + affordance_output
                    time.sleep(60)
                else:
                    ex = x[i]
                new_x.append(ex)
            pdb.set_trace()
            # x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance(task_description, new_x))
        # preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        preds = [x.strip() for x in new_answers]
        perf_array.append(substring_match(labels, preds))
        time.sleep(60)
        print(perf_array)
    print("Few-shot Affordance COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

# simple_text_editing(temperature=0.3)
# auto_cot(temperature=0.3)
# few_shot_cot(temperature=0.3)
# auto_decomp(10, 0.3)
affordance(temperature=0.3)