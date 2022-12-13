from enum import auto
from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match, get_answer
import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
import time
import openai
import sys
import subprocess
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta
from collections import Counter
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns
import sys
sys.path.append("/gscratch/zlab/bparan/projects/cascades/src/guidance")
import guidance

d = datasets.load_dataset('bigbench', 'simple_text_editing', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

task_description="(Simple text editing) Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."

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


def few_shot_cot(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 2
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            answers.extend(predict(task_description, x))
            # pdb.set_trace()
            time.sleep(60)
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


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):

    task_name = "Simple Text Editing"
    task_description = "Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."

    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)


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
            x = [ex.replace("\nEdited:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    global few_shot_cot_prompt
    task_name = "Simple Text Editing"
    task_description = "Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."

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
        runs = 5
        batch_size = 10
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nEdited:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for ans in answer_set:
                    new_answer = interpreter.batch_visit(prompts, ans)
                    new_answer = [get_answer(program) for program in new_answer]
                    for enum, pred in enumerate(new_answer):
                        if pred is not None:
                            result_counter[enum].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            time.sleep(60)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def measure_task_selection_accuracy():
    task_name = "Simple Text Editing"
    task_description = "Edit the text in quotes based the edit operation provided at the end. Only edit relevant portions and replicate the remaining text as is."
    from prompt_library import task_selection_accuracy
    task_selection_accuracy(["date_understanding", "language_games", "k_letter_concatenation"], task_name, task_description, io_pairs)


few_shot_notebook_prompt = """In [1]:
import os
import sys
import numpy as np
import re
from sympy.solvers import solve
from sympy import Symbol, Eq
import math
from sympy import simplify
import numpy as np
import cvxpy as cp
import statistics
from serpapi import GoogleSearch
from utils import string_compare

In [1]:
input_text = "Take the letters at position 3 of the words in 'Ibrahim Francois Pei Shu Ngo' and concatenate them using a space."
def get_sentence(text):
    text_start = re.search("words in '", text).span(0)[1]
    text_end = re.search("' and'", text).span(0)[0]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Ibrahim Francois Pei Shu Ngo"

In [2]: 
input_text = "Ibrahim Francois Pei Shu Ngo"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]

In [3]:
input_tokens = ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
def get_nth_char_in_list(token_list, n):
    char_list = [token[n-1] for token in token_list]
    return char_list
get_nth_char_in_list(input_tokens, 3)

Out [3]:
["r", "a", "i", "o", "u"]

In [4]: 
char_list = ["r", "a", "i", "o", "u"]
def merge_chars(char_list):
    return " ".join(char_list)
merge_chars(char_list)

Out [4]:
r a i u o

In [5]:
final_answer = r a i u o
print("Ans: ",final_answer)

Out [5]:
r a i u o

In [1]:
input_text = "Take the letters at position 3 of the words in 'Savita Saeed Ramos Sato Yadav' and concatenate them using a space."
def get_sentence(text):
    text_start = re.search("words in '", text).span(0)[1]
    text_end = re.search("' and'", text).span(0)[0]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Savita Saeed Ramos Sato Yadav"

In [2]: 
input_text = "Savita Saeed Ramos Sato Yadav"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]

In [3]:
input_tokens = ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
def get_nth_char_in_list(token_list, n):
    char_list = [token[n-1] for token in token_list]
    return char_list
get_nth_char_in_list(input_tokens, 3)

Out [3]:
["v", "e", "m", "t", "d"]

In [4]: 
char_list = ["v", "e", "m", "t", "d"]
def merge_chars(char_list):
    return " ".join(char_list)
merge_chars(char_list)

Out [4]:
v e m t d

In [5]:
final_answer = v e m t d
print("Ans: ",final_answer)

Out [5]:
Ans: v e m t d

In [1]:
input_text = "Translate English into Pig Latin. (English) Sami made his way across the bar and hugged Layla."

Out [1]:
def get_sentence(text):
    text_start = re.search("English into Pig Latin. \(English\) ", text).span(0)[1]
    text_end = re.search("\.", text).span(0)[1]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Sami made his way across the bar and hugged Layla"

In [2]:
input_text = "Sami made his way across the bar and hugged Layla"
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Sami", "made", "his", "way", "across", "the", "bar", "and", "hugged", "Layla"]

In [3]:                                  
input_tokens = ["Sami", "made", "his", "way", "across", "the", "bar", "and", "hugged", "Layla"]
def get_pig_latin(token_list):
    vowels = ["a", "e", "i", "o", "u", "y"]
    pig_latin_tokens = []
    for token in token_list:
        if token[0] in vowels:
            pig_latin_tokens.append(token + "yay")
        else:
            pig_latin_tokens.append(token[1:] + token[0] + "ay")
    return pig_latin_tokens
get_pig_latin(input_tokens)

Out [3]:
['Amisay', 'ademay', 'ishay', 'ayway', 'acrossyay', 'hetay', 'arbay', 'andyay', 'uggedhay', 'Aylalay']

In [4]:
pig_latin_tokens = ['amisyay', 'ademyay', 'ishyay', 'ayway', 'rossacay', 'ethyay', 'aryabay', 'andyay', 'uggedhay', 'aylaLay']
def merge_tokens(token_list):
    return " ".join(token_list)
merge_tokens(pig_latin_tokens)

Out [4]:
'Amisay ademay ishay ayway acrossyay hetay arbay andyay uggedhay Aylalay'

In [5]:
final_answer = 'amisyay ademyay ishyay ayway rossacay ethyay aryabay andyay uggedhay aylaLay'
print("Ans: ",final_answer)

Out [5]:
Ans: amisyay ademyay ishyay ayway rossacay ethyay aryabay andyay uggedhay aylaLay

In [1]:
input_text = "Translate English into Pig Latin. (English) Tom is the most popular boy in school."

Out [1]:
def get_sentence(text):
    text_start = re.search("(English) ", text).span(0)[1]
    text_end = re.search("\.", text).span(0)[1]
    return text[text_start:text_end]
get_sentence(input_text)

Out [1]:
"Tom is the most popular boy in school."

In [2]:
input_text = "Tom is the most popular boy in school."
def get_tokens(text):
    tokens = text.split()
    return tokens
get_tokens(input_text)

Out [2]:
["Tom", "is", "the", "most", "popular", "boy", "in", "school."]

In [3]:
input_tokens = ["Tom", "is", "the", "most", "popular", "boy", "in", "school."]
def isVowel(letter):
    if letter in ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'):
        return True
    else:
        return False
def translate(token_list):
    return ' '.join([token[1:] + token[0:1] + 'ay' if isVowel(token[0]) else token[2:] + token[0:2] + 'ay' for token in token_list])
translate(input_tokens)

Out [3]:
"Omtay isay ethay ostmay opularpay oybay inay oolschay."

In [4]:
final_answer = Omtay isay ethay ostmay opularpay oybay inay oolschay.
print("Ans: ",final_answer)

Out [4]:
"Ans: Omtay isay ethay ostmay opularpay oybay inay oolschay."

In [1]:
input_text = "%s"
"""

def notebook(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='In [1]:', n=1)
        prompts=[few_shot_notebook_prompt% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nEdited:", "") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# simple_text_editing(temperature=0.3)
# auto_cot(temperature=0.3)
# few_shot_cot(temperature=0.3, model_name="text-davinci-003")
# auto_decomp(10, 0.3)
# affordance(temperature=0.3)
# dynamic_few_shot_cot(temperature=0.3, strategy="llm_similar")
# prompt_guidance()
nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed")
# notebook(temperature=0.3, model_name="code-davinci-002")



