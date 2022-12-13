from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, search, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
import time
import ast
from utils import get_few_shot_prompt
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


d = datasets.load_dataset('bigbench', 'hindu_knowledge', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']

labels = [l[0] for l in labels]

task_description = "Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology."

io_pairs=[("""Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)""",
"""A: Flute"""),
("""Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga""",
"""A: Treta Yuga"""),
("""Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama""",
"""A: Shatrughna""")]

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


def hindu_knowledge():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-001",  max_length=200, quote='---', n=1)
        prompts = ["""Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)
A:
Flute
----
Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga
A:
Treta Yuga
----
Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama
A:
Shatrughna
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

# hindu_knowledge()

few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.  
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
  choice: Unknown
  choice: Five million
Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon? 
#1: 
Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: No. The question is too specific
Q3: What is the final answer?
Unknown
Q4: [EOC]
Unknown
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
Q1: [tag] What are the entities in this sentence?
#1: 
President George H. W. Bush
Gulf War
Q2: [search] When was President George H. W. Bush president?
#2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
Q3: [search] When was the Gulf War fought?
#3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
#4: Could these entities have co-existed based on thier time periods alone?
Yes. Their time periods intersect.
Q5: Is this an anachronism?
#5: No
Q6: [EOC]
No
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
Q1: [tag] What are the entities in this sentence?
#1: 
Kurt Cobain
"Twin Peaks"
Q2: [search] When did television show "Twin Peaks" air?
#2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
Q3: [search] When did Kurt Cobain live?
#3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
Q4: Could these entities have co-existed based on this information?
No. Musician  Kurt Cobain could not have starred in Twin Peaks.
Q5: Is this an anachronism?
#5: Yes
Q6: [EOC]
Yes
----
Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [search] In the Hindu epic Ramayana, who is the main villain? 
#1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q2: [compare] Which option is the answer in #3 most similar to?
#2: Brahmastra
Q3: [EOC]
Brahmastra
----
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Q1: [search] Where was Mark Twain born?
#1: 
Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: Yes. The answer is Florida, Missouri
Q3: What is the final answer?
Florida, Missouri
Q4: [EOC]
Florida, Missouri
----
Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Hindu epic Ramayana, the main villain was a devotee of which deity?
  choice: Indra
  choice: Vishnu
  choice: Brahma
  choice: Shiva
Q1: [subquestion] Can this question be answered step-by-step?
#1: Yes.
Q2: [search] In the Hindu epic Ramayana, who is the main villain? 
#2: Ravana is the main antagonist of the Hindu Epic, the Ramayana. 
Q3: [search] Ravana was a devotee of which deity?
#3: Ravana, was an ardent devotee of Lord Shiva, is depicted and described as a great scholar,a brahman,a capable ruler and a maestro of the Veena.
Q4: [compare] Which option is the answer in #3 most similar to?
#4: Shiva
Q5: [EOC]
Shiva
----
Desciption: %s 
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict("Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices.", x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer should be one of the provided choices.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer should be one of the provided choices.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
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


def affordance(temperature = 0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def search_query(answer):
        lines = answer.strip().split("\n")
        new_lines = []
        skip=False
        for line in lines:
            if "[search]" in line:
                query_no = re.search('[0-9]+', line.split()[0]).group(0)
                new_lines.append(line)
                query = line[re.search("\[search\]", line).span(0)[1]:].strip()
                search_snippet = search(query, top_k=1)
                new_lines.append("#%s: "%(query_no) + search_snippet)
                # skip a next line or add #2
                skip=True
            else:
                if skip:
                    skip=False
                    continue
                new_lines.append(line)
        return "\n".join(new_lines)

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers = predict("Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices.", x)
            affordance_outputs = [search_query("Q1:" + a) for a in answers]
            # Find the last search term and resume execution after that.
            # last_questions = [re.findall("Q[0-9]+: \[search\]", a)[-1] for a in affordance_outputs]
            # query_nos = [re.search('[0-9]+', question.split()[0]).group(0) for question in last_questions]
            # next_questions = [re.search(r"Q%s: "%str(int(q) + 1), a) for q, a in zip(query_nos,affordance_outputs)]
            new_x = []
            for ex, a in zip(x, affordance_outputs):
                last_question = re.findall("Q[0-9]+: \[search\]", a)
                if len(last_question) > 0:
                    last_question = last_question[-1]
                else:
                    # No search attempted.
                    new_x.append(ex)
                    continue
                query_no = re.search('[0-9]+', last_question.split()[0]).group(0)
                q = re.search(r"Q%s: "%str(int(query_no) + 1), a)
                if q:
                    new_prompt = ex + "\n" + a[:q.span(0)[1]]
                    if len(tokenizer(new_prompt)['input_ids']) + 1000 > 4097:
                        pdb.set_trace()
                        # Too much input.
                        new_x.append(ex)
                    else:
                        new_x.append(ex + "\n" + a[:q.span(0)[1]])
                else:
                    # No next question beyond the last search questions
                    new_x.append(ex + "\n" + a)
            # pdb.set_trace()
            new_answers.extend(predict_with_affordance("Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices.", new_x))
        preds = [x.strip() for x in new_answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Affordance performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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
input_text = \"""Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)\"""

In [2]:
input_query = "In Hinduism, which musical instrument is most associated with the god Krishna?"
search_result = GoogleSearch(input_query)
search_result

Out [2]:
The venu is associated with the Hindu god Krishna, who is often depicted playing it. This kind of flute is mainly used in South India. Vishnu is portrayed as Venugopala, playing the flute of creation.

In [3]:
search_result = "The venu is associated with the Hindu god Krishna, who is often depicted playing it. This kind of flute is mainly used in South India. Vishnu is portrayed as Venugopala, playing the flute of creation."
choices = ["Sitar", "Veena", "Flute", "Table (drum)"]
closest_choice = string_compare(search_result, choices)
closest_choice

Out [3]:
Flute

In [1]: 
input_text = \"""Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga\"""

In [2]:
input_query = "In Hinduism, the era known as Dvapara Yuga is preceded by which era?"
search_result = GoogleSearch(input_query)
search_result

Out [2]:
Dvapara Yuga ( a.k.a. Dwapara Yuga), in Hinduism, is the third and third best of the four yugas (world ages) in a Yuga Cycle, preceded by Treta Yuga and followed by Kali Yuga.

In [3]:
search_result = "Dvapara Yuga ( a.k.a. Dwapara Yuga), in Hinduism, is the third and third best of the four yugas (world ages) in a Yuga Cycle, preceded by Treta Yuga and followed by Kali Yuga."
choices = ["Krita Yuga", "Treta Yuga", "Kali Yuga", "Satya Yuga"]
closest_choice = string_compare(search_result, choices)
closest_choice

Out [3]:
Treta Yuga

In [1]: \"""Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama\"""

In [2]:
input_query = "In the Hindu epic Ramayana, which character killed Lavanasura?"
search_result = GoogleSearch(input_query)
search_result

Out [2]:
He is slain by Shatrughna, the youngest brother of Rama, in the Hindu epic Ramayana. Lavana is slain by Shatrughna.

In [3]:
search_result = "He is slain by Shatrughna, the youngest brother of Rama, in the Hindu epic Ramayana. Lavana is slain by Shatrughna."
choices = ["Lakshmana", "Shatrughna", "Bharata", "Rama"]
closest_choice = string_compare(search_result, choices)
closest_choice

Out [3]:
Shatrughna

In [1]:
input_text = \"""%s\"""
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
            x = [ex.replace("\nA:", "").replace('"', "'") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        final_preds = []
        for pred in preds:
            answer_span = re.search("Out \[3\]", pred)
            if answer_span:
                final_preds.append(pred[answer_span.span(0)[1]:])
            else:
                final_preds.append("")
        final_preds = [x.strip() for x in final_preds]
        # preds = [pred[re.search("Out \[3\]", pred).span(0)[1]:] for pred in preds]
        perf_array.append(substring_match(labels, final_preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

# auto_cot(temperature=0.3)
few_shot_cot(temperature=0.3, model_name="code-davinci-002")
# affordance(temperature=0.3)
# notebook(temperature=0.3, model_name="code-davinci-002")