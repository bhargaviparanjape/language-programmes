import argparse
from collections import Counter
import pdb
import re
import time

import datasets
import numpy as np
from prompt_library import (
    few_shot_retrieval_prompt,
    llm_similar_tasks,
    random_tasks,
    similar_auto_breakdowns,
    similar_tasks,
)
from sequential_interpreter import TopDownVisitorBeta
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (
    OpenAIModel,
    cache_dir,
    chunks,
    get_answer,
    get_autocot_answer,
    get_few_shot_prompt,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "hindu_knowledge", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology."

io_pairs = [
    (
        """Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)""",
        """A: Flute""",
    ),
    (
        """Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga""",
        """A: Treta Yuga""",
    ),
    (
        """Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama""",
        """A: Shatrughna""",
    ),
]


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


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=200, quote="---", n=1
        )
        prompts = [
            """Q: In Hinduism, which musical instrument is most associated with the god Krishna?
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
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)["input_ids"]))

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """%s\
%s"""
            % (few_shot_prompt, x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1:
# Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
# Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: [compare] What is the final answer?
# Unknown
# Q4: [EOQ]
# Ans: Unknown
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
# Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
# Q1: [tag] What are the entities in this sentence?
# #1:
# President George H. W. Bush
# Gulf War
# Q2: [search] When was President George H. W. Bush president?
# #2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
# Q3: [search] When was the Gulf War fought?
# #3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
# #4: [subquestion] Could these entities have co-existed and interacted based on this information?
# Yes. Their time periods intersect.
# Q5: [generate answer] Is this an anachronism?
# #5: No
# Q6: [EOQ]
# Ans: No
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
# Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
# Q1: [tag] What are the entities in this sentence?
# #1:
# Kurt Cobain
# "Twin Peaks"
# Q2: [search] When did television show "Twin Peaks" air?
# #2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
# Q3: [search] When did Kurt Cobain live?
# #3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
# Q4: [subquestion] Could these entities have co-existed and interacted based on this information?
# No. Musician  Kurt Cobain could not have starred in Twin Peaks.
# Q5: [generate answer] Is this an anachronism?
# #5: Yes
# Q6: [EOQ]
# Ans: Yes
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#   choice: Anjalikastra
#   choice: Narayanastra
#   choice: Agneyastra
#   choice: Brahmastra
# Q1: [subquestion] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
# #1: No.
# Q2: [search] In the Hindu epic Ramayana, who is the main villain?
# #2: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q3: [compare] Which option is the answer in #3 most similar to?
# #3: Brahmastra
# Q4: [EOQ]
# Ans: Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1:
# Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: [generate answer] What is the final answer?
# Florida, Missouri
# Q4: [EOQ]
# Ans: Florida, Missouri
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_retrieval_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Hindu Knowledge"
    task_description = "(Hindu Knowledge) Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices."

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
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

    def predict_complete(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        outputs = gpt3(prompts)
        completed_outputs = [
            interpreter.complete_program(prefix, output) for prefix, output in zip(prompts, outputs)
        ]
        return completed_outputs

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology.
Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)
The final answer should be one of the provided choices.
A: Let's think step-by-step.

First, we need to recall what Hinduism is. Hinduism is a religion that originated in India. It is based on the belief in reincarnation and the principle of karma.

Next, we need to think about the god Krishna. Krishna is a popular god in Hinduism. He is known as the "god of love" and is often portrayed playing a flute.

Based on this information, we can conclude that the most likely answer is "flute."

The final answer is "Flute".
----
Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology.
Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama
The final answer should be one of the provided choices.
A: Let's think step-by-step.

First, we need to identify the Hindu epic Ramayana. This is an ancient Indian epic poem that tells the story of Rama, an avatar of the god Vishnu.

Next, we need to identify the character Lavanasura. This is a demon king who is defeated by Rama in the story.

Finally, we need to identify the character who killed Lavanasura. This was Shatrughna, Rama's brother.

The final answer is "Shatrughna".
----
Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology.
Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga
The final answer should be one of the provided choices.
A: Let's think step-by-step.

The Dvapara Yuga is the third era in the Hindu cosmology, and it is preceded by the second era. The second era is called Treta Yuga.

The final answer is "Treta Yuga"."""

auto_cot_cleaned_prompt = """Q: In Hinduism, which musical instrument is most associated with the god Krishna?
  choice: Sitar
  choice: Veena
  choice: Flute
  choice: Tabla (drum)
The final answer should be one of the provided choices.
A: Let's think step-by-step.

First, we need to identify which god is most associated with music. In Hinduism, that would be Krishna.
Then, we need to identify which musical instrument is most associated with Krishna. Based on the choices provided, the answer would be the flute.

The best answer choice is flute.
----
Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology.
Q: In Hinduism, the era known as Dvapara Yuga is preceded by which era?
  choice: Krita Yuga
  choice: Treta Yuga
  choice: Kali Yuga
  choice: Satya Yuga
The final answer should be one of the provided choices.
A: Let's think step-by-step.

The Dvapara Yuga is the third age of the four ages of the Hindu cycle of time, so it is preceded by the first two ages, the Krita Yuga and the Treta Yuga.

The best answer choice is Krita Yuga.
----
Hindu Knowledge: Answer these multiple-choice questions about Hindu mythology.
Q: In the Hindu epic Ramayana, which character killed Lavanasura?
  choice: Lakshmana
  choice: Shatrughna
  choice: Bharata
  choice: Rama
The final answer should be one of the provided choices.
A: Let's think step-by-step.

First, we need to know who Lavanasura is. He's a demon king who was troubling the people of Ayodhya.
Next, we need to decide which of the provided choices is most likely to have killed Lavanasura. Lakshmana is Rama's brother, so he's a likely candidate. Shatrughna is another one of Rama's brothers, so he's also a likely candidate. Bharata is Rama's cousin, so he's a possible candidate.
Finally, we need to choose the most likely candidate. Based on the information given, it seems most likely that Lakshmana killed Lavanasura.

The best answer choice is Lakshmana.
----
"""


def auto_cot(
    temperature=0.3,
    model_name="text-davinci-002",
    predict=True,
    use_corrected=False,
    self_consistency=False,
):
    global auto_cot_corrected_prompt
    global auto_cot_cleaned_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=1000, temperature=0.7, quote="---", n=1
        )
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer should be one of the provided choices.\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Hindu Knowledge\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer should be one of the provided choices.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=500, temperature=temperature, quote="---", n=1
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer should be one of the provided choices.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1] :])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is """, x).span(0)[-1] :]]
                            )
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
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x, "The best answer choice is ") for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=2048,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def search_query(answer):
        lines = answer.strip().split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if "[search]" in line:
                query_no = re.search("[0-9]+", line.split()[0]).group(0)
                new_lines.append(line)
                query = line[re.search("\[search\]", line).span(0)[1] :].strip()
                search_snippet = search(query, top_k=1)
                new_lines.append("#%s: " % (query_no) + search_snippet)
                # skip a next line or add #2
                skip = True
            else:
                if skip:
                    skip = False
                    continue
                new_lines.append(line)
        return "\n".join(new_lines)

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=1000,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers = predict(
                "Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices.",
                x,
            )
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
                query_no = re.search("[0-9]+", last_question.split()[0]).group(0)
                q = re.search(r"Q%s: " % str(int(query_no) + 1), a)
                if q:
                    new_prompt = ex + "\n" + a[: q.span(0)[1]]
                    if len(tokenizer(new_prompt)["input_ids"]) + 1000 > 4097:
                        pdb.set_trace()
                        # Too much input.
                        new_x.append(ex)
                    else:
                        new_x.append(ex + "\n" + a[: q.span(0)[1]])
                else:
                    # No next question beyond the last search questions
                    new_x.append(ex + "\n" + a)
            # pdb.set_trace()
            new_answers.extend(
                predict_with_affordance(
                    "Answer these multiple-choice questions about Hindu mythology. The final answer should be one of the provided choices.",
                    new_x,
                )
            )
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
        gpt3 = OpenAIModel(
            model=model_name,
            max_length=2048,
            temperature=temperature,
            quote="In [1]:",
            n=1,
        )
        prompts = [few_shot_notebook_prompt % (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "").replace('"', "'") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        final_preds = []
        for pred in preds:
            answer_span = re.search("Out \[3\]", pred)
            if answer_span:
                final_preds.append(pred[answer_span.span(0)[1] :])
            else:
                final_preds.append("")
        final_preds = [x.strip() for x in final_preds]
        # preds = [pred[re.search("Out \[3\]", pred).span(0)[1]:] for pred in preds]
        perf_array.append(substring_match(labels, final_preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Hindu Knowledge"
    task_description = (
        "(Hindu Knowledge) Answer these multiple-choice questions about Hindu mythology."
    )

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
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    new_answer = interpreter.batch_visit([prompts[chunk_no]] * 5, ans_chunk)
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
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
                # time.sleep(30)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))

            # Report on interpreter performance
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / len(interpreter.execution_details)

        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "text-davinci-002",
            "text-davinci-003",
            "code-davinci-002",
            "code-cushman-001",
            "davinci-codex-002-msft",
        ],
        default="text-davinci-002",
    )
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument(
        "--inference_strategy",
        type=str,
        choices=[
            "dummy",
            "few_shot",
            "auto_cot",
            "cot_rollout",
            "few_shot_cot",
            "nl_program",
        ],
        default="few_shot",
    )
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action="store_true")
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"],
        default="fixed",
    )

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[: args.num_dev_examples]
    labels = labels[: args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)["input_ids"]))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(
            args.temperature,
            args.model_name,
            predict=True,
            use_corrected=False,
            self_consistency=False,
        )
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name, strategy=args.selection_strategy)
    elif args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            self_consistency=args.self_consistency,
            strategy=args.selection_strategy,
        )
