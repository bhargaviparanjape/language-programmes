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
                            similar_auto_breakdowns, similar_tasks,
                            few_shot_retrieval_prompt, few_shot_code_prompt, 
                            few_shot_arithmetic_prompt, few_shot_string_prompt)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta


d = datasets.load_dataset('bigbench', 'sentence_ambiguity', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']
labels = [l[0] for l in labels]

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = """Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False"."""

io_pairs = [("""Claim: LeBron James is universally considered the best football player of all time.
True or False?""",
"False"),
("""Claim: The cornea of your eye is likely where the light receptors are.
True or False?""",
"False"),
("""Claim: Few animals live more than 200 years, but none of these have fins.
True or False?""",
"False"),
("""Claim: The United States has never had a unanimously elected president.
True or False?""",
"False"),
("""Claim: The North Pole is likely not located on a continent.
True or False?""",
"True"),
("""Claim: Earth's oceans may include the Pacific, Atlantic, Indian, and Arctic.
True or False?""",
"True"),
("""Claim: Australia had more Cricket World Cup wins than the West Indies as of 2021.
True or False?""",
"True")]

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
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  temperature=temperature, max_length=200, quote='---', n=1)
        prompts = ["""Claim: LeBron James is universally considered the best football player of all time.
True or False?
False
----
Claim: The cornea of your eye is likely where the light receptors are.
True or False?
False
----
Claim: Few animals live more than 200 years, but none of these have fins.
True or False?
False
----
Claim: Earth's oceans may include the Pacific, Atlantic, Indian, and Arctic.
True or False?
True
----
Claim: The United States has never had a unanimously elected president.
True or False?
False
----
Claim: Australia had more Cricket World Cup wins than the West Indies as of 2021.
True or False?
True
----
%s""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

auto_cot_corrected_prompt = """(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: LeBron James is universally considered the best football player of all time.
True or False?
A: Let's think step-by-step.

The first step is to break down the claim into smaller pieces. In this case, we have:
-LeBron James is universally considered the best football player of all time.
The next step is to look at each individual part of the claim and see if there is evidence to support it. In this case, we have:
-LeBron James is universally considered the best football player of all time.
There is no evidence to support this claim. LeBron James is not universally considered the best football player of all time. Therefore, the claim is false.

The final answer is False.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: The cornea of your eye is likely where the light receptors are.
True or False?
A: Let's think step-by-step.

First, what are light receptors? They are cells that receive light and convert it into electrical signals.
Second, where are they located in the eye? They are located in the retina, which is at the back of the eye.
Third, what is the cornea? The cornea is the clear, outer layer of the eye.
Based on these facts, we can conclude that the claim is false. The light receptors are not located in the cornea.

The final answer is False.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Few animals live more than 200 years, but none of these have fins.
True or False?
A: Let's think step-by-step.

First, we need to determine what the claim is actually saying. In this case, the claim is saying that there are few animals that live more than 200 years, and none of those animals have fins.
Now, we need to think about whether or not this claim is true. In order to do that, we need to gather evidence.
First, let's look at the evidence for the first part of the claim, that there are few animals that live more than 200 years. According to this article, the oldest recorded age for various animals are as follows:
- humans: 122 years
- elephants: 86 years
- parrots: 82 years
- tortoises: 152 years
Based on this evidence, it appears that the claim is true - there are few animals that live more than 200 years.

Now, let's look at the evidence for the second part of the claim, that none of those animals have fins. According to this article, the oldest recorded age for various animals are as follows:
- humans: 122 years
- elephants: 86 years
- parrots: 82 years
- tortoises: 152 years
- salmon: 7 years
Based on this evidence, it appears that the claim is false - there is at least one animal that lives more than 200 years and has fins (salmon).

The final answer is False.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: The North Pole is likely not located on a continent.
True or False?
A: Let's think step-by-step.

There is no land at the North Pole - it is permanently covered in ice. So the North Pole is not located on a continent. Hence, the claim is true.

The final answer is True.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Earth's oceans may include the Pacific, Atlantic, Indian, and Arctic.
True or False?
A: Let's think step-by-step.

First, we need to determine what the claim is saying. The claim is saying that the oceans on Earth may include the Pacific, Atlantic, Indian, and Arctic oceans.
Next, we need to see if there is evidence to support this claim. From looking at a map, we can see that there is indeed evidence to support this claim, as the oceans mentioned are all present on Earth.
Therefore, we can conclude that the claim is true.

The final answer is True.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Australia had more Cricket World Cup wins than the West Indies as of 2021.
True or False?
A: Let's think step-by-step.

Australia has won the Cricket World Cup 5 times. The West Indies have won the Cricket World Cup 2 times. As of 2021, Australia has more Cricket World Cup wins than the West Indies. Therefore, the claim is true.

The final answer is True.
----
"""

def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    global auto_cot_corrected_prompt
    auto_cot_prompt = ""
    description = "(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence."
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Sentence ambiguity\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
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
            for x in tqdm(chunks(inputs, 10)):
                # x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                time.sleep(10)
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
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
# Q4: [subquestion] Could these entities have co-existed and interacted based on this information?
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
    task_name = "Sentence Ambiguity"
    task_description = "(Sentence Ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of 'True' or 'False'."

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
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict("""Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".""", x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def search_query(answer):
        lines = answer.strip().split("\n")
        print(lines)
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
        pdb.set_trace()
        return "\n".join(new_lines)

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers = predict("Determine if the claim is true or false.", x)
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [search_query("Q1:" + a) for a in answers]
            pdb.set_trace()
            x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Determine if the claim is true or false.", x))
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):

    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_description, io_pairs, N=6)

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    
    global few_shot_cot_prompt
    task_name = "Sentence Ambiguity"
    task_description = "(Sentence Ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of 'True' or 'False'."

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

    interpreter = TopDownVisitorBeta()

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            pdb.set_trace()
            time.sleep(30)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--selection_strategy", type=str, choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"], default="fixed")
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
        auto_cot(args.temperature, args.model_name, predict=True, use_corrected=False, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name, args.selection_strategy)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)