from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match, get_few_shot_prompt, get_answer

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
from utils import get_few_shot_prompt
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta
import time
from collections import Counter


d = datasets.load_dataset('bigbench', 'physics', cache_dir=cache_dir)
inputs = d['train']['inputs'] + d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['train']['targets'] + d['validation']['targets']
labels = [l[0] for l in labels]
print(len(inputs))

task_description = "Identify the physics formula that would be most useful for finding the answer to each of the following word problems."

io_pairs = [("""Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t""",
"F = q * E"),
("""
Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt""",
"Q = m * c * dT"),
("""Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt""",
"F = m * a")]

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
    few_shot_prompt = get_few_shot_prompt(inputs, [[l] for l in labels], n=N)
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

def physics():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t
A:
F = q * E
----
Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt
A:
Q = m * c * dT
----
Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt
A:
F = m * a
----
%s""" % x for x in chunk]
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


few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.  
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
  choice: Unknown
  choice: Five million
Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon? 
#1:
Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin.
Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon.
Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: No. The question is too specific
Q3: [compare] What is the final answer?
#3: Unknown
Q4: [EOQ]
Ans: Unknown
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
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
Q5: [generate output] Is this an anachronism?
#5: No
Q6: [EOQ]
Ans: No
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
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
Q5: [generate output] Is this an anachronism?
#5: Yes
Q6: [EOQ]
Ans: Yes
----
Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [search] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q2: [compare] Which option is the answer in #3 most similar to?
#2: Brahmastra
Q3: [EOQ]
Ans: Brahmastra
----
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Q1: [search] Where was Mark Twain born?
#1:
Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: Yes. The answer is Florida, Missouri
Q3: [compare] What is the final answer?
#3: Florida, Missouri
Q4: [EOQ]
Ans: Florida, Missouri
----
Desciption: %s 
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.\n\n\n", "") for ex in x]
            answers.extend(predict("Given a physics word problem, choose which physics formula from among the choices is most helpful in solving the word problem. The final answer should be one of the choices.", x))
            time.sleep(30)
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    description = """Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems."""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nThe final answer should be one of the choices.\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
    print(auto_cot_prompt)
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nThe final answer should be one of the choices.\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
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
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    global few_shot_cot_prompt
    task_name = "Physics Formulas"
    task_description = "Identify the physics formula that would be most useful for finding the answer to each of the following word problems."

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
                x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
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
            # x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.", "") for ex in x]
            x = [ex.replace("\n\n\nQ: ", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            prompts, answer = predict(task_description, x) 
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            preds = [get_answer(x.strip()) for x in answers]
            time.sleep(60)
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# auto_decomp(10, 0.3)
# affordance(temperature=0.3)
# dynamic_few_shot_cot(temperature=0.3, strategy="random")
few_shot_cot()
# few_shot(N=5, temperature=0.3)
# auto_cot()
#nl_program(temperature=0.3)
