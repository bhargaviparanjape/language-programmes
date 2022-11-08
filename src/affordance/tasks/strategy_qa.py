from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, cache_dir, search, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re


d = datasets.load_dataset('bigbench', 'strategyqa', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [
("Q: Has Johns Hopkins University always treated subjects ethically?",
"No. Henrietta Lacks' DNA was used and replicated by Johns Hopkins University without her family's knowledge or approval. Henrietta Lacks' family medical history was released by Johns Hopkins University without their knowledge."),
("Q: Could Carl Friedrich Gauss speak to someone 100 miles away?",
"No. Carl Friedrich Gauss was born in 1777. Speaking to someone 100 miles away requires a telephone. The telephone was invented in 1876."),
("Q: Is calling ABBA the Swedish Beatles a preposterous claim?",
"Yes. ABBA was a Swedish band that had 1 Billboard number 1 hit and 4 top 10 hits. The Beatles had 20 Billboard number 1 hits and 34 top 10 hits."),
("Q: Did compact discs make computer gaming more popular?",
"Yes. Compact discs contained significantly more storage space than the previously popular floppy disc format. Gaming studios were therefore able to significantly improve the graphics, sounds, and features of their games to make them more immersive. The better games led to a massive increase in popularity for computer gaming."),
("Q: Did Alice's Adventures in Wonderland inspire Macbeth?",
"No. Alice's Adventures in Wonderland was published in 1865. Macbeth was first performed in 1606.")
]


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

def strategyqa():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Q: Has Johns Hopkins University always treated subjects ethically?
A:
No. Henrietta Lacks' DNA was used and replicated by Johns Hopkins University without her family's knowledge or approval. Henrietta Lacks' family medical history was released by Johns Hopkins University without their knowledge.
----
Q: Could Carl Friedrich Gauss speak to someone 100 miles away?
A:
No. Carl Friedrich Gauss was born in 1777. Speaking to someone 100 miles away requires a telephone. The telephone was invented in 1876.
----
Q: Is calling ABBA the Swedish Beatles a preposterous claim?
A:
Yes. ABBA was a Swedish band that had 1 Billboard number 1 hit and 4 top 10 hits. The Beatles had 20 Billboard number 1 hits and 34 top 10 hits.
----
Q: Did compact discs make computer gaming more popular?
A:
Yes. Compact discs contained significantly more storage space than the previously popular floppy disc format. Gaming studios were therefore able to significantly improve the graphics, sounds, and features of their games to make them more immersive. The better games led to a massive increase in popularity for computer gaming.
----
Q: Did Alice's Adventures in Wonderland inspire Macbeth?
A:
No. Alice's Adventures in Wonderland was published in 1865. Macbeth was first performed in 1606.
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
        processed_labels = [l.split()[0] for l in labels]
        processed_preds = [p.split()[0] for p in preds]
        perf_array.append(exact_match(processed_labels, processed_preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



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


def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    description = """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible."""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nThe final answer should be "Yes" or "No" only.\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
    print(auto_cot_prompt)
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nThe final answer should be "Yes" or "No" only.\nA: Let's think step-by-step.\n"""%x for x in chunk]
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
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

def few_shot_cot(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict("""Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.""", x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))




# strategyqa()
auto_cot(temperature=0.3)