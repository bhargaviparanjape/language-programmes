import argparse
from collections import Counter

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

d = datasets.load_dataset("bigbench", "strategyqa", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]


train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = """(Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible."""

io_pairs = [
    (
        "Q: Has Johns Hopkins University always treated subjects ethically?",
        "No. Henrietta Lacks' DNA was used and replicated by Johns Hopkins University without her family's knowledge or approval. Henrietta Lacks' family medical history was released by Johns Hopkins University without their knowledge.",
    ),
    (
        "Q: Could Carl Friedrich Gauss speak to someone 100 miles away?",
        "No. Carl Friedrich Gauss was born in 1777. Speaking to someone 100 miles away requires a telephone. The telephone was invented in 1876.",
    ),
    (
        "Q: Is calling ABBA the Swedish Beatles a preposterous claim?",
        "Yes. ABBA was a Swedish band that had 1 Billboard number 1 hit and 4 top 10 hits. The Beatles had 20 Billboard number 1 hits and 34 top 10 hits.",
    ),
    (
        "Q: Did compact discs make computer gaming more popular?",
        "Yes. Compact discs contained significantly more storage space than the previously popular floppy disc format. Gaming studios were therefore able to significantly improve the graphics, sounds, and features of their games to make them more immersive. The better games led to a massive increase in popularity for computer gaming.",
    ),
    (
        "Q: Did Alice's Adventures in Wonderland inspire Macbeth?",
        "No. Alice's Adventures in Wonderland was published in 1865. Macbeth was first performed in 1606.",
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
            """Q: Has Johns Hopkins University always treated subjects ethically?
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
%s"""
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
        processed_labels = [l.split()[0] for l in labels]
        processed_preds = [p.split()[0] for p in preds]
        perf_array.append(exact_match(processed_labels, processed_preds))
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
        processed_labels = [l.split()[0] for l in labels]
        processed_preds = [p.split()[0] for p in preds]
        perf_array.append(exact_match(processed_labels, processed_preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def strategyqa(N=10, temperature=0.3, model_name="text-davinci-002"):
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


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1:
# Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
# Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: What is the final answer?
# Unknown
# Q4: [EOC]
# Unknown
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
# Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
# Q1: [tag] What are the entities in this sentence?
# #1:
# President George H. W. Bush
# Gulf War
# Q2: [search] When was President George H. W. Bush president?
# #2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
# Q3: [search] When was the Gulf War fought?
# #3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
# #4: Could these entities have co-existed based on thier time periods alone?
# Yes. Their time periods intersect.
# Q5: Is this an anachronism?
# #5: No
# Q6: [EOC]
# No
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
# Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
# Q1: [tag] What are the entities in this sentence?
# #1:
# Kurt Cobain
# "Twin Peaks"
# Q2: [search] When did television show "Twin Peaks" air?
# #2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
# Q3: [search] When did Kurt Cobain live?
# #3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
# Q4: Could these entities have co-existed based on this information?
# No. Musician  Kurt Cobain could not have starred in Twin Peaks.
# Q5: Is this an anachronism?
# #5: Yes
# Q6: [EOC]
# Yes
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#   choice: Anjalikastra
#   choice: Narayanastra
#   choice: Agneyastra
#   choice: Brahmastra
# Q1: [search] In the Hindu epic Ramayana, who is the main villain?
# #1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q2: [compare] Which option is the answer in #3 most similar to?
# #2: Brahmastra
# Q3: [EOC]
# Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1:
# Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: What is the final answer?
# Florida, Missouri
# Q4: [EOC]
# Florida, Missouri
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Hindu epic Ramayana, the main villain was a devotee of which deity?
#   choice: Indra
#   choice: Vishnu
#   choice: Brahma
#   choice: Shiva
# Q1: [subquestion] Can this question be answered step-by-step?
# #1: Yes.
# Q2: [search] In the Hindu epic Ramayana, who is the main villain?
# #2: Ravana is the main antagonist of the Hindu Epic, the Ramayana.
# Q3: [search] Ravana was a devotee of which deity?
# #3: Ravana, was an ardent devotee of Lord Shiva, is depicted and described as a great scholar,a brahman,a capable ruler and a maestro of the Veena.
# Q4: [compare] Which option is the answer in #3 most similar to?
# #4: Shiva
# Q5: [EOC]
# Shiva
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_retrieval_prompt

auto_cot_corrected_prompt = """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Has Johns Hopkins University always treated subjects ethically?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

On Feb. 6, 1951, a poor African American tobacco farmer named Henrietta Lacks entered the ward for colored women at Johns Hopkins Hospital to begin treatment for invasive cervical cancer. While she was unconscious on the operating table, the surgeon on duty, Dr. Lawrence Wharton Jr., shaved a dime-sized piece of tissue from her tumor and another from the healthy part of her cervix.
Those tissues went on to generate the first “immortal” cell line, endlessly reproduced, studied, experimented with and passed around, still today, by scientists worldwide.
But Lacks — who died later that year — was never told why her cells were taken and never gave permission for them to be used as they have been. In addition, more than two decades passed before her family learned, by accident, about HeLa’s significance.
In this instance, Johns Hopkins University may not have treated thier subject ethically.

The final answer is No.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Could Carl Friedrich Gauss speak to someone 100 miles away?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

Johann Carl Friedrich Gauss (/ɡaʊs/; German: Gauß [kaʁl ˈfʁiːdʁɪç ˈɡaʊs] (listen);[2][3] Latin: Carolus Fridericus Gauss; 30 April 1777 – 23 February 1855) was a German mathematician and physicist who made significant contributions to many fields in mathematics and science.
Speaking to someone 100 miles away requires a telephone. On 7 March 1876, Bell was granted US patent 174465A, for a method of transmitting speech by telegraphy—the telephone.
Gauss died before the telephone was invented and popularly used. Thus, its unlikely for him to have used a telephone to speak to someone 100 miles away.

The final answer is No.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Is calling ABBA the Swedish Beatles a preposterous claim?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

No, ABBA doesn’t deserves that sort of adoration. Abba produced well crafted Euro-pop, but that was it.
ABBA was a Swedish band that had 1 Billboard number 1 hit and 4 top 10 hits. The Beatles had 20 Billboard number 1 hits and 34 top 10 hits.
The Beatles took the world by storm and reached all capillaries of society. They affected the world philosophically, politically, in terms of fashion and musically.
Calling ABBA the Swedish Beatles maybe a preposterous claim.

The final answer is Yes.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Did compact discs make computer gaming more popular?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

First, we need to understand what is meant by "computer gaming." In this context, computer gaming refers to video games that are played on a computer.
The largest storage available on commercial floppy disks was about 2.8MB. Whereas even relatively early Optical Discs had capacities in the hundreds of MB. Without significantly increased cost.
Gaming studios were therefore able to significantly improve the graphics, sounds, and features of their games to make them more immersive. The better games led to a massive increase in popularity for computer gaming.

The final answer is Yes.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Did Alice's Adventures in Wonderland inspire Macbeth?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

First, we need to establish whether Alice's Adventures in Wonderland was published before or after Macbeth. If it was published after, then it's implausible that it could have inspired Macbeth. However, if it was published before, it's possible that it could have inspired Macbeth.
Alice's Adventures in Wonderland was published in 1865.
Macbeth was written in 1606.
Since Alice's Adventures in Wonderland was published befafterore Macbeth, it's impossible that it inspired Macbeth.

The final answer is No.
----
"""

auto_cot_cleaned_prompt = """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Has Johns Hopkins University always treated subjects ethically?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

1. Has Johns Hopkins University always conducted clinical research?
Yes. Johns Hopkins University has been conducting clinical research since its founding in 1876.
2. Has Johns Hopkins University always followed federal regulations for clinical research?
Yes. Johns Hopkins University has been following federal regulations for clinical research since the regulations were established in the 1970s.
3. Has Johns Hopkins University always treated subjects ethically?
Yes. Johns Hopkins University has been treating subjects ethically since its founding in 1876.

The final answer is Yes.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Could Carl Friedrich Gauss speak to someone 100 miles away?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

Can Carl Friedrich Gauss speak? Yes.
Can he speak 100 miles away? It's possible, but there's no evidence that he could, so we'll say no.

The final answer is No.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Is calling ABBA the Swedish Beatles a preposterous claim?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

Is ABBA a popular band?
Yes
Did the Beatles originate from Sweden?
No
Is it preposterous to compare ABBA to the Beatles?
Yes

The final answer is Yes
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Did compact discs make computer gaming more popular?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

1. Did compact discs make computer gaming more popular?
Yes, there is evidence that compact discs made computer gaming more popular. For example, when the Sony PlayStation released in 1994, it used compact discs instead of cartridges. This made it much easier to produce and distribute games, which led to a boom in the video game industry.
2. Did computer gaming make compact discs more popular?
No, there is no evidence that computer gaming made compact discs more popular. In fact, it's the other way around: compact discs made computer gaming more popular.

The final answer is Yes.
----
Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible.
Q: Did Alice's Adventures in Wonderland inspire Macbeth?
The final answer should be "Yes" or "No" only.
A: Let's think step-by-step.

1. Did Alice's Adventures in Wonderland come before Macbeth?
Yes.
2. Is there evidence that Macbeth was influenced by Alice's Adventures in Wonderland?
No.

The final answer is No.
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
    description = """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or the question is implausible."""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % description
            + io_pair[0]
            + """\nThe final answer should be "Yes" or "No" only.\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Anachronisms\n\n")
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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        processed_labels = [l.split()[0] for l in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(processed_labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Strategy QA"
    task_description = """(Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible."""

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
    runs = 3
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        processed_labels = [l.split()[0].replace(".", "") for l in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(
                predict_complete(
                    """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.""",
                    x,
                )
            )
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(processed_labels, preds))
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
        gpt3 = OpenAIModel(
            model=model_name, max_length=1024, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Strategy QA"
    task_description = """(Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible."""

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
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=6)

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=15):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 15)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in processed_answers:
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers[: len(inputs)]]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            processed_labels = [label.split(".")[0] for label in labels]
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(processed_labels, preds))
            print(perf_array)
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / len(interpreter.execution_details)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Is a person with St. Vitus's Dance likely to win a ballet competition?
Q1: [search] What is St. Vitus's Dance?
#1: St. Vitus's Dance is a disorder characterized by jerking or twitching movements.
Q2: [search] Can someone with jerky or twitching movements win a ballet competition?
#2: No, someone with jerky or twitching movements would not be able to win a ballet competition. Ballet requires precise, graceful movements that are difficult to achieve with jerky or twitching movements.
Smiling, shaking your limbs dynamically and looking upwards will help you do that. 2. Practice positive thinking. The quality of your self-talk ...
Q3: [generate output] Answer the original question given the information about St. Vitus's Dance and ballet competitions.
No. A person with St. Vitus's Dance is not likely to win a ballet competition.
Q4: [EOQ]
Ans: No
----
Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Was Amy Winehouse a fan of Star Wars: Rogue One?
Q1: [search] When did Amy Winehouse live?
#1: Amy Winehouse lived from September 14, 1983 to July 23, 2011. Amy Jade Winehouse (14 September 1983 – 23 July 2011) was an English singer and songwriter. She was known for her deep, expressive contralto vocals and her ...
Q2: [search] When did the movie Star Wars: Rogue One release?
#2: Star Wars: Rogue One was released on December 16, 2016. December 16, 2016
Q2: [generate output] Answer the original question given the information about Amy Winehouse and Star Wars: Rogue One
#2: No. Amy Winehouse was not a fan of Star Wars: Rogue One since the movie was released after her death.
Q3: [EOQ]
Ans: No
----
Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Would 1943-S penny be good for making silverware?
Q1: [search] What is the composition of the 1943-S penny?
#1: The 1943-S penny is special. The unique composition of the coin (low-grade steel coated with zinc, instead of the previously 95%%-copper-based bronze composition) has led to various nicknames, such as wartime cent, steel war penny, zinc cent and steelie.
Q2: [search] Is steel a suitable material for making silverware?
#2: No, steel is not a suitable material for making silverware as it is not malleable enough for the intricate details needed for silverware.
Q3: [generate output] Answer the original question given the information about the 1943-S penny and silverware.
No. The 1943-S penny is not suitable for making silverware due to its composition of low-grade steel coated with zinc.
Q4: [EOQ]
Ans: No
----
Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Do workers at Nissan's headquarters eat with chopsticks?
Q1: [search] Where is Nissan's headquarters located?
#1: Nissan's headquarters is located in Yokohama, Japan. Yokohama, Kanagawa, Japan
Q2: [search] Is chopsticks a common eating utensil in Japan?
#2: Yes, chopsticks are a common eating utensil in Japan. If you're planning on visiting or working in Japan, you'll probably need to use chopsticks at some point.
Q3: [generate output] Answer the original question given the information about Nissan's headquarters and chopsticks.
Yes. Workers at Nissan's headquarters in Yokohama, Japan likely eat with chopsticks.
Q4: [EOQ]
Ans: Yes
----
Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Can French Defence initial move defend against four move checkmate?
Q1: [search] What is French Defence?
#1: The French Defence is a chess opening where White moves their pawn to e4 and black moves thiers to e6.  The French Defense chess opening is a semi-closed chess opening for Black which occurs after the moves 1.e4 e6 2.d4 d5.
Q2: [search] Can the French Defence initial move defend against four move checkmate?
#2: No, the French Defence initial move cannot defend against four move checkmate. Four move checkmates can only be prevented by a move other than the French Defence initial move. Ever wondered how to defend against the four move checkmate?
Q3: [generate output] Answer the original question given the information about the French Defence and four move checkmate.
No. The French Defence initial move cannot defend against four move checkmate.
Q4: [EOQ]
Ans: No
----
Description: (Strategy QA) Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.
Input: Could Ryan Crouser throw a bengal fox with ease?
Q1: [search] Who is Ryan Crouser?
#1: Ryan Crouser is an American Olympic shot put and discus thrower. He is the current world record holder in the shot put and the Olympic champion in the event.
Q2: [search] What is the largest weight lifted by Ryan Crouser?
#2: Ryan Crouser holds the world record in the shot put with a throw of 22.72 m, which is equivalent to approximately 74.8 kg. In 2019, Wired broke down Crouser's skills, noting that at 6-foot-7, 315 lb, Crouser can bench press 225 lb 50 times, squat 700 lb, all while ...
Q3: [search] How much does a Bengal fox weigh?
#3: The average weight of a Bengal fox is between 3.5 and 6.8 kg. 5.2 lbs
Q4: [generate output] Answer the original question given the information about Ryan Crouser and a Bengal fox.
Yes. Ryan Crouser could easily throw a Bengal fox with ease since the fox weighs less than the weight of the shot put (approximately 74.8 kg) that Ryan Crouser is able to lift.
Q5: [EOQ]
Ans: Yes
----
Description: %s
Input: %s
Q1:"""


def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_human_prompt
    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[code generate]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1500, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=15):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\nEdited:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 15)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in processed_answers:
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers[: len(inputs)]]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            processed_labels = [label.split(".")[0] for label in labels]
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("Q: ", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(
                    """Strategy QA: Answer these questions with a "Yes" or "No", depending on whether there is sufficient evidence to answer the question or if the question is implausible.""",
                    x,
                )
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(processed_labels, preds))
            # Report on interpreter performance
            print(perf_array)
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / (len(interpreter.execution_details) + 1e-6)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


# human_intervention(0.3, "text-davinci-002")

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
            "auto_cot_corrected",
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
    elif args.inference_strategy == "auto_cot_corrected":
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
