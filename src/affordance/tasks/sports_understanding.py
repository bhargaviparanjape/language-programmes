import argparse
from collections import Counter
import json
import pdb
import re
import time

import datasets
import numpy as np
from prompt_library import (
    few_shot_free_prompt,
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

# international_phonetic_alphabet_transliterate too
d = datasets.load_dataset("bigbench", "sports_understanding", cache_dir=cache_dir)
inputs = d["validation"]["inputs"][:500]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"][:500]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = """(Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible"."""

io_pairs = [
    (
        """Determine whether the following statement or statements are plausible or implausible:
Statement: James Harden shot a free throw
Plausible/implausible?""",
        "plausible",
    ),
    (
        """Determine whether the following statement or statements are plausible or implausible:
Statement: Ben Simmons called for the screen
Plausible/implausible?""",
        "plausible",
    ),
    (
        """Determine whether the following statement or statements are plausible or implausible:
Statement: Philipp Lahm threw a touchdown
Plausible/implausible?""",
        "implausible",
    ),
    (
        """Determine whether the following statement or statements are plausible or implausible:
Statement: William Nylander scored in the shootout in the Stanley Cup
Plausible/implausible?""",
        "plausible",
    ),
]


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
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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
        if label.lower() in predict.lower().split():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def sports_understanding():
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=200, quote="---", n=1)
        prompts = [
            """Determine whether the following statement or statements are plausible or implausible:
Statement: James Harden shot a free throw
Plausible/implausible?
plausible
----
Determine whether the following statement or statements are plausible or implausible:
Statement: Ben Simmons called for the screen
Plausible/implausible?
plausible
----
Determine whether the following statement or statements are plausible or implausible:
Statement: Philipp Lahm threw a touchdown
Plausible/implausible?
implausible
----
Determine whether the following statement or statements are plausible or implausible:
Statement: William Nylander scored in the shootout in the Stanley Cup
Plausible/implausible?
plausible
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
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def human_decomp(model_name="text-davinci-002", temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Q: Is the following sentence plausible? "Bam Adebayo scored a reverse layup in the Western Conference Finals."
A: Let's think step by step. Bam Adebayo is an American basketball player. Scoring a reverse layup in the Western Conference Finals is part of the NBA Finals. So the answer is plausible.
----
Q: Is the following sentence plausible? "Santi Cazorla scored a touchdown."
A: Let's think step by step. Santi Cazorla is a soccer player. Touchdown is part of American football and rugby. So the answer is implausible.
----
Q: Is the following sentence plausible? "DeMar DeRozan was called for the goal tend."
A: Let's think step by step. DeMar DeRozan is an American basketball player. Goal tending is part of basketball. So the answer is plausible.
----
%s
A: Let's think step by step."""
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
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1:
# Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin.
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon.
# Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: [compare] What is the final answer?
# #3: Unknown
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
# #4: Could these entities have co-existed based on thier time periods alone?
# Yes. Their time periods intersect.
# Q5: [generate output] Is this an anachronism?
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
# Q4: [subquestion] Could these entities have co-existed based on this information?
# No. Musician  Kurt Cobain could not have starred in Twin Peaks.
# Q5: [generate output] Is this an anachronism?
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
# Q1: [search] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
# #1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q2: [compare] Which option is the answer in #3 most similar to?
# #2: Brahmastra
# Q3: [EOQ]
# Ans: Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1:
# Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: [compare] What is the final answer?
# #3: Florida, Missouri
# Q4: [EOQ]
# Ans: Florida, Missouri
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_free_prompt

auto_cot_corrected_prompt = """Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: James Harden shot a free throw
Plausible/implausible?
A: Let's think step-by-step.

First, we need to know who James Harden is. He is an American professional basketball player for the Houston Rockets of the National Basketball Association (NBA). So far, so good.
Next, we need to know what a free throw is. In basketball, a free throw is a shot that a player takes from the free throw line, which is located at the end of the court. The player is given this opportunity because the other team committed a foul against them.
So, given that we know who James Harden is and what a free throw is, is it plausible or implausible that he shot a free throw? It's plausible.

The final answer is Plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Ben Simmons called for the screen
Plausible/implausible?
A: Let's think step-by-step.

First we need to know what a screen is. A screen is a blocking move by an offensive player in which they stand beside or behind a defender in order to free a teammate to either shoot a pass or drive in to score. In basketball and field lacrosse, it is also known as a pick.
Benjamin David Simmons is an Australian professional basketball player for the Brooklyn Nets of the National Basketball Association.
Its possible for Ben Simmons to have called for a screen at some point while playing basketball. Therefore, the statement is plausible.

The final answer is Plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Philipp Lahm threw a touchdown
Plausible/implausible?
A: Let's think step-by-step.

In order to throw a touchdown, Philipp Lahm would first need to be playing American football. This is implausible, as Lahm is a German soccer player. Therefore, the statement that he threw a touchdown is also implausible.

The final answer is Implausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: William Nylander scored in the shootout in the Stanley Cup
Plausible/implausible?
A: Let's think step-by-step.

Firstly, William Nylander plays for the Toronto Maple Leafs, who made it to the Stanley Cup Playoffs this year. So it's also plausible that he would be playing in the Stanley Cup.
Secondly, shootouts are a common way to decide games in the Stanley Cup Playoffs. So it's plausible that a shootout would happen in this instance.
Finally, it's plausible that Nylander would score in the shootout, as he is known for being a skilled player. Therefore, the entire sentence is plausible.

The final answer is Plausible.
----
"""
auto_cot_cleaned_prompt = """Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: James Harden shot a free throw
Plausible/implausible?
A: Let's think step-by-step.

In order for the statement to be plausible, we need to know a few things. First, we need to know who James Harden is. James Harden is an American professional basketball player for the Houston Rockets of the National Basketball Association (NBA). He is one of the best players in the NBA, so it is not implausible that he would be shooting a free throw.
Next, we need to know what a free throw is. A free throw is a shot that a player takes from the free throw line, which is a line 15 feet from the basket. The player is given one shot and the shot is worth one point.
Based on the information given, the statement is plausible.

The final answer is plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Ben Simmons called for the screen
Plausible/implausible?
A: Let's think step-by-step.

In order for the sentence to be plausible, we need to know a few things. First, we need to know who Ben Simmons is. We'll assume that he is a professional basketball player. Second, we need to know what a screen is in the context of basketball. A screen is when one player stands still and another player uses their body to shield the first player from the defender.
Given this information, the sentence is plausible.

The final answer is plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Philipp Lahm threw a touchdown
Plausible/implausible?
A: Let's think step-by-step.

Can a soccer player throw a touchdown? No.
Can a football player throw a touchdown? Yes.
Therefore, the statement is implausible.

The final answer is implausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: William Nylander scored in the shootout in the Stanley Cup
Plausible/implausible?
A: Let's think step-by-step.

First, William Nylander is a professional hockey player. So far, so good.
Next, we need to see if he could score in a shootout in the Stanley Cup. This is a little more difficult to determine, but we know that shootouts are a part of hockey, and that the Stanley Cup is the biggest prize in hockey. So it's not implausible that William Nylander could score in a shootout in the Stanley Cup.

The final answer is plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: James Harden shot a free throw
Plausible/implausible?
A: Let's think step-by-step.

In order for the statement to be plausible, we need to know a few things. First, we need to know who James Harden is. James Harden is an American professional basketball player for the Houston Rockets of the National Basketball Association (NBA). He is one of the best players in the NBA, so it is not implausible that he would be shooting a free throw.
Next, we need to know what a free throw is. A free throw is a shot that a player takes from the free throw line, which is a line 15 feet from the basket. The player is given one shot and the shot is worth one point.
Based on the information given, the statement is plausible.

The final answer is plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Ben Simmons called for the screen
Plausible/implausible?
A: Let's think step-by-step.

In order for the sentence to be plausible, we need to know a few things. First, we need to know who Ben Simmons is. We'll assume that he is a professional basketball player. Second, we need to know what a screen is in the context of basketball. A screen is when one player stands still and another player uses their body to shield the first player from the defender.
Given this information, the sentence is plausible.

The final answer is plausible.
----
Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'
Determine whether the following statement or statements are plausible or implausible:
Statement: Philipp Lahm threw a touchdown
Plausible/implausible?
A: Let's think step-by-step.

Can a soccer player throw a touchdown? No.
Can a football player throw a touchdown? Yes.
Therefore, the statement is implausible.

The final answer is implausible.
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
    description = """Sports Understanding: Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'"""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.7, quote="---", n=1)
        prompt = """%s\n""" % description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Sports Understanding\n\n")
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
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Sports Understanding"
    task_description = """(Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible"."""

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
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
            pdb.set_trace()
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers = predict(
                "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                x,
            )
            pdb.set_trace()
            affordance_inputs = [
                json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers
            ]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [
                ex + a[: re.search("#2: ", a).span(0)[1]] + json.dumps(o)
                for ex, a, o in zip(x, answers, affordance_outputs)
            ]
            new_answers.extend(
                predict_with_affordance(
                    "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                    x,
                )
            )
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
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
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


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Sports Understandings"
    task_description = "(Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be 'plausible' or 'implausible'"

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

    interpreter = TopDownVisitorBeta(model_name=model_name)  # , rerun=False)

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
        runs = 5
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
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

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        old_answers = []
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            prompts, answer = predict(task_description, x)
            old_answers.extend(answer)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        pdb.set_trace()
        preds = [get_answer(x.strip()) for x in answers]
        [get_answer(x.strip()) for x in old_answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_human_prompt = """Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible".
Input: Determine whether the following statement or statements are plausible or implausible:
Statement: Joao Moutinho caught the screen pass in the NFC championship
Plausible/implausible?
Q1: [search] What is a screen pass?
#1: A screen pass is a play in gridiron football consisting of a short pass to a receiver who is behind the line of scrimmage. A screen pass is a play where the quarterback fakes a handoff or long pass but instead throws a short pass to a receiver who has positioned ...
Q2: [search] What is the NFC championship?
#2: The National Football Conference (NFC) Championship Game is one of the two semi-final playoff games of the National Football League (NFL), the largest professional American football league in the United States. The game is played on the penultimate Sunday in January by the two remaining playoff ... The NFC Championship Game is the annual championship game of the National Football Conference (NFC) and one of the two semifinal playoff games of the ...
Q3: [search] What sport does Joao Moutinho play and in what position?
#3: Joao Moutinho is a professional soccer player who plays as a midfielder for the French Ligue 1 club Olympique de Marseille.
Q3: [generate output] Could Joao Moutinho have caught the screen pass in the NFC championship?
#3: No, Joao Moutinho could not have caught the screen pass in the NFC championship, as he is a professional soccer player and not a gridiron football player.
Q4: [EOQ]
Ans: Implausible
----
Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible".
Input: Determine whether the following statement or statements are plausible or implausible:
Statement: Mathieu Valbuena scored a windmill dunk in the Western Conference Finals
Plausible/implausible?
Q1: [search] What is a windmill dunk?
#1: What Is The Definition Of A Windmill Dunk In Basketball? 1. This is a type of dunk where the player brings the ball up to their waste, then moves their arm straight back and up in a circular motion until they dunk the ball over their head.
Q2: [search] What is the Western Conference Finals?
#2: The Western Conference Finals is the third round of the National Basketball Association (NBA) playoffs. It is a best-of-seven series between the two teams from the Western Conference that win their respective divisions during the regular season.
Q3: [search] What sport does Mathieu Valbuena play and in what position?
#3: Mathieu Valbuena is a professional soccer player who plays as an attacking midfielder for the French Ligue 1 club Olympique de Marseille.
Q3: [generate output] Could Mathieu Valbuena have scored a windmill dunk in the Western Conference Finals?
#3: No, Mathieu Valbuena could not have scored a windmill dunk in the Western Conference Finals, as he is a professional soccer player and not a basketball player.
Q4: [EOQ]
Ans: Implausible
----
Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible".
Input: Determine whether the following statement or statements are plausible or implausible:
Statement: Igor Shesterkin was called for icing
Plausible/implausible?
Q1: [search] What is icing?
#1: Icing is a rule in some forms of ice hockey that prohibits a team from shooting the puck across the centre line and into the offensive zone of the other team. When the team does this, the referee blows the whistle and the opposing team is allowed a faceoff in that zone.
Q2: [search] What sport does Robert Covington play and in what position?
#2: Igor Shesterkin is a professional ice hockey goaltender who plays for the New York Rangers of the National Hockey League (NHL).
Q3: [generate output] Could Igor Shesterkin have been called for icing?
#3: Yes, Igor Shesterkin could have been called for icing, as he is a professional ice hockey goaltender.
Q4: [EOQ]
Ans: Plausible
----
Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible or implausible. The final answer should be "plausible" or "implausible".
Input: Determine whether the following statement or statements are plausible or implausible:
Statement: Matt Ryan scored a penalty kick
Plausible/implausible?
Q1: [search] What is a penalty kick?
#1: A penalty kick is a type of free kick in association football, taken from twelve yards (approximately eleven metres) out from goal and with only the goalkeeper of the defending team between the penalty taker and the goal.
Q2: [search] What sport does Matt Ryan play and in what position?
#2: Matt Ryan is a professional American football quarterback for the Atlanta Falcons of the National Football League (NFL).
Q3: [generate output] Could Matt Ryan have scored a penalty kick?
#3: No, Matt Ryan could not have scored a penalty kick, as he is a professional American football quarterback and not a soccer player.
Q4: [EOQ]
Ans: Implausible
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
    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1500, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=9):
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
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 9)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in enumerate(processed_answers):
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0][1] for x in answers[: len(inputs)]]
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
            for x in tqdm(chunks(inputs, 10)):
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
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


# human_decomp("text-davinci-002", 0.3)
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
            "cot_rollout",
            "few_shot_cot",
            "nl_program",
            "auto_cot_corrected",
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
            use_corrected=True,
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
