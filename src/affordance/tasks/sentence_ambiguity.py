import argparse
from collections import Counter
import json
import pdb
import re

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

d = datasets.load_dataset("bigbench", "sentence_ambiguity", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = """Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False"."""

io_pairs = [
    (
        """Claim: LeBron James is universally considered the best football player of all time.
True or False?""",
        "False",
    ),
    (
        """Claim: The cornea of your eye is likely where the light receptors are.
True or False?""",
        "False",
    ),
    (
        """Claim: Few animals live more than 200 years, but none of these have fins.
True or False?""",
        "False",
    ),
    (
        """Claim: The United States has never had a unanimously elected president.
True or False?""",
        "False",
    ),
    (
        """Claim: The North Pole is likely not located on a continent.
True or False?""",
        "True",
    ),
    (
        """Claim: Earth's oceans may include the Pacific, Atlantic, Indian, and Arctic.
True or False?""",
        "True",
    ),
    (
        """Claim: Australia had more Cricket World Cup wins than the West Indies as of 2021.
True or False?""",
        "True",
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
            """Claim: LeBron James is universally considered the best football player of all time.
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
auto_cot_cleaned_prompt = """(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: LeBron James is universally considered the best football player of all time.
True or False?
A: Let's think step-by-step.

First, we need to establish what "universally considered" means. This is a pretty strong claim, so we're going to need to see evidence that literally everyone believes LeBron James is the best football player of all time.
Next, we need to consider the evidence. Even if everyone believes LeBron James is the best football player of all time, is there any actual evidence to support this claim?
Based on these two considerations, it seems fair to say that the claim is false.

The final answer is false.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: The cornea of your eye is likely where the light receptors are.
True or False?
A: Let's think step-by-step.

The cornea of your eye is the outermost layer of your eye. The light receptors are located in the retina, which is the innermost layer of your eye. So, the claim that the cornea of your eye is likely where the light receptors are is false.

The final answer is false.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Few animals live more than 200 years, but none of these have fins.
True or False?
A: Let's think step-by-step.
First, we'll look at the claim that "few animals live more than 200 years." To determine if this is true, we'll need to find evidence of how long different animals live.
According to this website, the longest-living land animal is the bowhead whale, which can live up to 211 years. The longest-living vertebrate in the sea is the greenland shark, which can live up to 400 years.
Based on this information, we can say that the claim that "few animals live more than 200 years" is false - there are at least two animals that have been documented to live over 200 years.

Now, we'll look at the claim that "none of these have fins." We'll need to find evidence of whether or not the bowhead whale and the greenland shark have fins.
According to this website, the bowhead whale does have fins. This website says that the greenland shark does not have fins.
Based on this information, we can say that the claim that "none of these have fins" is also false - at least one of the animals that can live over 200 years (the bowhead whale) does have fins.
In conclusion, both claims in the sentence are false.

The final answer is false.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: The United States has never had a unanimously elected president.
True or False?
A: Let's think step-by-step.

First, what does it mean for a president to be "unanimously elected"? This means that every single person who voted, voted for that one president.
Now, has the United States ever had a president who was elected with every single vote?
The answer is no. The United States has never had a unanimously elected president.

The final answer is true.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: The North Pole is likely not located on a continent.
True or False?
A: Let's think step-by-step.

There are seven continents: Europe, Asia, Africa, North and South America, Australia, and Antarctica. The North Pole is located in the Arctic Ocean, which is adjacent to Europe, Asia, and North America. So, the North Pole is not on a continent.

The final answer is true.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Earth's oceans may include the Pacific, Atlantic, Indian, and Arctic.
True or False?
A: Let's think step-by-step.

The first part of the claim is that Earth has oceans. This is a fact that is widely accepted and supported by evidence.
The second part of the claim is that the oceans may include the Pacific, Atlantic, Indian, and Arctic. This is also a fact that is widely accepted and supported by evidence.
Therefore, the entire claim is true.

The final answer is true.
----
(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence.
Claim: Australia had more Cricket World Cup wins than the West Indies as of 2021.
True or False?
A: Let's think step-by-step.

Australia has won the Cricket World Cup 5 times as of 2021. The West Indies have won the Cricket World Cup 2 times as of 2021. 5 is greater than 2, so the claim is true.

The final answer is true.
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
    description = "(Sentence ambiguity) Determine if the claim is true or false based on whether facts stated in the sentence have evidence."
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
    f.write("Sentence ambiguity\n\n")
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
            + """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""
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
            + """%s\n""" % description
            + """%s\nA: Let's think step-by-step.\n""" % x
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
                # x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x) for x in answers]
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
            answers.extend(
                predict_complete(
                    """Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".""",
                    x,
                )
            )
            # time.sleep(10)
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

    def search_query(answer):
        lines = answer.strip().split("\n")
        print(lines)
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
        pdb.set_trace()
        return "\n".join(new_lines)

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
            answers = predict("Determine if the claim is true or false.", x)
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [search_query("Q1:" + a) for a in answers]
            pdb.set_trace()
            x = [
                ex + a[: re.search("#2: ", a).span(0)[1]] + json.dumps(o)
                for ex, a, o in zip(x, answers, affordance_outputs)
            ]
            new_answers.extend(
                predict_with_affordance("Determine if the claim is true or false.", x)
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

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Description: Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".
Input: Abraham Lincoln was the tallest president to serve the United States, followed by James Madison.
Q1: [search] How tall was Abraham Lincoln?
#1: Abraham Lincoln was 6 feet 4 inches tall.
Q2: [search] How tall was James Madison?
#2: James Madison was 5 feet 4 inches tall. 5′ 4″ 5′ 4″
Q3: [search] Who is the tallest president of US
#3: The tallest U.S. president was Abraham Lincoln at 6 feet 4 inches (193 centimeters), while the shortest was James Madison at 5 feet 4 inches (163 centimeters).
Q4: [subquestion] Is the claim true, given this information?
#4: No, James Madison was actually the shortest president to serve the United States, not the second tallest. The claim is false.
Q5: [EOQ]
Ans: False
----
Description: Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".
Input: The North Pole is likely not located on a continent.
Q1: [search] Where is the North Pole located?
#1:
The North Pole, also known as the Geographic North Pole or Terrestrial North Pole, is (subject to the caveats mentioned below) defined as the point in the
northern hemisphere where the Earth's axis of rotation meets its surface. The nearest land is usually said to be Kaffeklubben Island, off the northern coast of Greenland about 700 km (430 mi) away.
Q2: [search] Is the north pole located on a land mass?
#2: Unlike Antarctica, there's no land at the North Pole. Instead it's all ice that's floating on top of the Arctic Ocean.
Q3: [subquestion] Is the claim true, given this information?
#3: Yes. The North Pole is not one of Earth's seven main divisions of land. The claim is true.
Q4: [EOQ]
Ans: True
----
Description: Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".
Input: There is only one even prime number, and it occurs between 3 and 17.
Q1: [search] Is there an even prime number?
#1: The unique even prime number 2. All other primes are odd primes.
Q2: [subquestion] Is 2 between 3 and 17.
#2: No.
Q3: [subquestion] Is the claim true, given this information?
#3: No. There is no even prime between 3 and 17. The claim is false.
Q4: [EOQ]
Ans: False
----
Description: Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".
Input: The capital of India is likely not New Delhi.
Q1: [search] What is the capital of India?
#1: New Delhi is the capital of India. New Delhi, national capital of India. It is situated in the north-central part of the country on the west bank of the Yamuna River, ...
Q2: [subquestion] Is the claim true, given this information?
#2: No. The capital of India is infact New Delhi. The claim is false.
Q3: [EOQ]
Ans: False
----
Description: Determine if the claim is true or false based on whether facts stated in the sentence have evidence. The final answer should be one of "True" or "False".
Input: The population of Guangdong, a province in China, was more than twice that of South Korea in 2019.
Q1: [search] What was the population of Guangdong province of Chinain in 2019?
#1: With a population of 126.01 million (as of 2020) across a total area of about 179,800 km2 (69,400 sq mi), Guangdong is the most populous province of China and the 15th-largest by area as well as the second-most populous country subdivision in the world (after Uttar Pradesh in India).
Q2: [search] What was the population of South Korea in 2019?
#2: The population of South Korea was 51,225,308 in 2019. Population of South Korea (2020 and historical) ; 2019, 51,225,308, 0.10 %% ; 2018, 51,171,706, 0.15 %% ; 2017, 51,096,415, 0.22 %% ; 2016, 50,983,457, 0.32 %% ...
Q3: [subquestion]  Is the claim true, given this information?
#3: Yes, the population of Guangdong is more than twice that of South Korea since 126.01 million is more than 51,225,308 times 2. The claim is true.
Q4: [EOQ]
Ans: True
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
                x = [ex.replace("Claim: ", "") for ex in x]
                x = [ex.replace("\nTrue or False?", "") for ex in x]
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
    parser.add_argument(
        "--selection_strategy",
        type=str,
        choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"],
        default="fixed",
    )
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action="store_true")

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
