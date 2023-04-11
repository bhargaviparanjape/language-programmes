import argparse
from collections import Counter
import re
import time

import datasets
import numpy as np
from prompt_library import (
    few_shot_string_prompt,
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
    get_subset,
    propose_decomposition,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# international_phonetic_alphabet_transliterate too
d = datasets.load_dataset("bigbench", "date_understanding", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input."

io_pairs = [
    (
        """Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?""",
        """A:
05/01/2021""",
    ),
    (
        """Q: Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date today in MM/DD/YYYY?""",
        """A:
10/15/1924""",
    ),
    (
        """Q: Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?""",
        """A:
02/16/2010""",
    ),
    (
        """Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date a month ago in MM/DD/YYYY?""",
        """A:
04/06/2002""",
    ),
    (
        """Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date a month ago in MM/DD/YYYY?""",
        """A:
12/07/2018""",
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
    # Not all tokens are equal, it makes sense to only compare tokens that are generated at the end of the sequence.
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower().split():
            correct += 1
        count += 1
    return (1.0 * correct) / count


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


def human_decomp(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=1000,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [
            """Q: Today is Christmas Eve of 1937. What is the date 10 days ago in MM/DD/YYYY?
A: Let's think step by step.
If today is Christmas Eve of 1937, then today's date is December 24, 1937. 10 days before today is December 14, 1937, that is 12/14/1937. So the answer is 12/14/1937.
----
Q: Tomorrow is 11/12/2019. What is the date one year ago from today in MM/DD/YYYY?
A: Let's think step by step.
If tomorrow is 11/12/2019, then today is 11/11/2019. The date one year ago from today is 11/11/2018. So the answer is 11/11/2018.
----
Q: Jane and John married on Jan 2, 1958. It is their 5-year anniversary today. What is the date tomorrow in MM/DD/YYYY?
A: Let's think step by step.
If Jane and John married on Jan 2, 1958, then and if it is their 5-year anniversary today, then today's date is Jan 2, 1963. The date tomorrow is Jan 3, 1963, that is 01/03/1963. So the answer is 01/03/1963.
----
Q: %s
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
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def auto_decomp(N=10, temperature=0.3):
    # The other ways to do this is to ask what functions to use and given these functions, come up with required steps.
    decomp_prompt = (
        "%s I want to break this task into individual steps to make it easier to solve."
        % task_description
    )
    decompositions = propose_decomposition(decomp_prompt, io_pairs, N)

    def get_decomp_func(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])
        decomposition += "\n%s. The final answer should be in the MM/DD/YYYY format." % (last_n + 1)

        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(
                model="text-davinci-002",
                max_length=1000,
                temperature=temperature,
                quote="---",
                n=1,
            )
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """%s Using the following steps will help.
Steps:
%s
----
%s
Also show how you used the steps provided to arrive at the answer."""
                    % (task_description, decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(inputs, labels=labels, n=len(inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition:", z)
        print(decompositions[z])
        fn = get_decomp_func(decomposition, batch_size=20)
        subset = [ex.replace("\nA:", "") for ex in subset]
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


auto_cot_corrected_prompt = """(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

Yesterday was April 30, 2021.

Today is May 1, 2021. We now need to convert this date into the MM/DD/YYYY format.

Therefore, the required date is 05/01/2021.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

Tomorrow is 10/16/1924.

So the date today is 10/15/1924.

Therefore, the required date is 10/15/1924.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

Jane's first visit to the bookstore was on October 16th, 2009.
Her second visit will be on November 16th, 2009.
Her third visit will be on December 16th, 2009.
Her fourth visit will be on January 16th, 2010.
Her fifth visit, which is today, will be on February 16th, 2010.
We now need to convert this date into the MM/DD/YYYY format.

Therefore, the required date is 02/16/2010.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date a month ago in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

May 6, 1992 is ten years ago from today. So today is  May 6, 2002. A month ago from today will be April 6, 2002. We now need to convert this date into the MM/DD/YYYY format.

Therefore, the required date is 04/06/2002.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date a month ago in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

The first day of 2019 is a Tuesday. Thus January 1, 2019 is Tuesday.
Today is the first monday of 2019. The first Monday of 2019 will be 6 days away from that day. So today is January 7, 2019.
Therefore, 1 month ago, it was December 7, 2018. We now need to convert this date into the MM/DD/YYYY format.

Therefore, the required date is 12/07/2018.
----
"""

auto_cot_cleaned_prompt = """(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Yesterday was April 30, 2021. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

Yesterday was April 30, 2021.
The day before yesterday was April 29, 2021.
Today is April 30, 2021.

The final answer is April 30, 2021.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Today's meeting is rescheduled to 11 am tomorrow, 10/16/1924. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

First, we need to find out how many days have passed since the original meeting. We know the meeting was rescheduled to tomorrow, which means that one day has passed.
Therefore, the current date is 10/17/1924.

The final answer is 10/17/1924.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: Jane visits the bookstore on the 16th of each month starting from the October of 2009. It is her 5th visit to the bookstore today. What is the date today in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

Jane's first visit to the bookstore was on October 16th, 2009.
5 visits later would be January 16th, 2010.
Therefore, the answer is 01/16/2010.

The final answer is 01/16/2010.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: May 6, 1992 is like yesterday to Jane, but that is actually ten years ago. What is the date a month ago in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

A month ago from May 6, 1992 would be April 6, 1992.

The final answer is 04/06/1992.
----
(Date understanding task) Find the required date in MM/DD/YYYY using information about related events and dates in the input.
Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date a month ago in MM/DD/YYYY?
The final answer should be in the MM/DD/YYYY format.
A: Let's think step-by-step.

A month ago from today would be December 3, 2018.
The first day of 2019 is January 1, 2019.
Therefore, a month ago from the first day of 2019 would be December 1, 2018.

The final answer is 12/01/2018.
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
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer should be in the MM/DD/YYYY format.\nA: Let's think step-by-step.\n"""
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
    f.write("Date understanding\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    print(auto_cot_prompt)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer should be in the MM/DD/YYYY format.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.3, quote="---", n=1)
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer should be in the MM/DD/YYYY format.\nA: Let's think step-by-step.\n"""
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
                        if re.search("""Therefore, the required date is """, x):
                            preds.append(
                                x[
                                    re.search("""Therefore, the required date is """, x).span(0)[
                                        -1
                                    ] :
                                ]
                            )
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""Therefore, the required date is """, pred):
                            result_counter[chunk_no].update(
                                [
                                    pred[
                                        re.search("""Therefore, the required date is """, x).span(
                                            0
                                        )[-1] :
                                    ]
                                ]
                            )
                time.sleep(60)
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
                time.sleep(10)
            preds = [get_autocot_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
# Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
# Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
# Q1: [string reformat] first day of 2007 in MM/DD/YYYY
# #1: 01/01/2007
# Q2: [arithmetic] What date is one week from 01/01/2007?
# #2: 01/08/2007
# Q3: [EOQ]
# Ans: 01/08/2007
# ----
# Description: Translate English into Pig Latin.
# Input: (English) Sami made his way across the bar and hugged Layla.
# Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
# #1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
# Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
# #2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
# Q3: [string merge] Concatenate #2 into a full sentence.
# #3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
# Q4: [EOQ]
# Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
# ----
# Description: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
# Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
# Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
# #1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
# Q2: [string index] What is the third letter of words in the list in #1?
# #2: ["v", "e", "m", "t", "d"]
# Q3: [string merge] Concatenate #2 with spaces
# #3: "v e m t d"
# Q4: [EOQ]
# Ans: v e m t d
# ----
# Desciption: Take the letters at position 3 of the words in a list of words and concatenate them using a space.
# Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
# Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
# #1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
# Q2: [string index] What is the third letter of words in the list in #1?
# #2: ["r", "a", "i", "o", "u"]
# Q3: [string merge] Concatenate #2 with spaces
# #3: "r a i u o"
# Q4: [EOQ]
# Ans: r a i u o
# ----
# Description: Translate English into Pig Latin.
# Input: (English) Tom is the most popular boy in school.
# Q1: [string split] What are the words in "Tom is the most popular boy in school."?
# #1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
# Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
# #2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
# Q3: [string merge] Concatenate #2 into a full sentence.
# #3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
# Q4: [EOQ]
# Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
# ----
# Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
# Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
# Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
# #1: 06/01/2021
# Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
# #2: Today is 04/01/2021
# Q3: [arithmetic] What date is 24 hours later than today?
# #3: 05/01/2021
# Q4: [EOQ]
# Ans: 05/31/2021
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_string_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Date Understanding"
    task_description = "(Date Understanding) Find the required date in MM/DD/YYYY using information about related events and dates in the input."

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
            answers.extend(
                predict_complete(
                    "Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.",
                    x,
                )
            )
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = few_shot_free_prompt
def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Date Understanding"
    task_description = "(Date Understanding) Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today."

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
        runs = 5
        batch_size = 10
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
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
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """"""


def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_string_prompt
    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
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
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(
                    "Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.",
                    x,
                )
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
            # Report on interpreter performance
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
# human_decomp()

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

        # few_shot_cot(temperature=0.4)
        # few_shot(N=20)
        # auto_cot(temperature=0.3)
        # few_shot(N=5, temperature=0.7)
        # human_decomp(temperature=0.7)
        # auto_decomp(N=20, temperature=0.7)
        # auto_cot()
        # few_shot_cot(temperature=0.3)
