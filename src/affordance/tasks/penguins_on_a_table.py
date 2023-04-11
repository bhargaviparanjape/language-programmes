import argparse
from collections import Counter
import re

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

d = datasets.load_dataset("bigbench", "penguins_in_a_table", cache_dir=cache_dir)
inputs = d["validation"]["inputs"] + d["train"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"] + d["train"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\n\nWe then delete the penguin named Bernard from the table.\nHow many penguins are less than 8 years old and weight less than 12 kg?""",
        "1",
    ),
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\nWhich penguin is older than Bernard and younger than Gwen?""",
        "Louis",
    ),
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\nAnd here is a similar table, but listing giraffes:\n\nname, age, height (cm), weight (kg)\nJody, 5, 430, 620\nGladys, 10, 420, 590\nMarian, 2, 310, 410\nDonna, 9, 440, 650\nWhat is the cumulated age of the penguins?""",
        "29",
    ),
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\n\nWe now add a penguin to the table:\nJames, 12, 90, 12\n\nWe then delete the penguin named Bernard from the table.\nWhat is the name of the last penguin sorted by alphabetic order?""",
        "Vincent",
    ),
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\nWhat animals are listed in the table?""",
        "penguins",
    ),
    (
        """Here is a table where the first line is a header and each subsequent line is a penguin:\n\nname, age, height (cm), weight (kg)\nLouis, 7, 50, 11\nBernard, 5, 80, 13\nVincent, 9, 60, 11\nGwen, 8, 70, 15\n\nFor example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.\n\nAnd here is a similar table, but listing giraffes:\n\nname, age, height (cm), weight (kg)\nJody, 5, 430, 620\nGladys, 10, 420, 590\nMarian, 2, 310, 410\nDonna, 9, 440, 650\nWhich is the oldest giraffe?""",
        "Gladys",
    ),
]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Penguins in a table"
task_description = """(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits."""


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
            """Q: Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  We now add a penguin to the table:
James, 12, 90, 12
How many penguins are less than 8 years old?
A: Let's think step by step.
This question focuses on age. We know the following: Louis is 7 years old, Bernard is 5 years old, Vincent is 9 years old, and Gwen is 8 years old.
Now, we add James to this table: James is 12 years old.
The penguins that are less than 8 years old are Louis and Bernard.
There are 2 penguins less than 8 years old. So the answer is 2.
----
Q: Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  Which is the youngest penguin?
A: Let's think step by step.
This question focuses on age. We know the following: Louis is 7 years old, Bernard is 5 years old, Vincent is 9 years old, and Gwen is 8 years old.
According to the table, Bernard (5) is the youngest amongst them.
The youngest penguin is Bernard. So the answer is Bernard.
----
Q: Here is a table where the first line is a header and each subsequent line is a penguin:  name, age, height (cm), weight (kg) Louis, 7, 50, 11 Bernard, 5, 80, 13 Vincent, 9, 60, 11 Gwen, 8, 70, 15  For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.  What is the name of the second penguin sorted by alphabetic order?
A: Let's think step by step.
This question focuses on the name. We know the following: The names of the penguin in the table are Louis, Bernard, Vincent, and Gwen.
When we sort their names alphabetically, we get Bernard, Gwen, Louis, Vincent.
The name of the second penguin sorted by alphabetical order is Gwen.
The name of the second penguin sorted by alphabetic order is Gwen. So the answer is Gwen.
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
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
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


auto_cot_corrected_prompt = """"""

auto_cot_cleaned_prompt = """(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.


We then delete the penguin named Bernard from the table.
How many penguins are less than 8 years old and weight less than 12 kg?
A: Let's think step-by-step.
There are four penguins in the table. We delete one, so there are three left. Two of them are less than 8 years old (Louis and Gwen), and two of them weigh less than 12 kg (Louis and Vincent). So the answer is two.

The final answer is two
----
(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
, 'Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.

Which penguin is older than Bernard and younger than Gwen?
A: Let's think step-by-step.

First, we need to find the age of Bernard. We can see that Bernard is the second penguin listed in the table, and his age is 5.
Next, we need to find the age of Gwen. We can see that Gwen is the fourth penguin listed in the table, and her age is 8.
Now, we need to find a penguin who is older than Bernard and younger than Gwen. We can see that Louis is the first penguin listed in the table, and his age is 7. Therefore, Louis is the penguin we are looking for.

The final answer is Louis
----
(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
, 'Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.

And here is a similar table, but listing giraffes:

name, age, height (cm), weight (kg)
Jody, 5, 430, 620
Gladys, 10, 420, 590
Marian, 2, 310, 410
Donna, 9, 440, 650
What is the cumulated age of the penguins?
A: Let's think step-by-step.

First, we need to find the age of each penguin. We can see that in the table, next to each penguin's name.
So, the age of Louis is 7, the age of Bernard is 5, the age of Vincent is 9, and the age of Gwen is 8.
Next, we need to add those ages together.
7 + 5 + 9 + 8 = 29
Therefore, the cumulated age of the penguins is 29.

The final answer is 29
----
(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
, 'Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.


We now add a penguin to the table:
James, 12, 90, 12

We then delete the penguin named Bernard from the table.
What is the name of the last penguin sorted by alphabetic order?
A: Let's think step-by-step.

The last penguin sorted by alphabetic order is James.

The final answer is James
----
(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
, 'Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.

What animals are listed in the table?
A: Let's think step-by-step.

The table lists penguins.

The final answer is penguins
----
(Penguins in a table) Answer simple questions about the animals contained in a table, or multiple tables. Use information given both in the table(s) and natural language, selecting and extracting the relevant bits.
, 'Here is a table where the first line is a header and each subsequent line is a penguin:

name, age, height (cm), weight (kg)
Louis, 7, 50, 11
Bernard, 5, 80, 13
Vincent, 9, 60, 11
Gwen, 8, 70, 15

For example: the age of Louis is 7, the weight of Gwen is 15 kg, the height of Bernard is 80 cm.

And here is a similar table, but listing giraffes:

name, age, height (cm), weight (kg)
Jody, 5, 430, 620
Gladys, 10, 420, 590
Marian, 2, 310, 410
Donna, 9, 440, 650
Which is the oldest giraffe?
A: Let's think step-by-step.

First, we need to find the table that lists giraffes. We can see that the table with giraffes is the second table, so we'll focus on that one.
Next, we need to find the age of each giraffe. We can see that the age is listed in the second column of the table.
Now, we just need to compare the ages of the giraffes to find the oldest one. We can see that Gladys is 10 years old, which is the oldest age listed in the table. Therefore, Gladys is the oldest giraffe.

The final answer is Gladys
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
    for io_pair in io_pairs[:10]:
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompt = """%s\n""" % task_description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
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
    f.write("Penguins in a table\n\n")
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
                x = [ex.replace(" Answer:", "") for ex in x]
                answers.extend(predict(x))
            preds = [get_autocot_answer(x, answer_prompt="The final answer is ") for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = few_shot_free_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt

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

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

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
            x = [ex.replace(" Answer:", "") for ex in x]
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.

Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input:  In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:  A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_disntace = 600
original_speed = total_disntace / duration
reduced_speed = total_disntace / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2:
1.0
Q3: [compare] Which of the options among A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours is most similar to the answer?
#3: A
Q4: [EOQ]
Ans: A
----
Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: M men agree to purchase a gift for Rs. D. If 3 men drop out how much more will each have to contribute towards the purchase of the gift?  A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans=simplify(cost_after_dropout - cost_before_dropout)
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 3*D/(M*(M - 3))
Q3: [compare] Which of the options among A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these is most similar to the answer?
#3: D
Q4: [EOQ]
Ans: D
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [EOQ]
Ans:18
----
Description: Solve the following problems on financial data in the form of text and tables, writing out intermediate calculations as python code. Store your result as a variable named 'ans'.
Input: american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $75.0 million and network location intangibles of approximately $72.7 million. the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years. For acquired customer-related and network location intangibles, what is the expected annual amortization expenses, in millions?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 7.385
Q3: [EOQ]
Ans: 7.385
----
Descripton: %s
Input: %s
Q1:"""


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

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

    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

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
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            x = [ex.replace(" Answer:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        # Report on interpreter performance
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)

    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


human_decomp()

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
