import argparse
from collections import Counter

import datasets
import numpy as np
from prompt_library import (
    few_shot_arithmetic_prompt,
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

d = datasets.load_dataset("bigbench", "navigate", cache_dir=cache_dir)
inputs = d["validation"]["inputs"][:1000]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"][:1000]
labels = [l[0] for l in labels]
# print(len(inputs))


train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "(Navigation) If you follow these instructions, do you return to the starting point? Answer True or False."

io_pairs = [
    ("Q: Turn left. Take 4 steps. Turn around. Take 4 steps.", "True"),
    (
        "Q: Take 7 steps. Take 8 steps. Take 10 steps. Turn around. Turn around. Take 5 steps. Turn around.",
        "False",
    ),
    (
        "Q: Always face forward. Take 8 steps left. Take 2 steps right. Take 6 steps right.",
        "True",
    ),
    (
        "Q: Take 3 steps. Turn right. Turn left. Take 5 steps. Take 10 steps. Take 7 steps. Turn left.",
        "False",
    ),
    ("Q: Take 5 steps. Turn right. Turn left.", "False"),
    (
        "Q: Always face forward. Take 7 steps left. Take 10 steps right. Take 1 step right. Take 4 steps left. Take 1 step backward. Take 3 steps backward.",
        "False",
    ),
    (
        "Q: Always face forward. Take 7 steps left. Take 9 steps backward. Take 5 steps right. Take 5 steps backward.",
        "False",
    ),
    ("Q: Always face forward. Take 6 steps forward. Take 6 steps backward.", "True"),
    (
        "Q: Always face forward. Take 9 steps left. Take 9 steps right. Take 10 steps backward. Take 10 steps forward.",
        "True",
    ),
    ("Q: Always face forward. Take 7 steps right. Take 1 step left.", "False"),
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
            """Q: If you follow these instructions, do you return to the starting point? Turn left. Turn around. Turn left. Take 7 steps. Take 2 steps. Take 4 steps. Take 8 steps.
A: Let's think step by step.
We start at the origin (0, 0), facing the positive y-axis.
(1) Turn left: (0, 0), facing the negative x-axis.
(2) Turn around: (0, 0), facing the positive x-axis.
(3) Turn left: (0, 0), facing the positive y-axis.
(4) Take 7 steps: (0, 7), facing the positive y-axis.
(5) Take 2 steps: (0, 9), facing the positive y-axis.
(6) Take 4 steps: (0, 13), facing the positive y-axis.
(7) Take 8 steps: (0, 21), facing the positive y-axis.
Since (0, 21) is not (0, 0), we are not where we started. So the answer is False
----
Q: If you follow these instructions, do you return to the starting point? Turn around. Take 1 step. Take 6 steps. Turn around. Take 6 steps. Take 9 steps. Take 1 step.
A: Let's think step by step.
We start at the origin (0, 0), facing the positive y-axis.
(1) Turn around: (0, 0), facing the negative y-axis.
(2) Take 1 step: (0, -1), facing the negative y-axis.
(3) Take 6 steps: (0, -7), facing the negative y-axis.
(4) Turn around: (0, -7), facing the positive y-axis.
(5) Take 6 steps: (0, -1), facing the positive y-axis.
(6) Take 9 steps: (0, 8), facing the positive y-axis.
(7) Take 1 step: (0, 9), facing the positive y-axis.
Since (0, 9) is not (0, 0), we are not where we started. So the answer is True
----
Q: If you follow these instructions, do you return to the starting point? Always face forward. Take 2 steps right. Take 9 steps left. Take 7 steps right.
A: Let's think step by step.
We start at the origin (0, 0), facing the positive y-axis.
(1) Always face forward: (0, 0), facing the positive y-axis.
(2) Take 2 steps right: (0, 2), facing the positive y-axis.
(3) Take 9 steps left: (0, -7), facing the positive y-axis.
(4) Take 7 steps right: (0, 7), facing the positive y-axis.
Since (0, 0) is (0, 0), we are indeed where we started. So the answer is True
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


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=200, quote="---", n=1
        )
        prompts = [
            """If you follow these instructions, do you return to the starting point?
Q: Turn left. Take 4 steps. Turn around. Take 4 steps.
A:
True
----
If you follow these instructions, do you return to the starting point?
Q: Take 7 steps. Take 8 steps. Take 10 steps. Turn around. Turn around. Take 5 steps. Turn around.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 8 steps left. Take 2 steps right. Take 6 steps right.
A:
True
----
If you follow these instructions, do you return to the starting point?
Q: Take 3 steps. Turn right. Turn left. Take 5 steps. Take 10 steps. Take 7 steps. Turn left.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Take 5 steps. Turn right. Turn left.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 7 steps left. Take 10 steps right. Take 1 step right. Take 4 steps left. Take 1 step backward. Take 3 steps backward.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 7 steps left. Take 9 steps backward. Take 5 steps right. Take 5 steps backward.
A:
False
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 6 steps forward. Take 6 steps backward.
A:
True
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 9 steps left. Take 9 steps right. Take 10 steps backward. Take 10 steps forward.
A:
True
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 7 steps right. Take 1 step left.
A:
False
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
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
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


auto_cot_corrected_prompt = """If you follow these instructions, do you return to the starting point?
Q: Turn left. Take 4 steps. Turn around. Take 4 steps.
The final answer is "True" or "False".
A: Let's think step-by-step.

After turning left, you would be facing a different direction.
Then, you take 4 steps. At this point, you are 4 steps away from the starting point, and still facing the same direction as when you started.
Then you turn around. This means you are now facing the opposite direction.
Finally, you take 4 steps. This puts you back at the starting point, and facing the same direction as when you started.

The Final answer is "True".
----
If you follow these instructions, do you return to the starting point?
Q: Take 7 steps. Take 8 steps. Take 10 steps. Turn around. Turn around. Take 5 steps. Turn around.
The final answer is "True" or "False".
A: Let's think step-by-step.

You take 7 steps, which puts you 7 steps away from the starting point.
You take 8 steps, which puts you 15 steps away from the starting point.
You take 10 steps, which puts you 25 steps away from the starting point.
You turn around, which means you are now facing the opposite direction.
You are still 25 steps away from the starting point.
You turn around again, which means you are now facing the original direction.
You are still 25 steps away from the starting point.
You take 5 steps, which puts you 30 steps away from the starting point.
You turn around, which means you are now facing the opposite direction.
You are now 20 steps away from the starting point.

The Final answer is False.
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 8 steps left. Take 2 steps right. Take 6 steps right.
The final answer is "True" or "False".
A: Let's think step-by-step.

Always face forward. This means that if you start by facing north, you will continue to face north.
Take 8 steps left. This means that if you start by facing north, you will end up facing west and 8 steps away from the starting point.
Take 2 steps right. This means that if you start by facing east, and walking 2 steps. You are now 6 steps away from the starting point..
Take 6 steps right. This means that if continue to walk east for 6 more steps. You are now back to the starting point.

The final answer is "True"
----
If you follow these instructions, do you return to the starting point?
Q: Take 3 steps. Turn right. Turn left. Take 5 steps. Take 10 steps. Take 7 steps. Turn left.
The final answer is "True" or "False".
A: Let's think step-by-step.

Assume you are facing the front. Take 3 steps. This moves us 3 spaces in the front.
Turn right. This does not move us.
Turn left. This does not move us. You are again facing the front.
Take 5 more steps. This moves us 8 spaces in the front.
Take 10 more steps. This moves us 18 spaces in the front.
Take 7 steps. This moves us 25 spaces in the front.
Turn left. This does not move us.
Finally we are 25 steps in front of the starting point.

The final answer is False.
----
If you follow these instructions, do you return to the starting point?
Q: Take 5 steps. Turn right. Turn left.
The final answer is "True" or "False".
A: Let's think step-by-step.

You take 5 steps. This moves you 5 units in some direction.
You turn right. This changes the direction you're facing, but doesn't change your location.
You turn left. This changes the direction you're facing again, but still doesn't change your location.
In the end, you are 5 steps away from the starting point.

The final answer is False.
----
"""

auto_cot_cleaned_prompt = """If you follow these instructions, do you return to the starting point?
Q: Turn left. Take 4 steps. Turn around. Take 4 steps.
The final answer is "True" or "False".
A: Let's think step-by-step.

Turn left. Take 4 steps.
You are now facing the opposite direction.
Turn around. Take 4 steps.
You have returned to the original starting point. Therefore, the answer is True.

The final answer is True.
----
If you follow these instructions, do you return to the starting point?
Q: Take 7 steps. Take 8 steps. Take 10 steps. Turn around. Turn around. Take 5 steps. Turn around.
The final answer is "True" or "False".
A: Let's think step-by-step.

You take 7 steps.
You take 8 steps.
You take 10 steps.
You turn around.
You turn around again.
You take 5 steps.
You turn around one final time.
Now, let's think about where you are in relation to the starting point. You are 7 + 8 + 10 = 25 steps away from the starting point. However, you've turned around twice, so you've effectively walked back two steps for each turn around. This means you are now only 25 - 4 = 21 steps away from the starting point. Finally, you turn around one last time and take 5 steps. This puts you 21 + 5 = 26 steps away from the starting point, which is exactly where you started, so the answer is "True".

The final answer is True.
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 8 steps left. Take 2 steps right. Take 6 steps right.
The final answer is "True" or "False".
A: Let's think step-by-step.

Always face forward. This means that if you're facing north, you'll stay facing north.
Take 8 steps left. This means that if you're facing north, you'll end up facing west.
Take 2 steps right. This means that if you're facing west, you'll end up facing north.
Take 6 steps right. This means that if you're facing north, you'll end up facing east.
So in the end, you'll be back where you started, facing the same direction. Therefore, the answer is True.

The final answer is True.
----
If you follow these instructions, do you return to the starting point?
Q: Take 3 steps. Turn right. Turn left. Take 5 steps. Take 10 steps. Take 7 steps. Turn left.
The final answer is "True" or "False".
A: Let's think step-by-step.

You take 3 steps. Now you are 3 steps away from the starting point.
You turn right. Now you are still 3 steps away from the starting point, but facing a different direction.
You turn left. Now you are still 3 steps away from the starting point, but facing the original direction.
You take 5 steps. Now you are 8 steps away from the starting point.
You take 10 steps. Now you are 18 steps away from the starting point.
You take 7 steps. Now you are 25 steps away from the starting point.
You turn left. Now you are 25 steps away from the starting point, but facing a different direction.

The final answer is False.
----
If you follow these instructions, do you return to the starting point?
Q: Take 5 steps. Turn right. Turn left.
The final answer is "True" or "False".
A: Let's think step-by-step.

You take 5 steps. This moves you 5 units forward.
You turn right. This changes your direction by 90 degrees.
You turn left. This changes your direction by 90 degrees.
Now, let's consider your final position and direction. You are 5 units forward, and your direction is the same as it was at the start. Therefore, the answer is True.

The final answer is True.
----
If you follow these instructions, do you return to the starting point?
Q: Always face forward. Take 7 steps left. Take 10 steps right. Take 1 step right. Take 4 steps left. Take 1 step backward. Take 3 steps backward.
The final answer is "True" or "False".
A: Let's think step-by-step.

Always face forward. This will keep you oriented in the same direction.
Take 7 steps left.
Take 10 steps right.
Take 1 step right.
Take 4 steps left.
Take 1 step backward.
Take 3 steps backward.
After taking all of these steps, you will be back at the starting point. Therefore, the answer is True.

The final answer is True.
----
"""


def auto_cot(
    temperature=0.3,
    model_name="text-davinci-002",
    predict=True,
    use_corrected=False,
    self_consistency=False,
):
    global auto_cot_cleaned_prompt
    global auto_cot_corrected_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer is "True" or "False".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is "True" or "False".\nA: Let's think step-by-step.\n""" % (x)
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
            + """%s\nThe final answer is "True" or "False".\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("If you follow these instructions, do you return to the starting point?\n", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_arithmetic_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
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
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 20
Q3: [EOQ]
Ans: 20
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 5 steps backward. Take 7 steps backward. Take 8 steps forward. Take 10 steps backward.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 8
steps_backward = 5 + 7 + 10
steps_right = 0
steps_left = 0
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (-2,0)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 22
Q3: [EOQ]
Ans: 22
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 2 steps left. Take 4 steps backward. Take 7 steps left. Take 8 steps right. Take 5 steps forward. Take 1 step left.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 5
steps_backward = 4
steps_right = 8
steps_left = 2 + 7 + 1
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code and get the value of "ans"
#2: (1,-2)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Face forward. Turn left. Turn left. Turn right. Take 4 steps. Turn around. Take 1 step. Take 2 steps. Take 1 step.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 0
steps_backward = 0
steps_right = 1 + 2 + 1
steps_left = 4
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (0,0)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: True
Q4: [EOQ]
Ans: True
----
Descripton: %s
Input: %s
Q1:"""


few_shot_pot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
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
Q1: [generate python code] write down the arithmetic or algebra equations as python code
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
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 20
Q3: [EOQ]
Ans: 20
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 22
Q3: [EOQ]
Ans: 22
----
Description: Solve the following problems on financial data in the form of text and tables, writing out intermediate calculations as python code. Store your result as a variable named 'ans'.
Input: american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $75.0 million and network location intangibles of approximately $72.7 million. the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years. For acquired customer-related and network location intangibles, what is the expected annual amortization expenses, in millions?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
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

few_shot_cot_prompt = few_shot_arithmetic_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Navigation"
    task_description = "(Navigation) If you follow these instructions, do you return to the starting point? Answer in True or False."

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
            model=model_name, max_length=1024, temperature=temperature, quote="---", n=1
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
            # x = [ex.replace("If you follow these instructions, do you return to the starting point?\n", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
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
    task_name = "Navigation"
    task_description = "(Navigation) If you follow these instructions, do you return to the starting point? Answer True/False"

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
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
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


few_shot_human_prompt = """Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 5 steps backward. Take 7 steps backward. Take 8 steps forward. Take 10 steps backward.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 8
steps_backward = 5 + 7 + 10
steps_right = 0
steps_left = 0
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (-2,0)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Face forward. Turn left. Turn left. Turn right. Take 4 steps. Turn around. Take 1 step. Take 2 steps. Take 1 step.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 0
steps_backward = 0
steps_right = 1 + 2 + 1
steps_left = 4
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (0,0)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: True
Q4: [EOQ]
Ans: True
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 6 steps left. Take 3 steps right. Take 4 steps right. Take 9 steps right. Take 10 steps left.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 0
steps_backward = 0
steps_right = 3 + 4 + 9
steps_left = 6 + 10
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (0,0)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: True
Q4: [EOQ]
Ans: True
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 2 steps left. Take 4 steps backward. Take 7 steps left. Take 8 steps right. Take 5 steps forward. Take 1 step left.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 5
steps_backward = 4
steps_right = 8
steps_left = 2 + 7 + 1
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code and get the value of "ans"
#2: (1,-2)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False
----
Description: (Navigation) If you follow these instructions, do you return to the starting point? Answer True or False.
Input: If you follow these instructions, do you return to the starting point?
Always face forward. Take 6 steps. Take 5 steps. Turn right. Take 5 steps.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 6 + 5
steps_backward = 0
steps_right = 5
steps_left = 0
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code and get the value of "ans"
#2: (11,5)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False
----
Description: %s
Input: %s
Q1:"""


def human_intervention(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_human_prompt
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
                x = [ex.replace("repeat with logic:\n\n", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
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
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(ans) for ans in answers]
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


# human_intervention(0.3, "code-davinci-002")
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
