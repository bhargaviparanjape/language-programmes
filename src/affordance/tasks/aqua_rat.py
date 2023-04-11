import argparse
from collections import Counter
import re
import time

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
    gpt3,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

data = datasets.load_dataset("aqua_rat", "raw", cache_dir=cache_dir)["validation"]
inputs = [d["question"] + " " + " ".join(d["options"]) for d in data]
labels = [d["correct"] for d in data]
print(len(inputs))

data = datasets.load_dataset("aqua_rat", "raw", cache_dir=cache_dir)["train"]
train_inputs = [d["question"] + " " + " ".join(d["options"]) for d in data]
train_labels = [d["correct"] for d in data]

task_description = "Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer."

io_pairs = [
    (
        """What is the sum of 100 consecutive integers from -49 inclusive, in a increasing order? A)-29 B)50 C)-30 D)30 E)60""",
        "50",
    ),
    (
        """A box contains nine bulbs out of which 4 are defective. If four bulbs are chosen at random, find the probability that atleast one bulb is good? A)125/128 B)125/120 C)125/126 D)125/125 E)125/121""",
        "125/126",
    ),
    (
        """Four of the five parts numbered (a), (b), (c), (d) and (e) are exactly equal. Which of the parts is not equal to the other four? The number of that part is the answer. A)16.80 × 4.50 + 4.4 B)1600 ÷ 40 + 16 × 2.5 C)5.5 × 8.4 + 34.6 D)1620 ÷ 20 – 1 E)1856.95 – 1680.65 – 96.3""",
        "5.5 × 8.4 + 34.6",
    ),
    (
        """An investment of $3000 was made in a certain account and earned interest that was compounded annually. The annual interest rate was fixed for the duration of the investment, and after 12 years the $3000 increased to $12000 by earning interest. In how many years after the initial investment was made the $3000 have increased to $24000 by earning interest at that rate? A)16 B)22 C)20 D)18 E)30""",
        "18",
    ),
    (
        """Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900. What is his profit percent? A)16%% B)88%% C)18%% D)14%% E)28%%""",
        "18%%",
    ),
    (
        """C and D started a business investing Rs. 49,000 and Rs. 35,000 respectively. In what ratio the profit earned after 4 years be divided between C and D respectively? A)7:4 B)7:5 C)6:4 D)5:5 E)None of these""",
        "7:5",
    ),
    (
        """What is the area M of the square with the following coordinates: (x, y), (20, 20), (20, 5), (x, 5)? A)60. B)85. C)125. D)225. E)It cannot be determined from the information given""",
        "225",
    ),
    (
        """Shobha's Mathematics Test had 75 problems i.e. 10 arithmetic, 30 algebra and 35 geometry problems. Although she answered 70%% of the arithmetic, 40%% of the algebra and 60%% 0f the geometry problems correctly, she did not pass the test because she got less than 60%% of the problems right. How many more questions she would have needed to answer correctly to earn a 60%% passing grade? A)5 B)10 C)15 D)20 E)25""",
        "5",
    ),
    (
        """If a, b, and c are consecutive even integers and a < b < c, all of the following must be divisible by 4 EXCEPT A)a + c B)b + c C)ac D)(bc)/2 E)(abc)/4""",
        "b + c",
    ),
    (
        """Two trains, each 160 m long, moving in opposite directions, cross other in 8 sec. If one is moving twice as fast the other, then the speed of the faster train is? A)26 km/hr B)17 km/hr C)60 km/hr D)77 km/hr E)96 km/hr""",
        "96 km/hr",
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
            """What is the sum of 100 consecutive integers from -49 inclusive, in a increasing order? A)-29 B)50 C)-30 D)30 E)60
B
----
A box contains nine bulbs out of which 4 are defective. If four bulbs are chosen at random, find the probability that atleast one bulb is good? A)125/128 B)125/120 C)125/126 D)125/125 E)125/121
C
----
An investment of $3000 was made in a certain account and earned interest that was compounded annually. The annual interest rate was fixed for the duration of the investment, and after 12 years the $3000 increased to $12000 by earning interest. In how many years after the initial investment was made the $3000 have increased to $24000 by earning interest at that rate? A)16 B)22 C)20 D)18 E)30
D
----
Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900. What is his profit percent? A)16%% B)88%% C)18%% D)14%% E)28%%
C
----
C and D started a business investing Rs. 49,000 and Rs. 35,000 respectively. In what ratio the profit earned after 4 years be divided between C and D respectively? A)7:4 B)7:5 C)6:4 D)5:5 E)None of these
B
----
What is the area M of the square with the following coordinates: (x, y), (20, 20), (20, 5), (x, 5)? A)60. B)85. C)125. D)225. E)It cannot be determined from the information given
D
----
Shobha's Mathematics Test had 75 problems i.e. 10 arithmetic, 30 algebra and 35 geometry problems. Although she answered 70%% of the arithmetic, 40%% of the algebra and 60%% 0f the geometry problems correctly, she did not pass the test because she got less than 60%% of the problems right. How many more questions she would have needed to answer correctly to earn a 60%% passing grade? A)5 B)10 C)15 D)20 E)25
A
----
If a, b, and c are consecutive even integers and a < b < c, all of the following must be divisible by 4 EXCEPT A)a + c B)b + c C)ac D)(bc)/2 E)(abc)/4
B
----
Two trains, each 160 m long, moving in opposite directions, cross other in 8 sec. If one is moving twice as fast the other, then the speed of the faster train is? A)26 km/hr B)17 km/hr C)60 km/hr D)77 km/hr E)96 km/hr
E
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
            time.sleep(10)
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


auto_cot_corrected_prompt = """Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900. What is his profit percent? A)16%% B)88%% C)18%% D)14%% E)28%%
The final answer one of the five options.
A: Let's think step-by-step.

Ramu bought an old car for Rs. 42000.
He spent Rs. 13000 on repairs.
That means he paid Rs. 42000 + 13000 = 55000 for the car.
He sold it for Rs. 64900.
His profit was Rs. 64900 - Rs. 55000, or Rs. 9900.
His profit percent was Rs. 9900 / Rs. 55000, which is 18%.
The answer is 18%, or option B.

The final answer is C.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
C and D started a business investing Rs. 49,000 and Rs. 35,000 respectively. In what ratio the profit earned after 4 years be divided between C and D respectively? A)7:4 B)7:5 C)6:4 D)5:5 E)None of these
The final answer one of the five options.
A: Let's think step-by-step.

The first step is to calculate the total profit earned after 4 years. We know that the total investment is Rs. 49,000 + Rs. 35,000 = Rs. 84,000. With no loss of generality, lets assume profit earned is 10%.
10%% of Rs. 84,000 is Rs. 8,400.
This means that the total profit earned is Rs. 8,400.
The next step is to calculate the ratio of profit earned by C and D. We know that C invested Rs. 49,000 and D invested Rs. 35,000.
The ratio of profit earned by C and D is Rs. 8,400 : Rs. 8,400 = 49,000 : 35,000.
This means that the ratio of profit earned by C and D is 7:5.
The final answer is option B: 7:5.

The final answer is C.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
What is the area M of the square with the following coordinates: (x, y), (20, 20), (20, 5), (x, 5)? A)60. B)85. C)125. D)225. E)It cannot be determined from the information given
The final answer one of the five options.
A: Let's think step-by-step.

First, we need to calculate the length of one side of the square. We can do this by finding the difference between the x-coordinates or y-coordinates of the two points on that side.
There are two consecutive coordinates that we know make a side: (20, 20), (20, 5).
In this case, the difference between the y-coordinates of the points (20, 20) and (20, 5) is 20 - 5 = 15.
Next, we need to find the area of the square. The area of a square is equal to the length of one side squared. In this case, that would be 15^2.
Therefore, the answer is D: 225.

The final answer is D.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
Shobha's Mathematics Test had 75 problems i.e. 10 arithmetic, 30 algebra and 35 geometry problems. Although she answered 70%% of the arithmetic, 40%% of the algebra and 60%% 0f the geometry problems correctly, she did not pass the test because she got less than 60%% of the problems right. How many more questions she would have needed to answer correctly to earn a 60%% passing grade? A)5 B)10 C)15 D)20 E)25
The final answer one of the five options.
A: Let's think step-by-step.

Shobha's Mathematics Test had 75 problems in total.
10 arithmetic problems + 30 algebra problems + 35 geometry problems = 75 problems
Shobha answered 70%% of the arithmetic problems correctly.
70%% of 10 arithmetic problems = 7 arithmetic problems
Shobha answered 40%% of the algebra problems correctly.
40%% of 30 algebra problems = 12 algebra problems
Shobha answered 60%% of the geometry problems correctly.
60%% of 35 geometry problems = 21 geometry problems
Shobha needed a 60% passing grade, which would be 45 problems.
7 arithmetic problems + 12 algebra problems + 21 geometry problems = 45 problems
Shobha answered a total of 40 problems correctly.
40 problems < 45 problems
Shobha would need to answer 5 more problems correctly to earn a 60% passing grade.

The final answer is A.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
If a, b, and c are consecutive even integers and a < b < c, all of the following must be divisible by 4 EXCEPT A)a + c B)b + c C)ac D)(bc)/2 E)(abc)/4
The final answer one of the five options.
A: Let's think step-by-step.

a, b, and c are consecutive even integers and a < b < c. Thus b is a+2 and c is a+4. Since a it even, it is divisible by 2. Therefore, we can write a as 2x for some integer x.
Therefore a is 2x, b is 2x + 2, and c is 2x + 4
Let's consider each option.
a + c is 2x + 2x + 4 which is 2x + 4. 4x + 4 is divisible by 4. This eliminates option A.
b + c is 2x + 2 + 2x + 4 which is 4x + 6. 4x is divisible by 4 but 6 is not. Thus, this option is not divisible by 4.
ac is 2x*(2x+4), which is 4x^2 + 8x. 4x^2 + 8x is divisible by 4. This eliminates option C.
bc/2 is (2x + 2)*(2x + 4)/2. This evaluates to 2x^2 + 6x + 4. For both even and odd values of x, this expression is divisible by 4. This eliminates option D.
(abc)/4 is 2x*(2x + 2)*(2x + 4)/4. This evaluates to 2x^3 + 6x^2 + 4x. divisible by 4. As before, for both even and odd values of x, this expression is divisible by 4. This eliminates option E.

The final answer is B.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
Two trains, each 160 m long, moving in opposite directions, cross other in 8 sec. If one is moving twice as fast the other, then the speed of the faster train is? A)26 km/hr B)17 km/hr C)60 km/hr D)77 km/hr E)48 km/hr
The final answer one of the five options.
A: Let's think step-by-step.

We're told that two trains cross each other in 8 seconds.
We're told that one train is moving twice as fast as the other train.
That means the speed of the slower train is 1/2 the speed of the faster train.
We're asked to find the speed of the faster train.
The speed of the faster train is twice the speed of the faster train. Therefore, it covers twice the distance as the slower train.
Since the trains are moving in opposite directions and pass each other, 160 meters is the distance covered by both trains in 8 seconds.
2/3 of this distance is covered by the faster train. Therefore the faster train covers 160*2/3m in 8 seconds. Thus, it covers 320/3m in 8s. Its speed is (320/3)/8, which is 40/3 m/s.
Converting to kilometers per hour, we get 40/3*(18/5). This is 48 km/hr, or option E.

The final answer is E.
----
"""

auto_cot_cleaned_prompt = """Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
What is the sum of 100 consecutive integers from -49 inclusive, in a increasing order? A)-29 B)50 C)-30 D)30 E)60
The final answer one of the five options.
A: Let's think step-by-step.
The first number in the sequence is -49.
The last number in the sequence is 50.
To find the sum of consecutive integers, we need to add up all the numbers in between.
The sum of 100 consecutive integers from -49 inclusive to 50 inclusive is 50 + 49 + 48 + 47 + ... + -50 = 0.

The final answer is C.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
A box contains nine bulbs out of which 4 are defective. If four bulbs are chosen at random, find the probability that atleast one bulb is good? A)125/128 B)125/120 C)125/126 D)125/125 E)125/121
The final answer one of the five options.
A: Let's think step-by-step.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
There are 9 bulbs in the box, and 4 of them are defective. That means that 5 of them are good.
There are 5 good bulbs and 9 total bulbs, so the probability of choosing a good bulb is 5/9.
Now, we need to find the probability of choosing at least one good bulb.
The probability of choosing at least one good bulb is 1 - the probability of choosing all defective bulbs.
There are 4 defective bulbs and 9 total bulbs, so the probability of choosing all defective bulbs is 4/9.
Therefore, the probability of choosing at least one good bulb is 1 - 4/9, or 5/9.

The final answer is D 125/125.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
Four of the five parts numbered (a), (b), (c), (d) and (e) are exactly equal. Which of the parts is not equal to the other four? The number of that part is the answer. A)16.80 × 4.50 + 4.4 B)1600 ÷ 40 + 16 × 2.5 C)5.5 × 8.4 + 34.6 D)1620 ÷ 20 – 1 E)1856.95 – 1680.65 – 96.3
The final answer one of the five options.
A: Let's think step-by-step.
16.80 × 4.50 + 4.4 =
16.80 × 4.50 = 75.60
75.60 + 4.4 = 80.00
1600 ÷ 40 + 16 × 2.5 =
1600 ÷ 40 = 40
40 + 16 × 2.5 = 80
5.5 × 8.4 + 34.6 =
5.5 × 8.4 = 46.20
46.20 + 34.6 = 80.80
1620 ÷ 20 – 1 =
1620 ÷ 20 = 81
81 – 1 = 80
1856.95 – 1680.65 – 96.3 =
1856.95 – 1680.65 = 176.30
176.30 – 96.3 = 80.00
From this, we can see that B, C, D and E are equal, and that A is not equal to the other four.

The final answer is A.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
An investment of $3000 was made in a certain account and earned interest that was compounded annually. The annual interest rate was fixed for the duration of the investment, and after 12 years the $3000 increased to $12000 by earning interest. In how many years after the initial investment was made the $3000 have increased to $24000 by earning interest at that rate? A)16 B)22 C)20 D)18 E)30
The final answer one of the five options.
A: Let's think step-by-step.
The investment of $3,000 was made in a certain account and earned interest that was compounded annually. The annual interest rate was fixed for the duration of the investment, and after 12 years the $3,000 increased to $12,000 by earning interest.
In how many years after the initial investment was made the $3,000 have increased to $24,000 by earning interest at that rate?
A) 16
B) 22
C) 20
D) 18
E) 30

The final answer is C: 20.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900. What is his profit percent? A)16%% B)88%% C)18%% D)14%% E)28%%
The final answer one of the five options.
A: Let's think step-by-step.
Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900.
We want to calculate his profit percent. We can do this by calculating his profit, which is the selling price minus the cost of the car and repairs, and then dividing by the cost of the car and repairs.
Profit = 64900 - (42000 + 13000) = 10000
Profit Percent = (10000 / (42000 + 13000)) x 100 = 23.8%

The final answer is E.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
C and D started a business investing Rs. 49,000 and Rs. 35,000 respectively. In what ratio the profit earned after 4 years be divided between C and D respectively? A)7:4 B)7:5 C)6:4 D)5:5 E)None of these
The final answer one of the five options.
A: Let's think step-by-step.

First, we need to calculate the total profit earned after 4 years. We know that C invested Rs. 49,000 and D invested Rs. 35,000, so the total investment is Rs. 84,000. We also know that the profit was split evenly between C and D, so each person earned half of the total profit.
Now, we need to calculate the total profit. We can do this by multiplying the total investment by the profit percentage. We know that the total investment is Rs. 84,000 and the profit percentage is 4%, so the total profit is Rs. 3,360.
Finally, we need to calculate how much each person earned. We know that the total profit was Rs. 3,360 and that it was split evenly between C and D, so each person earned Rs. 1,680.
Therefore, the ratio of the profit earned by C to the profit earned by D is 7:4.

The final answer is A.
----
Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer.
What is the area M of the square with the following coordinates: (x, y), (20, 20), (20, 5), (x, 5)? A)60. B)85. C)125. D)225. E)It cannot be determined from the information given
The final answer one of the five options.
A: Let's think step-by-step.
First, we need to calculate the length of each side of the square.
The side length is 20 - x.
Now that we know the side length, we can calculate the area.
The area of a square is equal to the length of one side squared.
Therefore, the area of our square is (20 - x)^2.
Now, we need to plug in the values for x.
When x = 20, the area is (20 - 20)^2 = 0.
When x = 5, the area is (20 - 5)^2 = 225.

The final answer is D.
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
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer one of the five options.\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer one of the five options.\nA: Let's think step-by-step.\n"""
            % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        label_dict = ["ABCDE".index(l) for l in labels]
        new_labels = [
            re.split("[ABCDE]\)", inp[re.search("A\)", inp).span(0)[0] :])[1:][label_dict[i]]
            for i, inp in enumerate(inputs)
        ]
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            time.sleep(1)
        preds = [get_autocot_answer(x, answer_prompt="The final answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Perf Array", perf_array)
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use arithmetic and algebra functions in one or more of your substeps.
# Description: Solve the following arithmetic problems on ratios and fractions, , writing out intermediate arithmetic calculations as needed.
# Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number?
#     choice: 29
#     choice: 30
#     choice: 31
#     choice: 32
#     choice: None
# Q1: [algebra] Let the greater part be x. What is the lesser part?
# #1: 49-x
# Q2: [algebra] What is the greater part if increased by 6
# #2: x+6
# Q2: [algebra] What is the lesser part if decreased by 11
# #2: 49-x-11
# Q3: [algebra] What is the ratio of greater to lesser after the change?
# #3: (x+6):(49-x-11)
# Q4: [algebra] Write down the equation to solve for x
# #4: (x+6):(49-x-11) = 9:2
# Q5: [solve] Solve for x: (x+6):(49-x-11) = 9:2
# #5: x = 30
# Q6: [compare] Which option is closes to this answer?
# #6: 30
# Q7: [EOC]
# 30
# ----
# Description: Find the date based on provided information.
# Input: Today is the last day of the first quarter of 2008. What is the date one week from today in MM/DD/YYYY?
# Q1: [search] What is the first quarter of a year?
# #1: The traditional calendar quarters that make up the year are: Dates for Q1: January 1 – March 31. Dates for Q2: April 1 – June 3. Dates for Q3: July 1 – September 30. Dates for Q4: October 1 – December 31.
# Q2: [arithmetic] What is the last day of the first quarter?
# #2: March 31
# Q3: [arithmetic] What day is today?
# #3: March 31, 2008
# Q4: [string reformat] March 31, 2008
# #4: 03/31/2008
# Q5: [arithmetic] What is 1 week from today?
# #5: 04/07/2008
# Q6: [EOC]
# 04/07/2008
# ----
# Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
# Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
# Q1: What is total number of toys that need to be made?
# #1: 400
# Q2: How many workers in total?
# #2: 5
# Q3: [arithmetic] How many toys do 2 of the workers produce?
# #3: 2 workers working for 10 hours making 6 toys per hour produce 2*10*6 = 120 toys.
# Q4: [arithmetic] How many toys do another 2 of workers produce?
# #4: 2 workers working for 10 hours making 4 toys per hour produce 2*10*4 = 80 toys.
# Q5: [arithmetic] How many toys did the last worker make?
# #5: Out of 400 toys, 120+80=200 toys were made by the first 4 workers. The 5th worker didn't finish the job, since 20 toys were still remaining to be made. So they made 400-200-20=180 toys.
# Q6: [arithmetic] How many toys per hour does the fifth worker produce if he worked for 10 hours?
# #6: The 5th worker made 180/10 = 18 toys per hour.
# Q7: [EOC]
# 18
# ----
# Description: What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as needed.
# Input: add 70 to 90, subtract 20 from result, subtract result from 200.
#  choice:130
#  choice:80
#  choice:150
#  choice:100
#  choice:60
# Q1: [arithmetic] add 70 to 90
# #1: 70+90=160
# Q2: [arithmetic] subtract 20 from 160
# #2: 160-20=140
# Q3: [arithmetic] subtract result 140 from 200
# #3: 200-140=60
# Q4: [compare] Which option does the final answer match?
# #4: 60
# Q5: [EOC]
# 60
# ----
# Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
# Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
# Q1: [arithmetic] How much bread does Viola have if he had 167 loafs before and Nancy too 137 of them?
# #1: 167-137=30
# Q2: [EOC]
# 30
# ----
# Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No"
# Input: Take 7 steps. Turn right. Take 8 steps. Turn around.
# Q1: Does this question require vector arithmetic?
# #1: Yes.
# Q2: [subquestion] Which way are you facing when you start? If unknown, assume you face forward?
# #2: Face forward
# Q3: [subquestion] What is the distance moved forward?
# #3: 7 steps
# Q4: [subquestion] What is the distance moved right?
# #4: 8 steps
# Q5: [subquestion] What is the distance moved backward?
# #5: 0 steps
# Q6: [subquestion] What is the distance moved left?
# #6: 0 steps
# Q7: [arithmetic] What is total distance moved from starting point?
# #7: 7 steps vertically, 8 steps horizontally
# Q8: [subquestion] Is the total distance moved, both vertically and horizontally, 0?
# #8: No
# Q9: [EOC]
# No
# ----
# Description:
# Input: If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?
# Q1: [arithmetic] What is the speed of the second train?
# #1: 60/2=30 miles an hour
# Q2: [arithmetic] What is distance travelled by first train?
# #2: 60*3=180 miles
# Q3: [arithmetic] What is distance travelled by first train?
# #3: 30*3=90 miles
# Q4: [subquestion] Are the trains moving towards or away from each other?
# #4: Away from each other.
# Q5: [arithmetic] How far apart are the trains after 3 hours?
# #5: 180+90=270 miles
# Q6: [EOC]
# 270 miles
# ----
# Desciption: %s
# Input: %s
# Q1: """

few_shot_cot_prompt = few_shot_arithmetic_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Arithmetic about ratios"
    task_description = """(Arithmetic about ratios) Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer."""

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_pot_prompt
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
            model=model_name, max_length=1024, temperature=temperature, quote="---", n=1
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
        label_dict = ["ABCDE".index(l) for l in labels]
        new_labels = [
            re.split("[ABCDE]\)", inp[re.search("A\)", inp).span(0)[0] :])[1:][
                label_dict[i]
            ].strip()
            for i, inp in enumerate(inputs)
        ]
        for x in tqdm(chunks(inputs, 5)):
            x = [ex.replace("\nA:", "") for ex in x]
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
            time.sleep(5)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def PoT(temperature=0.3, model_name="text-davinci-002"):
    prompt_4shot = """# Write Python Code to solve the following questions. Store your result as a variable named 'ans'.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.

# Question: In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:
# Answer option: ['A)1 hour', 'B)2 hours', 'C)3 hours', 'D)4 hours', 'E)5 hours']
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_disntace = 600
original_speed = total_disntace / duration
reduced_speed = total_disntace / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]

# Question: M men agree to purchase a gift for Rs. D. If 3 men drop out how much more will each have to contribute towards the purchase of the gift?
# Answer options: ['A)D/(M-3)', 'B)MD/3', 'C)M/(D-3)', 'D)3D/(M2-3M)', 'E)None of these']
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans=simplify(cost_after_dropout - cost_before_dropout)

# Question: A sum of money at simple interest amounts to Rs. 815 in 3 years and to Rs. 854 in 4 years. The sum is:
# Answer option: ['A)Rs. 650', 'B)Rs. 690', 'C)Rs. 698', 'D)Rs. 700', 'E)None of these']
deposit = Symbol('deposit', positive=True)
interest = Symbol('interest', positive=True)
money_in_3_years = deposit + 3 * interest
money_in_4_years = deposit + 4 * interest
solution = solve_it([money_in_3_years - 815, money_in_4_years - 854], [deposit, interest])
ans = solution[deposit]

# Question: Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?
# Answer option: ['A)36', 'B)15', 'C)17', 'D)5', 'E)7']
options = [36, 15, 17, 5, 7]
for option in options:
    if option %% 9 == 0 and option %% 12 == 0:
        ans = option
        break

# Question: 35%% of the employees of a company are men. 60%% of the men in the company speak French and 40%% of the employees of the company speak French. What is %% of the women in the company who do not speak French?
# Answer option: ['A)4%%', 'B)10%%', 'C)96%%', 'D)90.12%%', 'E)70.77%%']
num_women = 65
men_speaking_french = 0.6 * 35
employees_speaking_french = 0.4 * 100
women_speaking_french = employees_speaking_french - men_speaking_french
women_not_speaking_french=num_women - women_speaking_french
ans = women_not_speaking_french / num_women

# Question: In one hour, a boat goes 11 km/hr along the stream and 5 km/hr against the stream. The speed of the boat in still water (in km/hr) is:
# Answer option: ['A)4 kmph', 'B)5 kmph', 'C)6 kmph', 'D)7 kmph', 'E)8 kmph']
boat_speed = Symbol('boat_speed', positive=True)
stream_speed = Symbol('stream_speed', positive=True)
along_stream_speed = 11
against_stream_speed = 5
solution = solve_it([boat_speed + stream_speed - along_stream_speed, boat_speed - stream_speed - against_stream_speed], [boat_speed, stream_speed])
ans = solution[boat_speed]

# Question: The difference between simple interest and C.I. at the same rate for Rs.5000 for 2 years in Rs.72. The rate of interest is?
# Answer option: ['A)10%%', 'B)12%%', 'C)6%%', 'D)8%%', 'E)4%%']
interest_rate = Symbol('interest_rate', positive=True)
amount = 5000
amount_with_simple_interest = amount * (1 + 2 * interest_rate / 100)
amount_with_compound_interest = amount * (1 + interest_rate / 100) ** 2
solution = solve_it(amount_with_compound_interest - amount_with_simple_interest - 72, interest_rate)
ans = solution[interest_rate]

# Question: The area of a rectangle is 15 square centimeters and the perimeter is 16 centimeters. What are the dimensions of the rectangle?
# Answer option: ['A)2&4', 'B)3&5', 'C)4&6', 'D)5&7', 'E)6&8']
width = Symbol('width', positive=True)
height = Symbol('height', positive=True)
area = 15
permimeter = 16
solution = solve_it([width * height - area, 2 * (width + height) - permimeter], [width, height])
ans = (solution[width], solution[height])

# Question: %s
# Answer option: [%s]
"""

    compare_prompt = """
Find the closest options based on the question and prediction.
Question: A company produces 420 units of a particular computer component every month, at a production cost to the company of $110 per component, and sells all of the components by the end of each month. What is the minimum selling price per component that will guarantee that the yearly profit (revenue from sales minus production costs) will be at least $626,400 ?
Options: ['A)226', 'B)230', 'C)240', 'D)260', 'E)280']
Prediction: 234.28571428571428
Closest Option: B
Question: In how many ways can the letters of the word "PROBLEC" be rearranged to make 7 letter words such that none of the letters repeat?
Options: ['A)2!', 'B)3!', 'C)7!', 'D)8!', 'E)9!']
Prediction: 5040
Closest Option: C
Question: An exam is given in a certain class. The average (arithmetic mean) of the highest score and the lowest score is equal to x. If the average score for the entire class is equal to y and there are z students in the class, where z > 5, then in terms of x, y, and z, what is the average score for the class excluding the highest and lowest scorers?
Options: ['A)(zy – 2x)/z', 'B)(zy – 2)/z', 'C)(zx – y)/(z – 2)', 'D)(zy – 2x)/(z -2)', 'E)(zy – x)/(z + 2)']
Prediction: (-2*x + y*z)/(z - 2)
Closest Option: D
Question: Find the total no. of distinct bike no.'s that can beformed using 2 letters followed by 2 no.'s. How many letters need to be distinct?
Options: ["A)74453", "B)64543", "C)74325", "D)65000", "E)97656"]
Prediction = 67600
Closest Option: D
Question: A wire in the shape of rectangle of length 27 cm and breadth 17 cm is rebent to form a square. What will be the measure of each side?
Options: ['A)9', 'B)11', 'C)22', 'D)25', 'E)31']
Prediction = [-21.42428528562855, 21.42428528562855]
Closest Option: C
Question: A point on the edge of a fan blade that is rotating in a plane 10 centimeters from the center of the fan. What is the distance traveled, in centimeters, by this point after 30 seconds when the fan runs at the rate of 300 revolutions per minutes?
Options: ['A)750pi', 'B)1500pi', 'C)1875pi', 'D)3000pi', 'E)7500pi']
Prediction: 9424.77
Closest Option: D
Question: %s
Options: %s
Prediction: %s
Closest Option: """

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name,
            max_length=1024,
            temperature=temperature,
            quote="# Question:",
            n=1,
        )
        prompts = [prompt_4shot % (x[0], x[1]) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        options = [
            repr(
                [
                    opt_name + ")" + opt.strip()
                    for opt_name, opt in zip(
                        ["A", "B", "C", "D", "E"],
                        re.split("[ABCDE]\)", inp[re.search("A\)", inp).span(0)[0] :])[1:],
                    )
                ]
            )
            for inp in inputs
        ]
        questions = [inp[: re.search("A\)", inp).span(0)[0] - 1] for inp in inputs]
        combined_inputs = [(q, o) for q, o in zip(questions, options)]
        for x in tqdm(chunks(combined_inputs, 10)):
            codes = predict(x)
            ans = []
            code_outputs = []
            for code in codes:
                execute_output = safe_execute(code)
                code_output = simplify_ans(execute_output)
                code_outputs.append(code_output)
                if code_output is not None:
                    ans.append(gpt3(compare_prompt % (x[0], x[1], code_output))[0])
                else:
                    ans.append("A")
            answers.extend(ans)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Arithmetic about ratios"
    task_description = """(Arithmetic about ratios) Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer."""

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_pot_prompt
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
        label_dict = ["ABCDE".index(l) for l in labels]
        new_labels = [
            re.split("[ABCDE]\)", inp[re.search("A\)", inp).span(0)[0] :])[1:][
                label_dict[i]
            ].strip()
            for i, inp in enumerate(inputs)
        ]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            # time.sleep(10)
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)

        # Report on interpreter performance
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
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
