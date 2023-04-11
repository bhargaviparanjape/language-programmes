import argparse
import re

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

data = datasets.load_dataset("gsm8k", "main", cache_dir=cache_dir)["test"]
inputs = [d["question"] for d in data]
labels = [d["answer"].split("#### ")[-1] for d in data]


train_data = datasets.load_dataset("gsm8k", "main", cache_dir=cache_dir)["train"]
train_inputs = [d["question"] for d in data]
train_labels = [d["answer"].split("#### ")[-1] for d in data]


task_description = "Answer the following middle school math word problems, which require multi-step arithmetic reasoning."

io_pairs = [
    (
        """Mason is cleaning out all the junk in his attic. 20%% of the items are useful, 10%% are valuable heirlooms, and 70%% are junk. If Marcus's attic has 8 useful items in it, how many junk items does it have?""",
        "28",
    ),
    (
        """A gecko eats 70 crickets every three days.  The first day she eats 30%% of the crickets.  The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?""",
        "34",
    ),
    (
        """My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?""",
        "118",
    ),
    (
        """Tim buys a cabinet for $1200 and gets a 15%% discount.  How much did he pay?""",
        "1020",
    ),
    (
        """Grant scored 10 points higher on his math test than John.  John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?""",
        "100",
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
            """Mason is cleaning out all the junk in his attic. 20%% of the items are useful, 10%% are valuable heirlooms, and 70%% are junk. If Marcus's attic has 8 useful items in it, how many junk items does it have?
28
----
A gecko eats 70 crickets every three days.  The first day she eats 30%% of the crickets.  The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?
34
----
My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?
118
----
Tim buys a cabinet for $1200 and gets a 15%% discount.  How much did he pay?
1020
----
Grant scored 10 points higher on his math test than John.  John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?
100
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
        preds = [re.search("[0-9,.]+", p).group(0).replace(",", "") for p in preds]
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


auto_cot_corrected_prompt = """Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Mason is cleaning out all the junk in his attic. 20%% of the items are useful, 10%% are valuable heirlooms, and 70%% are junk. If Marcus's attic has 8 useful items in it, how many junk items does it have?
A: Let's think step-by-step.

If Marcus had a total of x items in his attic, 20%% of the items are useful, 10%% are valuable heirlooms, and 70%% are junk.
We need to figure out what x is, given that 20%% of x is 8.
We can do this by dividing 8 by 0.2 (20%%) to get 40. This means that there are 40 items in total the attic.
Finally, to find the number of junk items, we need to figure out what 70% of 40 is. We can do this by multiplying 40 by 0.7 to get 28.
This means that there are 28 junk items in the attic.

The final answer is 28.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
A gecko eats 70 crickets every three days.  The first day she eats 30%% of the crickets.  The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?
A: Let's think step-by-step.

On the first day, the gecko eats 30% of 70 crickets, which is 21 crickets.
On the second day, she eats 6 less than that, so she eats 15 crickets.
On the third day, she eats the remaining crickets. That will be 70 - 21 - 15, which is 34.

The final answer is 34.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?
A: Let's think step-by-step.

First, we need to figure out how many large and small ceiling lights there are.
We know that there are 12 medium ceiling lights, so if there are twice as many large ones, that means there are 24 large ones.
We also know that there are ten more small ones than medium ones, so that means there are 22 small ones.
Now that we know how many of each type there are, we can figure out how many bulbs we need.
Remember, small ones require 1 bulb, medium ones require 2, and large ones require 3.
That means we need 22 bulbs for the small ones, 24 bulbs for the medium ones, and 72 bulbs for the large ones.
All together, we need 118 bulbs.

The final answer is 118.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Tim buys a cabinet for $1200 and gets a 15%% discount.  How much did he pay?
A: Let's think step-by-step.

To calculate the discount Tim gets, we find 15%% of 1200. This is 180. Subtracting the discount amount from 1200 gets us 1020. Thus, Tim paid 1020.

The final answer is 1020
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Grant scored 10 points higher on his math test than John.  John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?
A: Let's think step-by-step.

Hunter scored a 45 on his math test. John received twice as many points as Hunter. Thus John got 90. Grant scored 10 points higher on his math test than John. So Grant got a 100 on the test.

The final answer is 100
----
"""

auto_cot_cleaned_prompt = """Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Mason is cleaning out all the junk in his attic. 20%% of the items are useful, 10%% are valuable heirlooms, and 70%% are junk. If Marcus's attic has 8 useful items in it, how many junk items does it have?
A: Let's think step-by-step.

20% of 8 is 1.6, so we know there are 8 - 1.6 = 6.4 junk items in the attic.

The final answer is 6.4.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
A gecko eats 70 crickets every three days.  The first day she eats 30%% of the crickets.  The second day she eats 6 less than the first, and the third day she finishes up the remaining crickets.  How many crickets does she eat on the third day?
A: Let's think step-by-step.

The gecko eats 70 crickets every three days.
The first day, she eats 30% of 70, which is 21 crickets.
The second day, she eats 6 less than the first, so she eats 15 crickets.
The third day, she finishes up the remaining crickets, so she eats 34 crickets.

The final answer is 34 crickets.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
My new house has 12 medium ceiling lights but I have seen small and large ceiling lights in other rooms. The small ones require 1 bulb, the medium ones require 2, and the large ones need 3 bulbs. How many bulbs should I buy if my wife says she saw twice as many large ceiling lights as medium ceiling lights and ten more small lights than medium ones?
A: Let's think step-by-step.

There are 12 medium ceiling lights, so that's 24 bulbs right there.
There are twice as many large ceiling lights as medium ceiling lights. That's 24 more bulbs, for a total of 48.
There are ten more small ceiling lights than medium ceiling lights. That's 20 more bulbs, for a total of 68.
So, you should buy 68 light bulbs.

The final answer is 68 light bulbs.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Tim buys a cabinet for $1200 and gets a 15%% discount.  How much did he pay?
A: Let's think step-by-step.

First, we need to calculate the 15% discount, which would be $1200 * 0.15 = $180.
Then, we subtract the discount from the original price of the cabinet, so $1200 - $180 = $1020.
This means that Tim paid $1020 for the cabinet.

The final answer is $1020.
----
Answer the following middle school math word problems, which require multi-step arithmetic reasoning.
Grant scored 10 points higher on his math test than John.  John received twice as many points as Hunter who scored a 45 on his math test.  What was Grant's test score?
A: Let's think step-by-step.

Grant scored 10 points higher than John. So, if we know John's score, we can find Grant's. John received twice as many points as Hunter, who scored a 45. So, we can set up the following equation:
2(Hunter's score) = John's score
Solving for Hunter's score, we get:
2(45) = John's score
John's score = 90
Now we can plug John's score into our original equation:
Grant's score = 10 + John's score
Grant's score = 10 + 90
Grant's score = 100

The final answer is 100.
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
        for x in tqdm(chunks(inputs, 10)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x) for x in answers]
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


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Grade school math"
    task_description = "(Grade school math) Answer the following middle school math word problems, which require multi-step arithmetic reasoning."

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
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
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
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Grade school math"
    task_description = """(Grade school math) Answer the following middle school math word problems, which require multi-step arithmetic reasoning."""

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
