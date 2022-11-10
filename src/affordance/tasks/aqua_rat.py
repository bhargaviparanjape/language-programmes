from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

data = datasets.load_dataset('aqua_rat', 'raw', cache_dir=cache_dir)['validation']
inputs = [d['question'] + " " + " ".join(d['options']) for d in data]
labels = [d['correct'] for d in data]
print(len(inputs))

task_description="Answer the following multiple-choice arithmetic reasoning problems, choosing one of the five options as the final answer."

io_pairs = [
("""What is the sum of 100 consecutive integers from -49 inclusive, in a increasing order? A)-29 B)50 C)-30 D)30 E)60""",
"50"),
("""A box contains nine bulbs out of which 4 are defective. If four bulbs are chosen at random, find the probability that atleast one bulb is good? A)125/128 B)125/120 C)125/126 D)125/125 E)125/121""",
"125/126"),
("""Four of the five parts numbered (a), (b), (c), (d) and (e) are exactly equal. Which of the parts is not equal to the other four? The number of that part is the answer. A)16.80 × 4.50 + 4.4 B)1600 ÷ 40 + 16 × 2.5 C)5.5 × 8.4 + 34.6 D)1620 ÷ 20 – 1 E)1856.95 – 1680.65 – 96.3""",
"5.5 × 8.4 + 34.6"),
("""An investment of $3000 was made in a certain account and earned interest that was compounded annually. The annual interest rate was fixed for the duration of the investment, and after 12 years the $3000 increased to $12000 by earning interest. In how many years after the initial investment was made the $3000 have increased to $24000 by earning interest at that rate? A)16 B)22 C)20 D)18 E)30""",
"18"),
("""Ramu bought an old car for Rs. 42000. He spent Rs. 13000 on repairs and sold it for Rs. 64900. What is his profit percent? A)16%% B)88%% C)18%% D)14%% E)28%%""",
"18%%"),
("""C and D started a business investing Rs. 49,000 and Rs. 35,000 respectively. In what ratio the profit earned after 4 years be divided between C and D respectively? A)7:4 B)7:5 C)6:4 D)5:5 E)None of these""",
"7:5"),
("""What is the area M of the square with the following coordinates: (x, y), (20, 20), (20, 5), (x, 5)? A)60. B)85. C)125. D)225. E)It cannot be determined from the information given""",
"225"),
("""Shobha's Mathematics Test had 75 problems i.e. 10 arithmetic, 30 algebra and 35 geometry problems. Although she answered 70%% of the arithmetic, 40%% of the algebra and 60%% 0f the geometry problems correctly, she did not pass the test because she got less than 60%% of the problems right. How many more questions she would have needed to answer correctly to earn a 60%% passing grade? A)5 B)10 C)15 D)20 E)25""",
"5"),
("""If a, b, and c are consecutive even integers and a < b < c, all of the following must be divisible by 4 EXCEPT A)a + c B)b + c C)ac D)(bc)/2 E)(abc)/4""",
"b + c"),
("""Two trains, each 160 m long, moving in opposite directions, cross other in 8 sec. If one is moving twice as fast the other, then the speed of the faster train is? A)26 km/hr B)17 km/hr C)60 km/hr D)77 km/hr E)96 km/hr""",
"96 km/hr"),
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


def aqua_rat():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""What is the sum of 100 consecutive integers from -49 inclusive, in a increasing order? A)-29 B)50 C)-30 D)30 E)60
B
----
A box contains nine bulbs out of which 4 are defective. If four bulbs are chosen at random, find the probability that atleast one bulb is good? A)125/128 B)125/120 C)125/126 D)125/125 E)125/121
C
----
Four of the five parts numbered (a), (b), (c), (d) and (e) are exactly equal. Which of the parts is not equal to the other four? The number of that part is the answer. A)16.80 × 4.50 + 4.4 B)1600 ÷ 40 + 16 × 2.5 C)5.5 × 8.4 + 34.6 D)1620 ÷ 20 – 1 E)1856.95 – 1680.65 – 96.3
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
""" % x for x in chunk]
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

def auto_cot(temperature=0.3):
    auto_cot_prompt = ""
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer one of the five options.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer one of the five options.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 20
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        label_dict = ["ABCDE".index(l) for l in labels]
        pdb.set_trace()
        new_labels = [re.split('[ABCDE]\)', inp[re.search("A\)", inp).span(0)[0]:])[1:][label_dict[i]] for i, inp in enumerate(inputs)]
        for x in tqdm(chunks(inputs, 20)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
    print("Auto-CoT Performance:")
    print("Perf Array", perf_array)
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot(temperature=0.3)