import argparse
import ast
import json
import pdb
import re
import time
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import urllib.request
from collections import Counter

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks,
                            few_shot_retrieval_prompt, few_shot_code_prompt, 
                            few_shot_arithmetic_prompt, few_shot_string_prompt)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta

d = datasets.load_dataset('bigbench', 'symbol_interpretation', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = "In different SIT worlds, a structure is a sequence of six emojis. You are given the emojis used and thier descriptions. You are given two structures and have to choose the sentence from five provided choices that is consistently describes one structure consistently but not another."

io_pairs = [("""In the SIT-adversarial world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔺 is a red circle;
 🟦 is a blue circle;
 🔴 is a yellow circle;
 🟥 is a red triangle pointing up;
 🟨 is a red triangle pointing down;
 🔻 is a red square;
 🟡 is a blue square;
 _ is a yellow square;
 🔵 is an empty space.

Choose the sentence consistent with the structure 🟥 🔻 🟨 _ 🔴 🟨 and not consistent with 🔺 🟥 🟥 🟡 _ 🟦:

  choice: There are at most two blue circles.

  choice: There is exactly one blue piece.

  choice: There is at least one triangle pointing down.

  choice: There are at least two yellow squares.

  choice: There is exactly one blue circle.""",
"There is at least one triangle pointing down."),
("""In the SIT-adversarial world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔺 is a red circle;
 🟦 is a blue circle;
 🔴 is a yellow circle;
 🟥 is a red triangle pointing up;
 🟨 is a red triangle pointing down;
 🔻 is a red square;
 🟡 is a blue square;
 _ is a yellow square;
 🔵 is an empty space.

Choose the sentence consistent with the structure 🟥 🔴 🟥 🔺 🟥 🟡 and not consistent with 🔴 🔵 _ 🔵 🟨 🔴:

  choice: There are exactly two yellow pieces or exactly one yellow circle.

  choice: There is exactly one red circle and zero red pieces.

  choice: There are zero red squares or exactly one yellow piece.

  choice: There are at most two triangles pointing up and at most one blue square.

  choice: There is at most one triangle pointing down or at least one blue square.""",
"There are exactly two yellow pieces or exactly one yellow circle."),
("""In the SIT-emoji-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🐮 is a red circle;
 🐺 is a blue circle;
 🦁 is a yellow circle;
 🐷 is a red triangle pointing up;
 🐯 is a red triangle pointing down;
 🐱 is a red square;
 🦝 is a blue square;
 🐔 is a yellow square;
 🦄 is an empty space.

Choose the sentence consistent with the structure 🐱 🦁 🦁 🦝 🐔 🦁 and not consistent with 🐷 🐱 🦝 🦁 🐱 🦄:

  choice: There are exactly two triangles pointing up touching a triangle pointing up.

  choice: There is exactly one red piece at the left of a yellow circle.

  choice: There are zero red circles surrounded by squares.

  choice: There are at least two yellow circles surrounded by circles.

  choice: There is exactly one blue square at the right of a triangle pointing up.""",
"There is exactly one red piece at the left of a yellow circle."),
("""In the SIT-emoji-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🐮 is a red circle;
 🐺 is a blue circle;
 🦁 is a yellow circle;
 🐷 is a red triangle pointing up;
 🐯 is a red triangle pointing down;
 🐱 is a red square;
 🦝 is a blue square;
 🐔 is a yellow square;
 🦄 is an empty space.

Choose the sentence consistent with the structure 🐯 🐮 🐱 🐮 🐯 🐯 and not consistent with 🦄 🦁 🦁 🐔 🐯 🦄:

  choice: There are at most two yellow squares or at most two blue circles.

  choice: There is exactly one red square and at least two red circles.

  choice: There is at least one blue circle or exactly two triangles.

  choice: There are at most two yellow circles and exactly one triangle.

  choice: There is at most one red circle and exactly two red pieces.""",
"There is exactly one red square and at least two red circles."),
("""In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🟡 🔵 🔴 🟨 🟦 🔺 and not consistent with 🟡 🟡 🟨 🟥 🟡 🟨:

  choice: There are exactly two H Is and exactly one U I.

  choice: There is at most one U I or at most one H D.

  choice: There are exactly two U Is and at most one H B.

  choice: There are at least two X Ds or exactly one X I.

  choice: There is at least one U B or exactly one Y A.""",
"There are exactly two U Is and at most one H B."),
("""In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🟨 🟨 🔵 🔴 🟨 _ and not consistent with 🔺 🔻 🟦 🔻 🟡 _:

  choice: There are zero Ys touching a D.

  choice: There is at most one U D touching a X B.

  choice: There is at least one B touching a Y A.

  choice: There are at least two Ys surrounded by X Bs.

  choice: There is at least one U B at the left of a Y.""",
"There are zero Ys touching a D."),
("""In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🔻 🔻 🔻 🔴 🔻 🔻 and not consistent with 🟡 🔵 🟨 🔺 🔺 🟥:

  choice: There are exactly two X Is or at most two X Ds.

  choice: There is exactly one B and exactly two X Ds.

  choice: There are zero Ys or at most two Ds.

  choice: There are exactly two U Is or exactly one U D.

  choice: There is at most one H I and at least two Ys A.""",
"There is at most one H I and at least two Ys A."),
("""In the SIT-plain world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a red circle;
 🔵 is a blue circle;
 🟡 is a yellow circle;
 🔺 is a red triangle pointing_up;
 🔻 is a red triangle pointing_down;
 🟥 is a red square;
 🟦 is a blue square;
 🟨 is a yellow square;
 _ is an empty_space.

Choose the sentence consistent with the structure 🔵 🔻 🔺 🔺 🔵 🔵 and not consistent with 🔴 🟦 🔵 🟥 🟦 🔻:

  choice: There are at least two triangles touching a blue circle.

  choice: There is exactly one blue piece at the right of a circle.

  choice: There are at most two triangles pointing up at the right of a square.

  choice: There are at least two red pieces touching a blue square.

  choice: There are exactly two yellow pieces at the right of a yellow piece.""",
"There are at least two triangles touching a blue circle."),
("""In the SIT-plain world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a red circle;
 🔵 is a blue circle;
 🟡 is a yellow circle;
 🔺 is a red triangle pointing_up;
 🔻 is a red triangle pointing_down;
 🟥 is a red square;
 🟦 is a blue square;
 🟨 is a yellow square;
 _ is an empty_space.

Choose the sentence consistent with the structure 🔻 🟡 🟦 🟨 🟡 🟨 and not consistent with 🔺 🟦 🟦 🔺 🔺 🟦:

  choice: There are at most two triangles touching a blue square.

  choice: There is at most one circle at the right of a yellow square.

  choice: There are exactly two triangles at the right of a blue piece.

  choice: There are at least two blue circles touching a blue circle.

  choice: There are exactly two yellow circles at the left of a blue piece.""",
"There are at most two triangles touching a blue square."),
("""In the SIT-tricky world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a der elcric;
 🔵 is an eulb elcric;
 🟡 is a wolley elcric;
 🔺 is a der elgnairt gnitniop pu;
 🔻 is a der elgnairt gnitniop nwod;
 🟥 is a der erauqs;
 🟦 is an eulb erauqs;
 🟨 is a wolley erauqs;
 _ is a ytpme ecaps.

Choose the sentence consistent with the structure 🔵 🔻 🟡 🔴 _ 🟦 and not consistent with 🔴 🔺 🟦 🔻 🟦 🔴:

  choice: There are at least two wolley erauqss.

  choice: There is at least one eulb elcric.

  choice: There are zero wolley eceips.

  choice: There are at least two der elcrics.

  choice: There is at least one der eceip.""",
"There is at least one eulb elcric.")]


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

def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = ["""In the SIT-adversarial world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔺 is a red circle;
 🟦 is a blue circle;
 🔴 is a yellow circle;
 🟥 is a red triangle pointing up;
 🟨 is a red triangle pointing down;
 🔻 is a red square;
 🟡 is a blue square;
 _ is a yellow square;
 🔵 is an empty space.

Choose the sentence consistent with the structure 🟥 🔻 🟨 _ 🔴 🟨 and not consistent with 🔺 🟥 🟥 🟡 _ 🟦:

  choice: There are at most two blue circles.

  choice: There is exactly one blue piece.

  choice: There is at least one triangle pointing down.

  choice: There are at least two yellow squares.

  choice: There is exactly one blue circle.

A:
There is at least one triangle pointing down.
----
In the SIT-adversarial world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔺 is a red circle;
 🟦 is a blue circle;
 🔴 is a yellow circle;
 🟥 is a red triangle pointing up;
 🟨 is a red triangle pointing down;
 🔻 is a red square;
 🟡 is a blue square;
 _ is a yellow square;
 🔵 is an empty space.

Choose the sentence consistent with the structure 🟥 🔴 🟥 🔺 🟥 🟡 and not consistent with 🔴 🔵 _ 🔵 🟨 🔴:

  choice: There are exactly two yellow pieces or exactly one yellow circle.

  choice: There is exactly one red circle and zero red pieces.

  choice: There are zero red squares or exactly one yellow piece.

  choice: There are at most two triangles pointing up and at most one blue square.

  choice: There is at most one triangle pointing down or at least one blue square.

A:
There are exactly two yellow pieces or exactly one yellow circle.
----
In the SIT-emoji-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🐮 is a red circle;
 🐺 is a blue circle;
 🦁 is a yellow circle;
 🐷 is a red triangle pointing up;
 🐯 is a red triangle pointing down;
 🐱 is a red square;
 🦝 is a blue square;
 🐔 is a yellow square;
 🦄 is an empty space.

Choose the sentence consistent with the structure 🐱 🦁 🦁 🦝 🐔 🦁 and not consistent with 🐷 🐱 🦝 🦁 🐱 🦄:

  choice: There are exactly two triangles pointing up touching a triangle pointing up.

  choice: There is exactly one red piece at the left of a yellow circle.

  choice: There are zero red circles surrounded by squares.

  choice: There are at least two yellow circles surrounded by circles.

  choice: There is exactly one blue square at the right of a triangle pointing up.

A:
There is exactly one red piece at the left of a yellow circle.
----
In the SIT-emoji-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🐮 is a red circle;
 🐺 is a blue circle;
 🦁 is a yellow circle;
 🐷 is a red triangle pointing up;
 🐯 is a red triangle pointing down;
 🐱 is a red square;
 🦝 is a blue square;
 🐔 is a yellow square;
 🦄 is an empty space.

Choose the sentence consistent with the structure 🐯 🐮 🐱 🐮 🐯 🐯 and not consistent with 🦄 🦁 🦁 🐔 🐯 🦄:

  choice: There are at most two yellow squares or at most two blue circles.

  choice: There is exactly one red square and at least two red circles.

  choice: There is at least one blue circle or exactly two triangles.

  choice: There are at most two yellow circles and exactly one triangle.

  choice: There is at most one red circle and exactly two red pieces.

A:
There is exactly one red square and at least two red circles.
----
In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🟡 🔵 🔴 🟨 🟦 🔺 and not consistent with 🟡 🟡 🟨 🟥 🟡 🟨:

  choice: There are exactly two H Is and exactly one U I.

  choice: There is at most one U I or at most one H D.

  choice: There are exactly two U Is and at most one H B.

  choice: There are at least two X Ds or exactly one X I.

  choice: There is at least one U B or exactly one Y A.

A:
There are exactly two U Is and at most one H B.
----
In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🟨 🟨 🔵 🔴 🟨 _ and not consistent with 🔺 🔻 🟦 🔻 🟡 _:

  choice: There are zero Ys touching a D.

  choice: There is at most one U D touching a X B.

  choice: There is at least one B touching a Y A.

  choice: There are at least two Ys surrounded by X Bs.

  choice: There is at least one U B at the left of a Y.

A:
There are zero Ys touching a D.
----
In the SIT-name-agnostic world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a X D;
 🔵 is a H D;
 🟡 is a U D;
 🔺 is a X Y G;
 🔻 is a X Y A;
 🟥 is a X B;
 🟦 is a H B;
 🟨 is a U B;
 _ is a T.

Choose the sentence consistent with the structure 🔻 🔻 🔻 🔴 🔻 🔻 and not consistent with 🟡 🔵 🟨 🔺 🔺 🟥:

  choice: There are exactly two X Is or at most two X Ds.

  choice: There is exactly one B and exactly two X Ds.

  choice: There are zero Ys or at most two Ds.

  choice: There are exactly two U Is or exactly one U D.

  choice: There is at most one H I and at least two Ys A.

A:
There is at most one H I and at least two Ys A.
----
In the SIT-plain world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a red circle;
 🔵 is a blue circle;
 🟡 is a yellow circle;
 🔺 is a red triangle pointing_up;
 🔻 is a red triangle pointing_down;
 🟥 is a red square;
 🟦 is a blue square;
 🟨 is a yellow square;
 _ is an empty_space.

Choose the sentence consistent with the structure 🔵 🔻 🔺 🔺 🔵 🔵 and not consistent with 🔴 🟦 🔵 🟥 🟦 🔻:

  choice: There are at least two triangles touching a blue circle.

  choice: There is exactly one blue piece at the right of a circle.

  choice: There are at most two triangles pointing up at the right of a square.

  choice: There are at least two red pieces touching a blue square.

  choice: There are exactly two yellow pieces at the right of a yellow piece.

A:
There are at least two triangles touching a blue circle.
----
In the SIT-plain world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a red circle;
 🔵 is a blue circle;
 🟡 is a yellow circle;
 🔺 is a red triangle pointing_up;
 🔻 is a red triangle pointing_down;
 🟥 is a red square;
 🟦 is a blue square;
 🟨 is a yellow square;
 _ is an empty_space.

Choose the sentence consistent with the structure 🔻 🟡 🟦 🟨 🟡 🟨 and not consistent with 🔺 🟦 🟦 🔺 🔺 🟦:

  choice: There are at most two triangles touching a blue square.

  choice: There is at most one circle at the right of a yellow square.

  choice: There are exactly two triangles at the right of a blue piece.

  choice: There are at least two blue circles touching a blue circle.

  choice: There are exactly two yellow circles at the left of a blue piece.

A:
There are at most two triangles touching a blue square.
----
In the SIT-tricky world a structure is a sequence of six emojis.
Hereafter are reported the emojis used along with their descriptions.
 🔴 is a der elcric;
 🔵 is an eulb elcric;
 🟡 is a wolley elcric;
 🔺 is a der elgnairt gnitniop pu;
 🔻 is a der elgnairt gnitniop nwod;
 🟥 is a der erauqs;
 🟦 is an eulb erauqs;
 🟨 is a wolley erauqs;
 _ is a ytpme ecaps.

Choose the sentence consistent with the structure 🔵 🔻 🟡 🔴 _ 🟦 and not consistent with 🔴 🔺 🟦 🔻 🟦 🔴:

  choice: There are at least two wolley erauqss.

  choice: There is at least one eulb elcric.

  choice: There are zero wolley eceips.

  choice: There are at least two der elcrics.

  choice: There is at least one der eceip.

A:
There is at least one eulb elcric.
----
%s
""" % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """"""
def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    auto_cot_prompt = ""
    for io_pair in io_pairs[:3]:
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer is one of the five choices.\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Anachronisms\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is one of the five choices.\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use arithmetic and algebra functions in one or more of your substeps.
Description: Solve the following arithmetic problems on ratios and fractions, , writing out intermediate arithmetic calculations as needed.
Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number?
    choice: 29 
    choice: 30 
    choice: 31 
    choice: 32 
    choice: None
Q1: [algebra] Let the greater part be x. What is the lesser part?
#1: 49-x
Q2: [algebra] What is the greater part if increased by 6
#2: x+6
Q2: [algebra] What is the lesser part if decreased by 11
#2: 49-x-11
Q3: [algebra] What is the ratio of greater to lesser after the change?
#3: (x+6):(49-x-11)
Q4: [algebra] Write down the equation to solve for x
#4: (x+6):(49-x-11) = 9:2
Q5: [solve] Solve for x: (x+6):(49-x-11) = 9:2
#5: x = 30
Q6: [compare] Which option is closes to this answer?
#6: 30
Q7: [EOC]
30
----
Description: Find the date based on provided information.
Input: Today is the last day of the first quarter of 2008. What is the date one week from today in MM/DD/YYYY?
Q1: [search] What is the first quarter of a year?
#1: The traditional calendar quarters that make up the year are: Dates for Q1: January 1 – March 31. Dates for Q2: April 1 – June 3. Dates for Q3: July 1 – September 30. Dates for Q4: October 1 – December 31.
Q2: [arithmetic] What is the last day of the first quarter?
#2: March 31
Q3: [arithmetic] What day is today?  
#3: March 31, 2008
Q4: [string reformat] March 31, 2008
#4: 03/31/2008
Q5: [arithmetic] What is 1 week from today?
#5: 04/07/2008
Q6: [EOC]
04/07/2008
----
Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
Q1: What is total number of toys that need to be made?
#1: 400
Q2: How many workers in total?
#2: 5
Q3: [arithmetic] How many toys do 2 of the workers produce?
#3: 2 workers working for 10 hours making 6 toys per hour produce 2*10*6 = 120 toys.
Q4: [arithmetic] How many toys do another 2 of workers produce?
#4: 2 workers working for 10 hours making 4 toys per hour produce 2*10*4 = 80 toys.
Q5: [arithmetic] How many toys did the last worker make?
#5: Out of 400 toys, 120+80=200 toys were made by the first 4 workers. The 5th worker didn't finish the job, since 20 toys were still remaining to be made. So they made 400-200-20=180 toys.
Q6: [arithmetic] How many toys per hour does the fifth worker produce if he worked for 10 hours?
#6: The 5th worker made 180/10 = 18 toys per hour.
Q7: [EOC]
18
----
Description: What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as needed.
Input: add 70 to 90, subtract 20 from result, subtract result from 200.
 choice:130
 choice:80
 choice:150
 choice:100
 choice:60
Q1: [arithmetic] add 70 to 90
#1: 70+90=160
Q2: [arithmetic] subtract 20 from 160
#2: 160-20=140
Q3: [arithmetic] subtract result 140 from 200
#3: 200-140=60
Q4: [compare] Which option does the final answer match?
#4: 60
Q5: [EOC]
60
----
Description: Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.
Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
Q1: [arithmetic] How much bread does Viola have if he had 167 loafs before and Nancy too 137 of them?
#1: 167-137=30
Q2: [EOC]
30
----
Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No"
Input: Take 7 steps. Turn right. Take 8 steps. Turn around.
Q1: Does this question require vector arithmetic?
#1: Yes.
Q2: [subquestion] Which way are you facing when you start? If unknown, assume you face forward?
#2: Face forward
Q3: [subquestion] What is the distance moved forward?
#3: 7 steps
Q4: [subquestion] What is the distance moved right?
#4: 8 steps
Q5: [subquestion] What is the distance moved backward?
#5: 0 steps
Q6: [subquestion] What is the distance moved left?
#6: 0 steps
Q7: [arithmetic] What is total distance moved from starting point?
#7: 7 steps vertically, 8 steps horizontally 
Q8: [subquestion] Is the total distance moved, both vertically and horizontally, 0? 
#8: No
Q9: [EOC]
No
----
Description: 
Input: If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?
Q1: [arithmetic] What is the speed of the second train? 
#1: 60/2=30 miles an hour
Q2: [arithmetic] What is distance travelled by first train?
#2: 60*3=180 miles
Q3: [arithmetic] What is distance travelled by first train?
#3: 30*3=90 miles
Q4: [subquestion] Are the trains moving towards or away from each other?
#4: Away from each other.
Q5: [arithmetic] How far apart are the trains after 3 hours?
#5: 180+90=270 miles
Q6: [EOC]
270 miles
----
Desciption: %s
Input: %s
Q1: """



def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Symbol interpretation"
    task_description = "(Symbol interpretation) In different SIT worlds, a structure is a sequence of six emojis. You are given the emojis used and thier descriptions. You are given two structures and have to choose the sentence from five provided choices that is consistently describes one structure consistently but not another."

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
        gpt3 = OpenAIModel(model=model_name,  max_length=1024, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description + """Clue: Translating the two subsequences will help.""", x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
            time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    
    global few_shot_cot_prompt
    task_name = "Symbol interpretation"
    task_description = "(Symbol interpretation) In different SIT worlds, a structure is a sequence of six emojis. You are given the emojis used and thier descriptions. You are given two structures and have to choose the sentence from five provided choices that is consistently describes one structure consistently but not another."

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

    interpreter = TopDownVisitorBeta()

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action='store_true')

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[:args.num_dev_examples]
    labels = labels[:args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(args.temperature, args.model_name, predict=True, use_corrected=False, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)