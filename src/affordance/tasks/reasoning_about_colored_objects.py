from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'reasoning_about_colored_objects', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

task_description = "Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects."

io_pairs = [("""Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?""",
"pink"),
("""Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?""",
"yes"),
("""Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?""",
"fuchsia"),
("""Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?""",
"3"),
("""Q: On the table, there is a red dog leash, a brown teddy bear, a silver pencil, and a teal paperclip. What color is the paperclip?""",
"teal"),
("""Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?""",
"teal"),
("""Q: On the floor, there is one yellow textbook, one teal necklace, three yellow puzzles, three teal puzzles, two purple textbooks, two magenta pencils, one yellow pencil, two yellow necklaces, and one purple necklace. If I remove all the magenta things from the floor, how many puzzles remain on it?""",
"6"),
("""Q: On the desk, you see several things arranged in a row: a blue textbook, a red jug, a green mug, and a grey keychain. What is the color of the thing directly to the left of the grey thing?""",
"green"),
("""Q: On the nightstand, you see one gold plate, three orange puzzles, three turquoise pairs of sunglasses, three orange pairs of sunglasses, two gold pairs of sunglasses, two gold dog leashes, two orange dog leashes, one silver dog leash, one turquoise dog leash, one orange plate, and one turquoise plate. If I remove all the plates from the nightstand, how many gold objects remain on it?""",
"4"),
("""Q: On the floor, there are two silver dog leashes, two red dog leashes, one silver bracelet, one silver keychain, three pink dog leashes, three red bracelets, one red cat toy, and one red keychain. If I remove all the silver objects from the floor, how many cat toys remain on it?""",
"1")]

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

def reasoning_about_colored_objects():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?
A:
pink
----
Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?
A:
yes
----
Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?
A:
fuchsia
----
Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?
A:
3
----
Q: On the table, there is a red dog leash, a brown teddy bear, a silver pencil, and a teal paperclip. What color is the paperclip?
A:
teal
----
Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?
A:
teal
----
Q: On the floor, there is one yellow textbook, one teal necklace, three yellow puzzles, three teal puzzles, two purple textbooks, two magenta pencils, one yellow pencil, two yellow necklaces, and one purple necklace. If I remove all the magenta things from the floor, how many puzzles remain on it?
A:
6
----
Q: On the desk, you see several things arranged in a row: a blue textbook, a red jug, a green mug, and a grey keychain. What is the color of the thing directly to the left of the grey thing?
A:
green
----
Q: On the nightstand, you see one gold plate, three orange puzzles, three turquoise pairs of sunglasses, three orange pairs of sunglasses, two gold pairs of sunglasses, two gold dog leashes, two orange dog leashes, one silver dog leash, one turquoise dog leash, one orange plate, and one turquoise plate. If I remove all the plates from the nightstand, how many gold objects remain on it?
A:
4
----
Q: On the floor, there are two silver dog leashes, two red dog leashes, one silver bracelet, one silver keychain, three pink dog leashes, three red bracelets, one red cat toy, and one red keychain. If I remove all the silver objects from the floor, how many cat toys remain on it?
A:
1
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
    for io_pair in io_pairs[:3]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)

    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is a color (eg. teal, blue), an integer (eg. 1,4) or the words "yes" or "no".\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            # pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
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


def few_shot_cot(temperature=0.3):
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description + """The final answer is the count of requested items in words, like "six" or "ten".""", x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns

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
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# auto_decomp(10, 0.3)
# affordance(temperature=0.3)
dynamic_few_shot_cot(temperature=0.3, strategy="random")
# few_shot_cot()
# few_shot(N=5, temperature=0.3)
# auto_cot()
