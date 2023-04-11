import argparse
from collections import Counter
import pdb
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

d = datasets.load_dataset("bigbench", "reasoning_about_colored_objects", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "Given a collection of colored objects in the text input, answer the question at the end of the input. This question requires logical, spatial and arithmetic reasoning about colored objects."

io_pairs = [
    (
        """Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?""",
        "pink",
    ),
    (
        """Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?""",
        "yes",
    ),
    (
        """Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?""",
        "fuchsia",
    ),
    (
        """Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?""",
        "3",
    ),
    (
        """Q: On the table, there is a red dog leash, a brown teddy bear, a silver pencil, and a teal paperclip. What color is the paperclip?""",
        "teal",
    ),
    (
        """Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?""",
        "teal",
    ),
    (
        """Q: On the floor, there is one yellow textbook, one teal necklace, three yellow puzzles, three teal puzzles, two purple textbooks, two magenta pencils, one yellow pencil, two yellow necklaces, and one purple necklace. If I remove all the magenta things from the floor, how many puzzles remain on it?""",
        "6",
    ),
    (
        """Q: On the desk, you see several things arranged in a row: a blue textbook, a red jug, a green mug, and a grey keychain. What is the color of the thing directly to the left of the grey thing?""",
        "green",
    ),
    (
        """Q: On the nightstand, you see one gold plate, three orange puzzles, three turquoise pairs of sunglasses, three orange pairs of sunglasses, two gold pairs of sunglasses, two gold dog leashes, two orange dog leashes, one silver dog leash, one turquoise dog leash, one orange plate, and one turquoise plate. If I remove all the plates from the nightstand, how many gold objects remain on it?""",
        "4",
    ),
    (
        """Q: On the floor, there are two silver dog leashes, two red dog leashes, one silver bracelet, one silver keychain, three pink dog leashes, three red bracelets, one red cat toy, and one red keychain. If I remove all the silver objects from the floor, how many cat toys remain on it?""",
        "1",
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
            """Q: On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?
A: Let's think step by step.
According to this question, the color of the stress ball is blue. So the answer is blue.
----
Q: On the table, you see a bunch of objects arranged in a row: a purple paperclip, a pink stress ball, a brown keychain, a green scrunchiephone charger, a mauve fidget spinner, and a burgundy pen. What is the color of the object directly to the right of the stress ball?
A: Let's think step by step.
According to this question, the objects are arranged in a row, from left to right, as follows: (1) a purple paperclip, (2) a pink stress ball, (3) a brown keychain, (4) a green scrunchiephone charger, (5) a mauve fidget spinner, (6) a burgundy pen.
The stress ball is the second object on the list, namely (2). The object that is to the right of the stress ball corresponds to (3), which is a brown keychain.
The color of the keychain is brown. So the answer is brown.

Q: On the nightstand, you see the following items arranged in a row: a teal plate, a burgundy keychain, a yellow scrunchiephone charger, an orange mug, a pink notebook, and a grey cup. How many non-orange items do you see to the left of the teal item?
A: Let's think step by step.
According to this question, the objects are arranged in a row, from left to right, as follows: (1) a teal plate, (2) a burgundy keychain, (3) a yellow scrunchiephone charger, (4) an orange mug, (5) a pink notebook, (6) a grey cup.
The teal plate is the first item, namely (1). There is no item to the left of the teal item.
The number of non-orange items to the left of the teal item is zero. So the answer is zero.
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


auto_cot_corrected_prompt = """Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

First, we need to identify the teal object. We see that the teal object is the bracelet.
Next, we need to find the object to the right of the bracelet. We see that the object to the right of the bracelet is the pink booklet.
Therefore, the color of the object to the right of the teal bracelet is pink.

The final answer is pink.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

First, we see that there are three objects on the floor.
Second, we see that two of the objects are pink and one is magenta.
Third, we see that the jug is one of the two pink objects.
Therefore, the answer is yes, the jug is pink.

The final answer is Yes.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

The right-most item is the a fuchsia sheet of paper.
Therefore, the color of the right-most item is fuchsia.

The final answer is fuchsia.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

There are two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper.
We are removing all the grey items, so that means we are removing three grey mugs, one grey pen, and three grey sheets of paper.
That leaves us with two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, two fuchsia paperclips, and one fuchsia mug.
So, the final answer is that there are three mugs remaining, two brown mugs and one fuchsia mug.

The final answer is 3.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

The teal keychain is the first item, so it's the closest to the left. The turquoise scrunchie is next, followed by the blue bracelet, and finally a pink fidget spinner.
That means the furthest item from the bracelet is the keychain on the extreme left. The color of the keychain is teal.

The final answer is teal.
----
"""

auto_cot_cleaned_prompt = """Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the desk, you see a bunch of things arranged in a row: a red textbook, a yellow pencil, a brown puzzle, a silver stress ball, a teal bracelet, and a pink booklet. What is the color of the thing directly to the right of the teal thing?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

The teal thing is the bracelet.
The bracelet is to the right of the silver stress ball.
So the thing directly to the right of the bracelet is the silver stress ball.

The final answer is silver.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the floor, you see a pink jug, a magenta mug, and a teal scrunchiephone charger. Is the jug pink?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.
First, we see that there are three objects on the floor.
Second, we see that the objects are colored pink, magenta, and teal.
Third, we see that the question is asking if the jug is pink.
From this, we can infer that the answer is yes, the jug is pink.

The final answer is yes.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the table, you see the following items arranged in a row: a purple pair of sunglasses, a black teddy bear, an orange fidget spinner, a teal scrunchiephone charger, a mauve paperclip, and a fuchsia sheet of paper. What is the color of the right-most item?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

The right-most item is a fuchsia sheet of paper.

The final answer is fuchsia.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the desk, I see two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper. If I remove all the grey items from the desk, how many mugs remain on it?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

There are two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, three grey mugs, one grey pen, two fuchsia paperclips, one fuchsia mug, and three grey sheets of paper.
We want to remove all the grey items from the desk.
This means we would remove the three grey mugs, one grey pen, and three grey sheets of paper.
This would leave us with two brown mugs, one brown sheet of paper, one fuchsia sheet of paper, one brown pen, two fuchsia paperclips, and one fuchsia mug.
So, the answer is that there would be five mugs remaining on the desk.

The final answer is five.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the table, there is a red dog leash, a brown teddy bear, a silver pencil, and a teal paperclip. What color is the paperclip?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

First, we see that there are four objects on the table. Two of the objects are different shades of brown, and two of the objects are different shades of blue.
We also see that the brown teddy bear is next to the red dog leash. This means that the paperclip must be next to the silver pencil.
Now, we know that the silver pencil is to the left of the teal paperclip. This means that the teal paperclip is to the right of the silver pencil.
Therefore, the teal paperclip is to the right of the silver pencil and to the left of the red dog leash. The only object that meets this criteria is the teal paperclip.

The final answer is teal.
----
Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects.
Q: On the floor, you see a bunch of items arranged in a row: a teal keychain, a turquoise scrunchiephone charger, a blue bracelet, and a pink fidget spinner. What is the color of the item furthest from the bracelet?
The final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".
A: Let's think step-by-step.

First, we need to find the bracelet. It's the only blue item, so it must be in the middle. That means the keychain is to the left, and the charger is to the right.
Now, we need to find the furthest item from the bracelet. The keychain is one item to the left, and the charger is two items to the right. So the charger is further away and the answer is turquoise.

The final answer is turquoise.
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
            + """\nThe final answer is a color (eg. teal, blue), a number (eg. 1,4) or the words "yes" or "no".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is a color (eg. teal, blue), an integer (eg. 1,4) or the words "yes" or "no".\nA: Let's think step-by-step.\n"""
            % (x)
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
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use arithmetic and algebra functions in one or more of your substeps.
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

few_shot_cot_prompt = few_shot_arithmetic_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Reasoning about colored objects"
    task_description = "(Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects."

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
        prompts = [
            few_shot_cot_prompt
            % (
                description
                + """The final answer is the count of requested items in words, like "six" or "ten".""",
                x,
            )
            for x in chunk
        ]
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
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
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
Q3: [EOQ]
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


def few_shot_pot(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_pot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    task_description = "Select the option that best replaces '()' in each text input given the chocies presented, by solving the arithmetic word problem in the text input. Solving the problem will require reasoning about units of measurement. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("FS-CoT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3, model_name="text-davinci-002"):
    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=2048, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_pot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    task_description = "Select the option that best replaces '()' in each text input given the chocies presented, by solving the arithmetic word problem in the text input. Solving the problem will require reasoning about units of measurement. Write out intermediate calculations as python code and store result as a variable named 'ans'."
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
            new_answers = []
            for ans in answers:
                try:
                    parsed_program = parse_program("Input: dummy\nQ1:" + ans)
                    code_snippet = parsed_program.node_list[0].command_output
                    result = safe_execute(code_snippet)
                except:
                    result = None
                if result:
                    new_answers.append(result)
                else:
                    new_answers.append(
                        ans[re.search("Ans:", ans).span(0)[0] :] if "Ans:" in ans else ans
                    )
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [x.strip() for x in answers]
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
    task_name = "Reasoning about colored objects"
    task_description = "(Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input. THis question requires logical, spatial and arithmetic reasoning about colored objects."

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_name, task_description, io_pairs, N=3)

    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=15):
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
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 15)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in processed_answers:
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers[: len(inputs)]]
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
            new_labels = ["Ans: " + label for label in labels]
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
                # time.sleep(10)
                pdb.set_trace()
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
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


few_shot_human_prompt = """Descripton: Given a collection of colored objects in the text input, answer the question at the end of the input. This question requires logical, spatial and arithmetic reasoning about colored objects. Write code, store the final result as a variable named 'ans' and print it.
Input: On the floor, you see a teal pencil, a yellow textbook, a pink bracelet, a green necklace, a burgundy crayon, and a fuchsia scrunchiephone charger. Is the necklace black?
Q1: [generate python code] Write python code to answer is the necklace black.
#1:
color_object = [("teal", "pencil"), ("yellow", "textbook"), ("pink", "bracelet"), ("green", "necklace"), ("burgundy", "crayon"), ("fuchsia", "scrunchiephone charger")]

for color, object in color_object:
    if object == "necklace":
        if color == "black":
            ans = "Yes"
        else:
            ans = "No"

print(ans)
Q2: [code execute] Execute the python snippet.
color_object = [("teal", "pencil"), ("yellow", "textbook"), ("pink", "bracelet"), ("green", "necklace"), ("burgundy", "crayon"), ("fuchsia", "scrunchiephone charger")]

for color, object in color_object:
    if object == "necklace":
        if color == "black":
            ans = "Yes"
        else:
            ans = "No"

print(ans)
#2: Yes
Q3: [EOQ]
Ans: Yes
----
Descripton: Given a collection of colored objects in the text input, answer the question at the end of the input. This question requires logical, spatial and arithmetic reasoning about colored objects. Write code, store the final result as a variable named 'ans' and print it.
Input: On the nightstand, there are two silver textbooks and three silver cat toys. If I remove all the cat toys from the nightstand, how many fuchsia things remain on it?
Q1: [generate python code] Write python code to find if I remove all the cat toys from the nightstand, how many fuchsia things remain on it
#1:
color_object_count = [("silver", "textbooks", 2), ("silver", "cat toys", 3)]
removed_objects = [("silver", "cat toys", 3)]

def remove_objects(color_object_count, removed_objects):
    color_object_count_updated = []
    for color, object, count in color_object_count:
        for color2, object2, count2 in removed_objects:
            if color == color2 and object == object2:
                count -= count2
        color_object_count_updated.append([color, object, count])
    return color_object_count_updated

color_object_count = remove_objects(color_object_count, removed_objects)

ans = 0
for color, object, count in color_object_count:
    if color == "fuchsia":
        ans = count
print(ans)
Q2: [code execute] Execute the python snippet.
color_object_count = [("silver", "textbooks", 2), ("silver", "cat toys", 3)]
removed_objects = [("silver", "cat toys", 3)]

def remove_objects(color_object_count, removed_objects):
    color_object_count_updated = []
    for color, object, count in color_object_count:
        for color2, object2, count2 in removed_objects:
            if color == color2 and object == object2:
                count -= count2
        color_object_count_updated.append([color, object, count])
    return color_object_count_updated

color_object_count = remove_objects(color_object_count, removed_objects)

ans = 0
for color, object, count in color_object_count:
    if color == "fuchsia":
        ans = count
print(ans)
#2: 0
Q3: [EOQ]
Ans: 0
----
Descripton: Given a collection of colored objects in the text input, answer the question at the end of the input. This question requires logical, spatial and arithmetic reasoning about colored objects. Write code, store the final result as a variable named 'ans' and print it.
Input: On the floor, you see a black plate, a fuchsia pair of sunglasses, and a burgundy booklet. What color is the pair of sunglasses?
Q1: [generate python code] Write python code to find color is the pair of sunglasses.
#1:
color_object = [("black", "plate") , ("fuchsia", "sunglasses"), ("burgundy", "booklet")]

for color, object in color_object:
    if object == "sunglasses":
        ans = color
print(ans)
Q2: [code execute] Execute the python snippet.
color_object = [("black", "plate") , ("fuchsia", "sunglasses"), ("burgundy", "booklet")]

for color, object in color_object:
    if object == "sunglasses":
        ans = color
print(ans)
#2: fuchsia
Q3: [EOQ]
Ans: fuchsia
----
Descripton: %s
Input: %s
Q1:"""


# ----
# Descripton: Given a collection of colored objects in the text input, answer the question at the end of the input. This question requires logical, spatial and arithmetic reasoning about colored objects. Write code, store the final result as a variable named 'ans' and print it.
# Input: On the table, you see a bunch of items arranged in a row: a teal necklace, a purple fidget spinner, and a silver mug. How many non-teal items do you see to the left of the fidget spinner?
# Q1: [generate python code]
# Q2: [code execute]
# Q3: [EOQ]
# Ans:
def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
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
                x = [ex.replace("\nA:", "") for ex in x]
                x = [ex.replace("Q: ", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [x.strip() for x in answers]
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


# human_intervention(0.3, "davinci-codex-002-msft")
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
