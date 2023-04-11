import argparse

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
    substring_match_v2,
)
from word2number import w2n

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "object_counting", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
# print(len(inputs))

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]


task_description = (
    "Object counting: You are given a bunch of items and asked to count a specific kind of item."
)

io_pairs = [
    (
        "Q: I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano. How many musical instruments do I have?",
        "thirteen",
    ),
    ("Q: I have a mouse, a goat, and two fish. How many animals do I have?", "four"),
    (
        "Q: I have a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. How many fruits do I have?",
        "ten",
    ),
    (
        "Q: I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone. How many musical instruments do I have?",
        "eight",
    ),
    (
        "Q: I have three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin. How many musical instruments do I have?",
        "five",
    ),
    (
        "Q: I have five stoves, a fridge, a lamp, a microwave, a table, and an oven. How many objects do I have?",
        "ten",
    ),
    (
        "Q: I have a cabbage, a stalk of celery, a carrot, a lettuce head, a yam, three heads of broccoli, three potatoes, and two cauliflowers. How many vegetables do I have?",
        "thirteen",
    ),
    (
        "Q: I have a donkey, a cat, two bears, a pig, and a goat. How many animals do I have?",
        "six",
    ),
    (
        "Q: I have a frog, a duck, a bear, two donkeys, a fish, a goat, and a rabbit. How many animals do I have?",
        "eight",
    ),
    (
        "Q: I have a duck, a donkey, a cow, a bear, a dog, and a mouse. How many animals do I have?",
        "six",
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
            """Q: I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano. How many musical instruments do I have?
A:
thirteen
----
Q: I have a mouse, a goat, and two fish. How many animals do I have?
A:
four
----
Q: I have a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. How many fruits do I have?
A:
ten
----
Q: I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone. How many musical instruments do I have?
A:
eight
----
Q: I have three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin. How many musical instruments do I have?
A:
five
----
Q: I have five stoves, a fridge, a lamp, a microwave, a table, and an oven. How many objects do I have?
A:
ten
----
Q: I have a cabbage, a stalk of celery, a carrot, a lettuce head, a yam, three heads of broccoli, three potatoes, and two cauliflowers. How many vegetables do I have?
A:
thirteen
----
Q: I have a donkey, a cat, two bears, a pig, and a goat. How many animals do I have?
A:
six
----
Q: I have a frog, a duck, a bear, two donkeys, a fish, a goat, and a rabbit. How many animals do I have?
A:
eight
----
Q: I have a duck, a donkey, a cow, a bear, a dog, and a mouse. How many animals do I have?
A:
six
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


def human_decomp(model_name="text-davinci-002", temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Q: I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?
A: Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- blackberry (1)
- nectarine (1)
- plum (1)
- strawberry (1)
- banana (1)
- orange (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 1 + 1 = 6. So the answer is 6.
----
Q: I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?
A: Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- orange (1)
- raspberry (1)
- peaches (2)
- blackberry (1)
- apple (1)
- grape (1)
- nectarine (1)
- plums (3)
Now, let's add the numbers in parentheses: 1 + 1 + 2 + 1 + 1 + 1 + 1 + 3 = 11. So the answer is 11.
----
Q: I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?
A: Let's think step by step.
We first identify the vegetables on the list and include their quantity in parentheses:
- lettuce (1)
- broccoli (1)
- onion (1)
- celery (1)
- carrots (2)
- garlic (1)
- yam (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 2 + 1 + 1 = 8. So the answer is 8.
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
        numeric_labels = [str(w2n.word_to_num(l)) for l in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(numeric_labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

I have a trombone.
I have a flute.
I have an accordion.
I have three violins.
I have four clarinets.
I have a drum.
I have a trumpet.
I have a piano.
I have thirteen musical instruments in total.

The final answer is thirteen.
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a mouse, a goat, and two fish. How many animals do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

First, we need to identify what kind of item we're looking for. In this case, the question is asking for animals, so we're looking for anything that's classified as an animal.
We have a mouse, which is an animal, and a goat, which is also an animal. So far, we have two animals.
But wait! Fish are also animals. So we have two fish, which means we have a total of four animals.

The final answer is four.
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. How many fruits do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

First, we need to identify what kind of item we're looking for. In this case, the question is asking for fruits, so we're looking for anything that's classified as an fruit.
There is a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. All these items are fuits. There are a total of ten fruits.

The final answer is ten.
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are a lot of items, so it might be helpful to make a list.
The list of items is:
violin
two nectarines
flute
banana
accordion
two strawberries
clarinet
drum
trumpet
piano
trombone

The list of musical instruments is:
violin
flute
accordion
clarinet
drum
trumpet
piano
trombone

There are a total of eight musical instruments.

The final answer is eight.
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

First, we need to identify which items are musical instruments. We see that the trumpet, clarinet, piano, trombone, and violin are all musical instruments.
Next, we need to count how many musical instruments there are. We see that there are five musical instruments.
Finally, we need to write the answer in words. We see that there are five musical instruments, so we would say "five."

The final answer is five.
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a cabbage, a stalk of celery, a carrot, a lettuce head, a yam, three heads of broccoli, three potatoes, and two cauliflowers. How many vegetables do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are three potatoes, so we can say "three" or "3".
There are two cauliflowers, so we can say "two" or "2".
Adding the potatoes and cauliflowers together, we get "five" or "5".
Now let's think about the broccoli.
There are three heads of broccoli, so we can say "three" or "3".
Adding the potatoes, cauliflowers, and broccoli together, we get "eight" or "8".
Now let's add the rest of the vegetables.
There is one cabbage, so we can say "one" or "1".
There is one stalk of celery, so we can say "one" or "1".
There is one carrot, so we can say "one" or "1".
There is one lettuce head, so we can say "one" or "1".
There is one yam, so we can say "one" or "1".

Adding all of the vegetables together, we get "thirteen" or "13".

The final answer is thirteen.
----
"""
auto_cot_cleaned_prompt = """Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have a trombone, a flute, an accordion, three violins, four clarinets, a drum, a trumpet, and a piano.
I have ten musical instruments.

The final answer is ten
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a mouse, a goat, and two fish. How many animals do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

We can see that we have three animals- a mouse, a goat, and two fish. So, we have a total of three animals.

The final answer is three
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a strawberry, a grape, an apple, an orange, a plum, two nectarines, two bananas, and a peach. How many fruits do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are a total of nine items.
Of those nine items, there are seven fruits.
Therefore, there are seven fruits.

The final answer is seven
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are a lot of items, so it might be helpful to make a list.
I have a violin, two nectarines, a flute, a banana, an accordion, two strawberries, a clarinet, a drum, a trumpet, a piano, and a trombone.
The question is asking for a count of musical instruments, so we need to identify which items are musical instruments. A violin, flute, accordion, clarinet, drum, trumpet, piano, and trombone are all musical instruments.
There are eight musical instruments.

The final answer is eight
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin. How many musical instruments do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are three bananas, a trumpet, a clarinet, a piano, a nectarine, a trombone, a blackberry, and a violin.
We are looking for musical instruments, so we can ignore the bananas and the nectarine.
That leaves us with a trumpet, a clarinet, a piano, a trombone, a violin.
All of these except for the blackberry are musical instruments, so the answer is four.

The final answer is four
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have five stoves, a fridge, a lamp, a microwave, a table, and an oven. How many objects do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are five stoves, so the count is "five."
There is one fridge, so the count is "one."
There is one lamp, so the count is "one."
There is one microwave, so the count is "one."
There is one table, so the count is "one."
There is one oven, so the count is "one."
The total count is "five," "one," "one," "one," "one," and "one," which adds up to twelve.

The final answer is twelve
----
Object counting: You are given a bunch of items and asked to count a specific kind of item.
Q: I have a donkey, a cat, two bears, a pig, and a goat. How many animals do I have?
The final answer is the count of requested items in words, like "six" or "ten".
A: Let's think step-by-step.

There are a bunch of items.
You are asked to count a specific kind of item.
In this case, you are asked to count animals.
So, you need to look at each item and decide if it is an animal or not.
The donkey is an animal.
The cat is an animal.
The bears are animals.
The pig is an animal.
The goat is an animal.
That means there are five animals in total.

The final answer is five
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
            + """\nThe final answer is the count of requested items in words, like "six" or "ten".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is the count of requested items in words, like "six" or "ten".\nA: Let's think step-by-step.\n"""
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
    task_name = "Object counting"
    task_description = "(Object counting) You are given a bunch of items and asked to count a specific kind of item."

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
        prompts = [
            few_shot_cot_prompt
            % (
                description
                + """The final answer is the count of requested items in words, like "six" or "ten".""",
                x,
            )
            for x in chunk
        ]
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
        numeric_labels = [str(w2n.word_to_num(l)) for l in labels]
        new_labels = [[l1, l2] for l1, l2 in zip(labels, numeric_labels)]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        # preds = [[y.strip() for y in x.split("\n")] for x in answers]
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match_v2(new_labels, preds))
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
    task_name = "Object counting"
    task_description = "(Object counting) You are given a bunch of items and asked to count a specific kind of item."

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
        prompts = [
            few_shot_cot_prompt
            % (
                description
                + """The final answer is the count of requested items in words, like "six" or "ten".""",
                x,
            )
            for x in chunk
        ]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        numeric_labels = [str(w2n.word_to_num(l)) for l in labels]
        new_labels = [[l1, l2] for l1, l2 in zip(labels, numeric_labels)]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match_v2(new_labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


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
