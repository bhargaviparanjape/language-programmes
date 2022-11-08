from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'tracking_shuffled_objects', cache_dir=cache_dir)
inputs =  d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels =  d['validation']['targets']
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [("""Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball. 

As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Bob and Alice swap balls. Then, Eve and Dave swap balls. Then, Dave and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Claire has the""",
"yellow ball."),
("""Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball. 

As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Alice and Claire swap balls. Then, Eve and Alice swap balls. Then, Claire and Bob swap balls. Finally, Claire and Dave swap balls. At the end of the game, Eve has the""",
"red ball."),
("""Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present. 

As the event progresses, pairs of people swap gifts. First, Claire and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Bob and Dave swap their gifts. Then, Eve and Dave swap their gifts. Finally, Dave and Alice swap their gifts. At the end of the event, Eve has the""",
"yellow present."),
("""Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball. 

As the game progresses, pairs of players trade balls. First, Alice and Claire swap balls. Then, Dave and Alice swap balls. Then, Bob and Eve swap balls. Then, Claire and Eve swap balls. Finally, Alice and Dave swap balls. At the end of the game, Claire has the""",
"purple ball."),
("""Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present. 

As the event progresses, pairs of people swap gifts. First, Gertrude and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Alice and Dave swap their gifts. Then, Fred and Dave swap their gifts. Then, Bob and Dave swap their gifts. Then, Claire and Gertrude swap their gifts. Finally, Gertrude and Dave swap their gifts. At the end of the event, Alice has the""",
"brown present."),
("""Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Melissa, Bob is dancing with Ophelia, Claire is dancing with Patrick, Dave is dancing with Jamie, Eve is dancing with Lola, Fred is dancing with Helga, and Gertrude is dancing with Izzi. 

Throughout the song, the dancers often trade partners. First, Alice and Fred switch partners. Then, Gertrude and Eve switch partners. Then, Gertrude and Alice switch partners. Then, Bob and Alice switch partners. Then, Dave and Fred switch partners. Then, Eve and Alice switch partners. Finally, Eve and Claire switch partners. At the end of the dance, Alice is dancing with""",
"Izzi.")]

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

def tracking_shuffled_objects():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball. 

As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Bob and Alice swap balls. Then, Eve and Dave swap balls. Then, Dave and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Claire has the
yellow ball.
----
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball. 

As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Alice and Claire swap balls. Then, Eve and Alice swap balls. Then, Claire and Bob swap balls. Finally, Claire and Dave swap balls. At the end of the game, Eve has the
red ball.
----
Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present. 

As the event progresses, pairs of people swap gifts. First, Claire and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Bob and Dave swap their gifts. Then, Eve and Dave swap their gifts. Finally, Dave and Alice swap their gifts. At the end of the event, Eve has the
yellow present.
----
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball. 

As the game progresses, pairs of players trade balls. First, Alice and Claire swap balls. Then, Dave and Alice swap balls. Then, Bob and Eve swap balls. Then, Claire and Eve swap balls. Finally, Alice and Dave swap balls. At the end of the game, Claire has the
purple ball.
----
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present. 

As the event progresses, pairs of people swap gifts. First, Gertrude and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Alice and Dave swap their gifts. Then, Fred and Dave swap their gifts. Then, Bob and Dave swap their gifts. Then, Claire and Gertrude swap their gifts. Finally, Gertrude and Dave swap their gifts. At the end of the event, Alice has the
brown present.
----
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Melissa, Bob is dancing with Ophelia, Claire is dancing with Patrick, Dave is dancing with Jamie, Eve is dancing with Lola, Fred is dancing with Helga, and Gertrude is dancing with Izzi. 

Throughout the song, the dancers often trade partners. First, Alice and Fred switch partners. Then, Gertrude and Eve switch partners. Then, Gertrude and Alice switch partners. Then, Bob and Alice switch partners. Then, Dave and Fred switch partners. Then, Eve and Alice switch partners. Finally, Eve and Claire switch partners. At the end of the dance, Alice is dancing with
Izzi.
----
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Izzi, Bob is dancing with Helga, Claire is dancing with Melissa, Dave is dancing with Sam, Eve is dancing with Karl, Fred is dancing with Ophelia, and Gertrude is dancing with Lola. 

Throughout the song, the dancers often trade partners. First, Fred and Eve switch partners. Then, Bob and Dave switch partners. Then, Eve and Alice switch partners. Then, Bob and Claire switch partners. Then, Eve and Alice switch partners. Then, Gertrude and Fred switch partners. Finally, Bob and Dave switch partners. At the end of the dance, Bob is dancing with
Helga.
----
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book: Alice gets Ulysses, Bob gets The Odyssey, Claire gets Moby Dick, Dave gets The Great Gatsby, Eve gets Lolita, Fred gets Frankenstein, and Gertrude gets The Fellowship of the Ring. 

As the semester proceeds, they start trading around the new books. First, Eve and Bob swap books. Then, Eve and Claire swap books. Then, Gertrude and Alice swap books. Then, Gertrude and Claire swap books. Then, Alice and Bob swap books. Then, Bob and Dave swap books. Finally, Fred and Claire swap books. At the end of the semester, Fred has
Ulysses.
----
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a yellow ball, Dave has a purple ball, Eve has a blue ball, Fred has a orange ball, and Gertrude has a pink ball. 

As the game progresses, pairs of players trade balls. First, Bob and Claire swap balls. Then, Dave and Gertrude swap balls. Then, Fred and Gertrude swap balls. Then, Bob and Fred swap balls. Then, Gertrude and Eve swap balls. Then, Claire and Eve swap balls. Finally, Claire and Alice swap balls. At the end of the game, Fred has the
yellow ball.
----
Alice, Bob, and Claire are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book: Alice gets Frankenstein, Bob gets Catch-22, and Claire gets Ulysses. 

As the semester proceeds, they start trading around the new books. First, Bob and Alice swap books. Then, Alice and Claire swap books. Finally, Claire and Bob swap books. At the end of the semester, Claire has
Frankenstein.
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
        preds = [x.strip().split("\n") for x in answers]
        perf_array.append(token_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

tracking_shuffled_objects()

few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used. 
Description: 
Input: 
```
if x < 5:
	pass
```
What error does this program surface?
Q1: [execute] Execute the following python code snippet.
if x < 5:
	pass
#1:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Q2: What is the final error message?
#2: NameError: name 'x' is not defined
Q3: [EOC]
NameError: name 'x' is not defined
----
Desciption:
Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to get the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOC]
{1, 2, 3}
----
Description:
Input:
Python code:
try:
    n = int(input())
    m = int(input())
    integer_sum = int(n) + int(m)
    print(integer_sum)
except:
    print('error')

  choice: prints number between 5 and 6
  choice: try input and except error
  choice: inputs the string 'try'
  choice: prints sum of two input numbers only if they are integers otherwise raises error
Q1: [code generate] prints number between 5 and 6
#1:
import random

print(random.uniform(5,6))
Q2: [code generate] try input and except error
#2
try:
    #open file
    file = open(file_name, "r")
    #read file
    data = file.read()
    #close file
    file.close()
    #split data
    data = data.split("\n")
    #remove empty string
Q3: [code generate] inputs the string 'try'
#3: print('try')
Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
#4:
#!/usr/bin/python

a=raw_input("enter first number: ")
b=raw_input("enter second number: ")
try:
    sum=int(a)+int(b)
    print "sum is: ",sum
except:
    print "enter integer values only"
Q5: Which of the generated code snippets are most like the original one?
#5: prints sum of two input numbers only if they are integers otherwise raises error
Q6: [EOC]
----
Desciption: %s
Input: %s
Q1: """


def auto_cot():
    auto_cot_prompt = ""
    description = "TASK: Take the letters at position 3 of the words in a list of words and concatenate them using a space."
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
    print(auto_cot_prompt)
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

auto_cot()


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character  = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers = predict("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x)
            pdb.set_trace()
            affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x))
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

affordance()