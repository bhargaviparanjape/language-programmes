import argparse
from collections import Counter
import json
import pdb
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
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "tracking_shuffled_objects", cache_dir=cache_dir)
inputs = d["validation"]["inputs"][:250]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"][:250]
labels = [l[0] for l in labels]
print(len(inputs))

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = """(Shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object."""

io_pairs = [
    (
        """Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball.

As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Bob and Alice swap balls. Then, Eve and Dave swap balls. Then, Dave and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Claire has the""",
        "yellow ball.",
    ),
    (
        """Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball.

As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Alice and Claire swap balls. Then, Eve and Alice swap balls. Then, Claire and Bob swap balls. Finally, Claire and Dave swap balls. At the end of the game, Eve has the""",
        "red ball.",
    ),
    (
        """Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present.

As the event progresses, pairs of people swap gifts. First, Claire and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Bob and Dave swap their gifts. Then, Eve and Dave swap their gifts. Finally, Dave and Alice swap their gifts. At the end of the event, Eve has the""",
        "yellow present.",
    ),
    (
        """Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball.

As the game progresses, pairs of players trade balls. First, Alice and Claire swap balls. Then, Dave and Alice swap balls. Then, Bob and Eve swap balls. Then, Claire and Eve swap balls. Finally, Alice and Dave swap balls. At the end of the game, Claire has the""",
        "purple ball.",
    ),
    (
        """Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present.

As the event progresses, pairs of people swap gifts. First, Gertrude and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Alice and Dave swap their gifts. Then, Fred and Dave swap their gifts. Then, Bob and Dave swap their gifts. Then, Claire and Gertrude swap their gifts. Finally, Gertrude and Dave swap their gifts. At the end of the event, Alice has the""",
        "brown present.",
    ),
    (
        """Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Melissa, Bob is dancing with Ophelia, Claire is dancing with Patrick, Dave is dancing with Jamie, Eve is dancing with Lola, Fred is dancing with Helga, and Gertrude is dancing with Izzi.

Throughout the song, the dancers often trade partners. First, Alice and Fred switch partners. Then, Gertrude and Eve switch partners. Then, Gertrude and Alice switch partners. Then, Bob and Alice switch partners. Then, Dave and Fred switch partners. Then, Eve and Alice switch partners. Finally, Eve and Claire switch partners. At the end of the dance, Alice is dancing with""",
        "Izzi.",
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
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball.

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
        preds = [x.strip().split("\n") for x in answers]
        perf_array.append(token_match(labels, preds))
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
            """Q: Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a yellow ball, Bob has a blue ball, and Claire has a pink ball.
As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Alice and Bob swap balls. Finally, Claire and Bob swap balls. At the end of the game, Bob has the
A: Let's think step by step.
(0) At the start: Alice: yellow, Bob: blue, Claire: pink.
(1) Claire and Alice swap balls: Alice: pink, Bob: blue, Claire: yellow.
(2)  Alice and Bob swap balls: Alice: blue, Bob: pink, Claire: yellow.
(3) Claire and Bob swap balls: Alice: blue, Bob: yellow, Claire: pink.
At the end of the game, Bob has the yellow ball. So the answer is yellow.
----
Q: Alice, Bob, and Claire are playing a game. At the start of the game, they are each holding a ball: Alice has a white ball, Bob has a purple ball, and Claire has a pink ball.
As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Bob and Claire swap balls. Finally, Bob and Alice swap balls. At the end of the game, Alice has the
(A) white ball
(B) purple ball
(C) pink ball
A: Let's think step by step.
(0) At the start: Alice: white, Bob: purple, Claire: pink.
(1) Bob and Alice swap balls: Alice: purple, Bob: white, Claire: pink.
(2) Bob and Claire swap balls: Alice: purple, Bob: pink, Claire: white.
(3) Bob and Alice swap balls: Alice: pink, Bob: purple, Claire: white.
At the end of the game, Alice has the pink ball. So the answer is pink.
----
Q: Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Lola, Bob is dancing with Rodrigo, and Claire is dancing with Patrick.
Throughout the song, the dancers often trade partners. First, Alice and Bob switch partners. Then, Claire and Bob switch partners. Finally, Bob and Alice switch partners. At the end of the dance, Alice is dancing with
A: Let's think step by step.
(0) At the start: Alice: Lola, Bob: Rodrigo, Claire: Patrick.
(1) Alice and Bob switch partners: Alice: Rodrigo, Bob: Lola, Claire: Patrick.
(2) Claire and Bob switch partners: Alice: Rodrigo, Bob: Patrick, Claire: Lola.
(3) Bob and Alice switch partners: Alice: Patrick, Bob: Rodrigo, Claire: Lola.
At the end of the dance, Alice is dancing with Patrick. So the answer is Patrick.
----
%s
A: Let's think step by step."""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    new_labels = [l.replace(".", "") for l in labels]
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [
            get_autocot_answer(x, answer_prompt="So the answer is ").replace(".", "")
            for x in answers
        ]
        perf_array.append(substring_match(new_labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball.

As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Bob and Alice swap balls. Then, Eve and Dave swap balls. Then, Dave and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Claire has the
A: Let's think step-by-step.

Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball.
Claire and Alice swap balls. Now Claire has the brown ball and Alice has the black ball.
Bob and Alice swap balls. Now Bob has the black ball and Alice has the purple ball.
Eve and Dave swap balls. Now Eve has the green ball and Dave has the yellow ball.
Dave and Claire swap balls. Now Dave has the brown ball and Claire has the yellow ball.
Alice and Bob swap balls. Now Alice has the purple ball and Bob has the black ball.
Therefore, at the end of the game, Alice has the purple ball, Bob has the black ball, Claire has the yellow ball, Dave has the brown ball and Eve has the green ball.

The final answer is yellow ball.
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball.

As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Alice and Claire swap balls. Then, Eve and Alice swap balls. Then, Claire and Bob swap balls. Finally, Claire and Dave swap balls. At the end of the game, Eve has the
A: Let's think step-by-step.

Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball.
First, Bob and Alice swap balls. So now Alice has a brown ball and Bob has a black ball.
Then, Alice and Claire swap balls. So now Alice has a red ball and Claire has a brown ball.
Then, Eve and Alice swap balls. So now Eve has a red ball and Alice has a green ball.
Then, Claire and Bob swap balls. So now Claire has a brown ball and Bob has a black ball.
Finally, Claire and Dave swap balls. So now Claire has a white ball and Dave has a brown ball.
At the end of the game, Alice has the green ball, Bob has the black ball, Claire has the white ball, Dave has the brown ball and Eve has the red ball.

The final answer is red ball.
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present.

As the event progresses, pairs of people swap gifts. First, Claire and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Bob and Dave swap their gifts. Then, Eve and Dave swap their gifts. Finally, Dave and Alice swap their gifts. At the end of the event, Eve has the
A: Let's think step-by-step.

Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present.
First, Claire and Alice swap their gifts. This means that Claire now has a red present, and Alice has a green present.
Next, Dave and Eve swap their gifts. This means that Dave now has a white present, and Eve has a pink ball.
Then, Bob and Dave swap their gifts. This means that Bob now has a white ball, and Dave has a yellow present.
Then, Eve and Dave swap their gifts. This means that Eve now has a yellow present, and Dave has a pink present.
Finally, Dave and Alice swap their gifts. This means that Dave now has a green present, and Alice has a pink present.
Therefore, at the end of the event, Alice has the pink present, Bob has the white present, Claire has the red present, Dave has the green present and Eve has the yellow present.

The final answer is yellow present.
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball.

As the game progresses, pairs of players trade balls. First, Alice and Claire swap balls. Then, Dave and Alice swap balls. Then, Bob and Eve swap balls. Then, Claire and Eve swap balls. Finally, Alice and Dave swap balls. At the end of the game, Claire has the
A: Let's think step-by-step.

Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball.
First, Alice and Claire swap balls. This means that Alice now has the green ball, and Claire has the brown ball.
Then, Dave and Alice swap balls. This means that Dave now has the green ball, and Alice has the white ball.
Then, Bob and Eve swap balls. This means that Bob now has the blue ball, and Eve has the purple ball.
Then, Claire and Eve swap balls. This means that Claire now has the purple ball, and Eve has the brown ball.
Finally, Alice and Dave swap balls. This means that Alice now has the green ball, and Dave has the white ball.
So at the end of the game, Alice has the green ball, Bob has the blue ball, Claire has the purple ball, Dave has the white ball and Eve has the brown ball.

The final answer is purple ball.
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present.

As the event progresses, pairs of people swap gifts. First, Gertrude and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Alice and Dave swap their gifts. Then, Fred and Dave swap their gifts. Then, Bob and Dave swap their gifts. Then, Claire and Gertrude swap their gifts. Finally, Gertrude and Dave swap their gifts. At the end of the event, Alice has the
A: Let's think step-by-step.

Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present.
Gertrude and Alice swap their gifts. Gertrude now has a blue present, and Alice has a yellow present.
Dave and Eve swap their gifts. Dave now has a brown present, and Eve has a black ball.
Alice and Dave swap their gifts. Alice now has a brown ball, and Dave has a yellow present.
Fred and Dave swap their gifts. Fred now has a yellow ball, and Dave has an orange ball.
Bob and Dave swap their gifts. Bob now has an orange ball, and Dave has a white present.
Claire and Gertrude swap their gifts. Claire now has a blue present, and Gertrude has a purple present.
Gertrude and Dave swap their gifts. Gertrude now has a white present, and Dave has a purple present.
So, at the end of the event, Alice has the brown present, Bob has the orange present, Claire has the blue present, Dave has the purple present, Eve has the black present, Fred has the yellow present, and Gertrude has the white present.

The final answer is brown present.
----
"""

auto_cot_cleaned_prompt = """(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a black ball, Dave has a green ball, and Eve has a yellow ball.

As the game progresses, pairs of players trade balls. First, Claire and Alice swap balls. Then, Bob and Alice swap balls. Then, Eve and Dave swap balls. Then, Dave and Claire swap balls. Finally, Alice and Bob swap balls. At the end of the game, Claire has the
A: Let's think step-by-step.

First, Claire and Alice swap balls. So now Claire has a brown ball and Alice has a black ball.
Then, Bob and Alice swap balls. So now Bob has a brown ball and Alice has a purple ball.
Then, Eve and Dave swap balls. So now Eve has a green ball and Dave has a yellow ball.
Then, Dave and Claire swap balls. So now Dave has a black ball and Claire has a green ball.
Finally, Alice and Bob swap balls. So now Alice has a yellow ball and Bob has a purple ball.
Therefore, the final pairs are:
Claire has the green ball and Dave has the black ball.

The final answer is 'green ball'
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a black ball, Bob has a brown ball, Claire has a red ball, Dave has a white ball, and Eve has a green ball.

As the game progresses, pairs of players trade balls. First, Bob and Alice swap balls. Then, Alice and Claire swap balls. Then, Eve and Alice swap balls. Then, Claire and Bob swap balls. Finally, Claire and Dave swap balls. At the end of the game, Eve has the
A: Let's think step-by-step.

1. At the start of the game, each player has a ball:
Alice has a black ball,
Bob has a brown ball,
Claire has a red ball,
Dave has a white ball,
Eve has a green ball.
2. Bob and Alice swap balls.
Alice now has a brown ball,
Bob now has a black ball.
3. Alice and Claire swap balls.
Alice now has a red ball,
Claire now has a brown ball.
4. Eve and Alice swap balls.
Alice now has a green ball,
Eve now has a red ball.
5. Claire and Bob swap balls.
Claire now has a black ball,
Bob now has a green ball.
6. Claire and Dave swap balls.
Claire now has a white ball,
Dave now has a black ball.
7. At the end of the game, Eve has the green ball, Alice has the white ball, Claire has the brown ball, Bob has the red ball, and Dave has the black ball.

The final answer is 'green ball'
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a red present, Bob has a yellow present, Claire has a green present, Dave has a pink ball, and Eve has a white present.

As the event progresses, pairs of people swap gifts. First, Claire and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Bob and Dave swap their gifts. Then, Eve and Dave swap their gifts. Finally, Dave and Alice swap their gifts. At the end of the event, Eve has the
A: Let's think step-by-step.

At the start of the event:

Alice has a red present.
Bob has a yellow present.
Claire has a green present.
Dave has a pink ball.
Eve has a white present.

After the first swap (Claire and Alice):

Alice has a green present.
Bob has a yellow present.
Claire has a red present.
Dave has a pink ball.
Eve has a white present.

After the second swap (Dave and Eve):

Alice has a green present.
Bob has a yellow present.
Claire has a red present.
Dave has a white present.
Eve has a pink ball.

After the third swap (Bob and Dave):

Alice has a green present.
Bob has a pink ball.
Claire has a red present.
Dave has a yellow present.
Eve has a white present.

After the fourth swap (Eve and Dave):

Alice has a green present.
Bob has a pink ball.
Claire has a red present.
Dave has a white present.
Eve has a yellow present.

After the fifth and final swap (Dave and Alice):

Alice has a pink ball.
Bob has a green present.
Claire has a red present.
Dave has a yellow present.
Eve has a white present.

The final answer is 'white present'
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a brown ball, Bob has a purple ball, Claire has a green ball, Dave has a white ball, and Eve has a blue ball.

As the game progresses, pairs of players trade balls. First, Alice and Claire swap balls. Then, Dave and Alice swap balls. Then, Bob and Eve swap balls. Then, Claire and Eve swap balls. Finally, Alice and Dave swap balls. At the end of the game, Claire has the
A: Let's think step-by-step.

Alice starts with a brown ball.
Claire starts with a green ball.
Alice and Claire swap balls.
Now, Alice has a green ball and Claire has a brown ball.
Dave and Alice swap balls.
Now, Dave has a green ball and Alice has a white ball.
Bob and Eve swap balls.
Now, Bob has a blue ball and Eve has a purple ball.
Claire and Eve swap balls.
Now, Claire has a purple ball and Eve has a green ball.
Alice and Dave swap balls.
Now, Alice has a blue ball and Dave has a white ball.

The final answer is 'purple ball'
----
(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Alice, Bob, Claire, Dave, Eve, Fred, and Gertrude are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a white present, Claire has a purple present, Dave has a black ball, Eve has a brown present, Fred has a orange ball, and Gertrude has a yellow present.

As the event progresses, pairs of people swap gifts. First, Gertrude and Alice swap their gifts. Then, Dave and Eve swap their gifts. Then, Alice and Dave swap their gifts. Then, Fred and Dave swap their gifts. Then, Bob and Dave swap their gifts. Then, Claire and Gertrude swap their gifts. Finally, Gertrude and Dave swap their gifts. At the end of the event, Alice has the
A: Let's think step-by-step.

1. Gertrude and Alice swap their gifts.
2. Dave and Eve swap their gifts.
3. Alice and Dave swap their gifts.
4. Fred and Dave swap their gifts.
5. Bob and Dave swap their gifts.
6. Claire and Gertrude swap their gifts.
7. Gertrude and Dave swap their gifts.
So, at the end of the event, Alice has the yellow present, Bob has the purple present, Claire has the white present, Dave has the orange ball, Eve has the black ball, Fred has the brown present, and Gertrude has the blue present.

The final answer is 'yellow present'
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
    description = """(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object."""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=500, temperature=0.2, quote="---", n=1)
        prompt = """%s\n""" % description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

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
            + """%s\n""" % description
            + """%s\nA: Let's think step-by-step.\n""" % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        processed_labels = [l.replace(".", "") for l in labels]
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(processed_labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name, max_length=2048, temperature=0.4, quote="---", n=1)
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers = predict(
                "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                x,
            )
            pdb.set_trace()
            affordance_inputs = [
                json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers
            ]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [
                ex + a[: re.search("#2: ", a).span(0)[1]] + json.dumps(o)
                for ex, a, o in zip(x, answers, affordance_outputs)
            ]
            new_answers.extend(
                predict_with_affordance(
                    "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
                    x,
                )
            )
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used.
# Description:
# Input:
# ```
# if x < 5:
# 	pass
# ```
# What error does this program surface?
# Q1: [execute] Execute the following python code snippet.
# if x < 5:
# 	pass
# #1:
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'x' is not defined
# Q2: What is the final error message?
# #2: NameError: name 'x' is not defined
# Q3: [EOC]
# NameError: name 'x' is not defined
# ----
# Desciption:
# Input:
# ```
# x = set([1, 1, 2, 3])
# ```
# What is the value of x after this program executes?
# Q1: [code edit] Edit the following code to get the value of x
# x = set([1, 1, 2, 3])
# #1:
# x = set([1, 1, 2, 3])
# print(x)
# Q2: [execute] Execute the following python code snippet.
# x = set([1, 1, 2, 3])
# print(x)
# #2: {1, 2, 3}
# Q3: [EOC]
# {1, 2, 3}
# ----
# Description:
# Input:
# Python code:
# try:
#     n = int(input())
#     m = int(input())
#     integer_sum = int(n) + int(m)
#     print(integer_sum)
# except:
#     print('error')

#   choice: prints number between 5 and 6
#   choice: try input and except error
#   choice: inputs the string 'try'
#   choice: prints sum of two input numbers only if they are integers otherwise raises error
# Q1: [code generate] prints number between 5 and 6
# #1:
# import random

# print(random.uniform(5,6))
# Q2: [code generate] try input and except error
# #2
# try:
#     #open file
#     file = open(file_name, "r")
#     #read file
#     data = file.read()
#     #close file
#     file.close()
#     #split data
#     data = data.split("\n")
#     #remove empty string
# Q3: [code generate] inputs the string 'try'
# #3: print('try')
# Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
# #4:
# #!/usr/bin/python

# a=raw_input("enter first number: ")
# b=raw_input("enter second number: ")
# try:
#     sum=int(a)+int(b)
#     print "sum is: ",sum
# except:
#     print "enter integer values only"
# Q5: Which of the generated code snippets are most like the original one?
# #5: prints sum of two input numbers only if they are integers otherwise raises error
# Q6: [EOC]
# ----
# Desciption: %s
# Input: %s
# Q1: """

few_shot_cot_prompt = few_shot_arithmetic_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Tracking shuffled objects"
    task_description = """(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object."""

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
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.4, quote="---", n=1)
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
        processed_labels = [
            label.replace(" ball.", "").replace(" present.", "").replace(".", "")
            for label in labels
        ]
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict_complete(task_description, x))
            time.sleep(5)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(processed_labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Tracking shuffled objects"
    task_description = """(Tracking shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object."""

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
        # new_labels = ["Ans: " + label for label in labels]
        processed_labels = [
            label.replace(" ball.", "").replace(" present.", "").replace(".", "")
            for label in labels
        ]
        for x in tqdm(chunks(inputs, 10)):
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            pdb.set_trace()
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(processed_labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Description: (Shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Input: Alice, Bob, Claire, Dave, and Eve are playing a game. At the start of the game, they are each holding a ball: Alice has a pink ball, Bob has a white ball, Claire has a red ball, Dave has a purple ball, and Eve has a yellow ball.

As the game progresses, pairs of players trade balls. First, Alice and Dave swap balls. Then, Claire and Eve swap balls. Then, Alice and Bob swap balls. Then, Dave and Claire swap balls. Finally, Alice and Claire swap balls. At the end of the game, Claire has the
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
players = ['Alice', 'Bob', 'Claire', 'Dave', 'Eve']
balls = ['pink', 'white', 'red', 'purple', 'yellow']

swap_list = [[0, 3], [2, 4], [0, 1], [3, 2], [0, 2]]

for swap in swap_list:
    balls[swap[0]], balls[swap[1]] = balls[swap[1]], balls[swap[0]]

ans = [players[i] + ' has a ' + balls[i] for i in range(5)]
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2:
['Alice has a pink', 'Bob has a purple', 'Claire has a white', 'Dave has a yellow', 'Eve has a red']
Q3: [subquestion] At the end, what color does Claire have?
#3: white
Q3: [EOQ]
Ans: white
----
Description: (Shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Input: Alice, Bob, Claire, Dave, and Eve are friends and avid readers who occasionally trade books. At the start of the semester, they each buy one new book: Alice gets Catch-22, Bob gets Hound of the Baskervilles, Claire gets Frankenstein, Dave gets The Pearl, and Eve gets The Fellowship of the Ring.

As the semester proceeds, they start trading around the new books. First, Eve and Alice swap books. Then, Alice and Claire swap books. Then, Alice and Bob swap books. Then, Dave and Alice swap books. Finally, Dave and Claire swap books. At the end of the semester, Alice has
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
persons = ["Alice", "Bob", "Claire", "Dave", "Eve"]
books = ["Catch-22", "Hound of the Baskervilles", "Frankenstein", "The Pearl", "The Fellowship of the Ring"]

swap_list = [(4,0), (0,2), (0,1), (3,0), (3,2)]

for swap in swap_list:
    books[swap[0]], books[swap[1]] = books[swap[1]], books[swap[0]]

ans = [persons[i] + ' has a ' + books[i] for i in range(5)]
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: ['Alice has a The Pearl', 'Bob has a Frankenstein', 'Claire has a Hound of the Baskervilles', 'Dave has a The Fellowship of the Ring', 'Eve has a Catch-22']
Q3: [subquestion] At the end, what book does Alice have?
#3: The Pearl
Q3: [EOQ]
Ans: The Pearl
----
Description: (Shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Input: Alice, Bob, Claire, Dave, and Eve are holding a white elephant gift exchange. At the start of the event, they are each holding a present of a different color: Alice has a blue present, Bob has a green present, Claire has a pink ball, Dave has a yellow present, and Eve has a black ball.

As the event progresses, pairs of people swap gifts. First, Bob and Dave swap their gifts. Then, Bob and Alice swap their gifts. Then, Eve and Dave swap their gifts. Then, Bob and Eve swap their gifts. Finally, Eve and Claire swap their gifts. At the end of the event, Bob has the
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
persons = ["Alice", "Bob", "Claire", "Dave", "Eve"]
color = ["blue", "green", "pink", "yellow", "black"]

swap_list = [(1,3), (1,0), (4,3), (1,4), (4,2)]

for swap in swap_list:
    color[swap[0]], color[swap[1]] = color[swap[1]], color[swap[0]]

ans = [persons[i] + ' has a ' + color[i] + ' present' for i in range(5)]
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: ['Alice has a yellow present', 'Bob has a green present', 'Claire has a blue present', 'Dave has a black present', 'Eve has a pink present']
Q3: [subquestion] At the end, what gift does Bob have?
#3: green present
Q3: [EOQ]
Ans: green present
----
Description: (Shuffled objects) You are given a list of persons or objects that sequentially interact with each other in pairs. Track how the pairs change over the course of the interaction and figure out the pair of the final person or object.
Input: Alice, Bob, Claire, Dave, and Eve are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Jamie, Bob is dancing with Lola, Claire is dancing with Izzi, Dave is dancing with Rodrigo, and Eve is dancing with Ophelia.

Throughout the song, the dancers often trade partners. First, Bob and Eve switch partners. Then, Alice and Bob switch partners. Then, Dave and Alice switch partners. Then, Dave and Claire switch partners. Finally, Bob and Claire switch partners. At the end of the dance, Dave is dancing with
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
persons = ["Alice", "Bob", "Claire", "Dave", "Eve"]
partner = ["Jamie", "Lola", "Izzi", "Rodrigo", "Ophelia"]

swap_list = [(1,4), (0,1), (3,0), (3,2), (1,2)]

for swap in swap_list:
    partner[swap[0]], partner[swap[1]] = partner[swap[1]], partner[swap[0]]

ans = [persons[i] + ' is with ' + partner[i] for i in range(5)]
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: ['Alice is with Rodrigo', 'Bob is with Ophelia', 'Claire is with Jamie', 'Dave is with Izzi', 'Eve is with Lola']
Q3: [subquestion] At the end, who is Dave dancing with?
#3: Izzi
Q3: [EOQ]
Ans: Izzi
----
Description: %s
Input: %s
Q1:"""


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
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            processed_labels = [
                label.replace(" ball.", "").replace(" present.", "").replace(".", "")
                for label in labels
            ]
            for x in tqdm(chunks(inputs, 10)):
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(processed_labels, preds))
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
