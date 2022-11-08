from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir

import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re

d = datasets.load_dataset('bigbench', 'symbol_interpretation', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]
# print(len(inputs))

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

def symbol_interpretation():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
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
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


symbol_interpretation()