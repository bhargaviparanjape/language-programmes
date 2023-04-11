import datasets
import numpy as np
from tqdm import tqdm
from utils import OpenAIModel, cache_dir, chunks

data = datasets.load_dataset("commonsense_qa", cache_dir=cache_dir)["validation"]
inputs = [
    d["question"]
    + " "
    + " ".join([k + ") " + v for k, v in zip(d["choices"]["label"], d["choices"]["text"])])
    for d in data
][:500]
labels = [d["answerKey"] for d in data][:500]
print(len(inputs))


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


def commonsenseqa():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """How can you get the attention of a person across the room? A) board ship B) shout at C) smile at D) cross street E) feel happy
B
----
Where would you find a metal rod in most people's preferred method of transportation? A) airplane B) construction site C) shops D) engine E) broken bone
D
----
What can you do in your house during the winter? A) ski B) skate C) play hockey D) blaze it E) knit
E
----
The drought was dangerous for the trees, they were more likely to what? A) wall in B) grow tall C) fire D) covered in snow E) burn
E
----
Where do you get a cheap room at a reception desk? A) lobby B) office park C) at hotel D) cellphone E) motel
E
----
What kind of food is not sweet? A) dry B) wet C) bitter D) nonsweet E) decaying
C
----
What is an easy way to make a potato soft? A) restaurants B) cooking pot C) beef stew D) steak house E) let it rot
B
----
Billy Bob sat on a bar stool and ordered a steak.  He ordered a steak because despite the style this place didn't serve alcohol.  What sort of place was this? A) dining hall B) drunker C) tavern D) kitchen E) restaurant
E
----
Where should I not put these boxes if my house is likely to flood? A) garage B) cellar C) basement D) kitchen E) attic
C
----
What do soldiers do when their opponents get the upper hand? A) follow orders B) use weapons C) call reinforcements D) coming home E) die in battle
E
----
%s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 2
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


commonsenseqa()
