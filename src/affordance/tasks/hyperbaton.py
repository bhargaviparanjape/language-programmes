import argparse
from collections import Counter
import pdb
import re
import time

import datasets
import numpy as np
from prompt_library import (
    few_shot_free_prompt,
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

d = datasets.load_dataset("bigbench", "hyperbaton", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [
    (
        'Q: Which sentence has the correct adjective order: a " gold old-fashioned exercise little awful blue monkey " b " awful little old-fashioned blue gold exercise monkey " ?',
        "b",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " obnoxious lead old massive circular white sock " b " obnoxious massive old circular white lead sock " ?',
        "b",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " normal-size archaic pyramidal Brazilian drinking motorcycle " b " pyramidal drinking archaic normal-size Brazilian motorcycle " ?',
        "a",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " huge brown iron eating sweater " b " brown iron huge eating sweater " ?',
        "a",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " spherical wonderful archaic sock " b " wonderful archaic spherical sock " ?',
        "b",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " brand-new driving leather car " b " brand-new leather driving car " ?',
        "b",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " medium-size ancient rectangular red typing computer " b " medium-size ancient typing rectangular red computer " ?',
        "a",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " nice prismlike tan American eating shoe " b " eating American prismlike tan nice shoe " ?',
        "a",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " archaic Iranian wool match " b " Iranian wool archaic match " ?',
        "a",
    ),
    (
        'Q: Which sentence has the correct adjective order: a " prismlike archaic sock " b " archaic prismlike sock " ?',
        "b",
    ),
]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Hyperbation"
task_description = """(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker."""


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
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
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """"""

auto_cot_cleaned_prompt = """(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " gold old-fashioned exercise little awful blue monkey " b " awful little old-fashioned blue gold exercise monkey " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "gold". This is a color, so it goes before the other adjectives.
The next adjective is "old-fashioned". This is a description of the noun, so it goes after the color adjective.
The next adjective is "little". This is a size adjective, so it goes after the description adjective.
The next adjective is "awful". This is a negative adjective, so it goes after the size adjective.
The next adjective is "blue". This is another color adjective, so it goes after the negative adjective.
The final adjective is "exercise". This is another description adjective, so it goes after the second color adjective.


----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " obnoxious lead old massive circular white sock " b " obnoxious massive old circular white lead sock " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "obnoxious".
The second adjective is either "massive" or "old".
If the second adjective is "massive", then the third adjective is "old".
If the second adjective is "old", then the third adjective is "massive".
The fourth adjective is "circular".
The fifth adjective is "white".
The sixth and final adjective is "sock".
----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " normal-size archaic pyramidal Brazilian drinking motorcycle " b " pyramidal drinking archaic normal-size Brazilian motorcycle " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "normal-size". This is an adjective that describes the noun "motorcycle".
The next adjective is "archaic". This is an adjective that describes the noun "pyramidal".
The next adjective is "pyramidal". This is an adjective that describes the noun "drinking".
The next adjective is "drinking". This is an adjective that describes the noun "motorcycle".
The last adjective is "Brazilian". This is an adjective that describes the noun "motorcycle".
Therefore, the correct adjective order is "normal-size archaic pyramidal drinking Brazilian motorcycle".

----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " huge brown iron eating sweater " b " brown iron huge eating sweater " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "huge." This is correct.
The second adjective is "brown." This is also correct.
The third adjective is "iron." This is incorrect. The correct word would be "metal."
The fourth adjective is "eating." This is incorrect. The correct word would be "hungry."
The fifth and final adjective is "sweater." This is correct.
Therefore, the correct sentence is: "A huge brown metal eating sweater."
----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " spherical wonderful archaic sock " b " wonderful archaic spherical sock " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

We have the word "sock". This is our noun.
Next, we have the adjectives "spherical", "wonderful", and "archaic".
The order of adjectives in English is:
Opinion (e.g. wonderful)
Size (e.g. spherical)
Age (e.g. archaic)
Shape (e.g. spherical)
Color (e.g. red)
Origin (e.g. French)
Material (e.g. woolen)
So, the correct sentence would be "a".

----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " brand-new driving leather car " b " brand-new leather driving car " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

"Brand-new" is an adjective that would describe the car.
"Driving" is a gerund, or a verb that acts as a noun, that would describe what you do with the car.
"Leather" is an adjective that would describe the car.
----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " medium-size ancient rectangular red typing computer " b " medium-size ancient typing rectangular red computer " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "medium-size".
The second adjective is "ancient".
The third adjective is "rectangular".
The fourth adjective is "red".
The final word is "computer".

Therefore, the correct sentence is "a medium-size ancient rectangular red computer".
----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " nice prismlike tan American eating shoe " b " eating American prismlike tan nice shoe " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "nice." This is a good word to describe someone's character, so it goes before words that describe physical appearance.
The next word is "prismlike." This word is not as common, so it might go in the middle of the adjectives.
The next word is "tan." This word describes someone's physical appearance, so it goes after "nice."
The next word is "American." This word describes someone's nationality, so it goes after "tan."
The final word is "eating." This word describes what the person is doing, so it goes at the end.
This leaves us with the following sentence: "A nice, tan, American eating shoe."

----
(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " archaic Iranian wool match " b " Iranian wool archaic match " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "archaic". This is describing the noun "match".
The second adjective is "Iranian". This is also describing the noun "match".
The third adjective is "wool". This is describing the noun "match".
Therefore, the correct adjective order would be "archaic Iranian wool match".
----

(Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Q: Which sentence has the correct adjective order: a " prismlike archaic sock " b " archaic prismlike sock " ?
The final answer is either "a" or "b".
A: Let's think step-by-step.

The first adjective is "prismlike." This is a descriptive adjective, which means it describes the noun that follows it.
The next adjective is "archaic." This is a limiting adjective, which means it limits the scope of the noun that follows it.
The final adjective is "sock." This is the noun that the adjectives modify.
Based on this analysis, the correct adjective order is "prismlike archaic sock."
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
    for io_pair in io_pairs[:10]:
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer is either "a" or "b".\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Hyperbation\n\n")
    f.write(auto_cot_prompt)

    pdb.set_trace()

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [
            auto_cot_prompt
            + """%s\n""" % task_description
            + """%s\nThe final answer is either "a" or "b".\nA: Let's think step-by-step.\n""" % (x)
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
            + """%s\nThe final answer is either "a" or "b".\nA: Let's think step-by-step.\n""" % (x)
            for x in chunk
        ]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1] :])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The final answer is """, x).span(0)[-1] :]]
                            )
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0] for x in answers]
            perf_array.append(substring_match(labels, preds))
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
                answers.extend(predict(x))
                time.sleep(1)
            preds = [get_autocot_answer(x, answer_prompt="The final answer is ") for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = few_shot_free_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt

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

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

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
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


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
