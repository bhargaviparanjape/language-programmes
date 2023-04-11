import argparse
from collections import Counter
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
    substring_match_v2,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "ruin_names", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'rain man'?\n  choice: rainmman\n  choice: ruin man\n  choice: rain men\n  choice: rains man",
        "ruin man",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'the dark knight rises'?\n  choice: the dark kniggt rises\n  choice: thetdark knight rises\n  choice: the bark knight rises\n  choice: the dork knight rises",
        "the dork knight rises",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'the shawshank redemption'?\n  choice: the shawshanknredemption\n  choice: the shcawshank redemption\n  choice: the shapwshank redemption\n  choice: the shawshark redemption",
        "the shawshark redemption",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'the third man'?\n  choice: the third pan\n  choice: thed third man\n  choice: the third men\n  choice: the trird man",
        "the third pan",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'coldplay'?\n  choice: coldpnay\n  choice: soldplay\n  choice: coldglay\n  choice: colldplay",
        "soldplay",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'paul revere and the raiders'?\n  choice: paul erevere and the raiders\n  choice: paul severe and the raiders\n  choice: mpaul revere and the raiders\n  choice: paul rfevere and the raiders",
        "paul severe and the raiders",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'the smashing pumpkins'?\n  choice: the smashing bumpkins\n  choice: thez smashing pumpkins\n  choice: the rmashing pumpkins\n  choice: the smashingq pumpkins",
        "the smashing bumpkins",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'guns n' roses'?\n  choice: guns n' hoses\n  choice: guns n'z roses\n  choice: gunh n' roses\n  choice: guns n' ropes",
        "guns n' hoses",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'the beatles'?\n  choice: the bdeatles\n  choice: thp beatles\n  choice: tce beatles\n  choice: the beatless",
        "the beatless",
    ),
    (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'hellboy'?\n  choice: hellbby\n  choice: hpllboy\n  choice: cellboy\n  choice: hellbfoy",
        "cellboy",
    ),
]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Ruin Names"
task_description = """(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny. Go over the choices one by one, figuring out if they are funny."""


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

auto_cot_cleaned_prompt = """(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'rain man'?
  choice: rainmman
  choice: ruin man
  choice: rain men
  choice: rains man
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to decide what makes a humorous edit. In this case, we might want to look for something that changes the meaning of the original word or phrase, or that makes it sound silly.
Next, we need to look at the choices and see which one meets our criteria.
Choice A: rainmman
This doesn't change the meaning of the original word, so it's not particularly funny.
Choice B: ruin man
This changes the meaning of the original word to something completely different, so it could be funny.
Choice C: rain men
This doesn't change the meaning of the original word, so it's not particularly funny.
Choice D: rains man
This changes the meaning of the original word to something completely different, so it could be funny.
Based on this, we would say that choice B (ruin man) is the most likely to be the humorous edit of the original artist or movie name (rain man).

The final answer is ruin man
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'the dark knight rises'?
  choice: the dark kniggt rises
  choice: thetdark knight rises
  choice: the bark knight rises
  choice: the dork knight rises
The final answer is one of the choices.
A: Let's think step-by-step.

The first word is "the." We could change this to "teh" for a lulz, but that's not very funny.
The second word is "dark." We could change this to "dork" for a lulz, but that's not very funny.
The third word is "knight." We could change this to "kniggt" for a lulz, but that's not very funny.
The fourth word is "rises." We could change this to "rises" for a lulz, but that's not very funny.
So, the final answer is "the dork knight rises."

The final answer is the dork knight rises
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'the shawshank redemption'?
  choice: the shawshanknredemption
  choice: the shcawshank redemption
  choice: the shapwshank redemption
  choice: the shawshark redemption
The final answer is one of the choices.
A: Let's think step-by-step.

The first step is to identify which edit makes the name funniest. In this case, it's probably the edit that changes "shawshank" to "shapwshank."
The second step is to determine whether this edit is simple enough to be considered a "simple edit." One way to do this is to calculate the Levenshtein distance between the original name and the edited name.
The Levenshtein distance between "shawshank" and "shapwshank" is 3, so this edit is simple enough to be considered a "simple edit."
Therefore, the answer is "the shapwshank redemption."

The final answer is the shapwshank redemption
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'the third man'?
  choice: the third pan
  choice: thed third man
  choice: the third men
  choice: the trird man
The final answer is one of the choices.
A: Let's think step-by-step.

The first step is to calculate the Levenshtein distance between the original name and the edit. In this case, the Levenshtein distance between "the third man" and "the third pan" is 3. The Levenshtein distance between "the third man" and "thed third man" is 4. The Levenshtein distance between "the third man" and "the third men" is 3. The Levenshtein distance between "the third man" and "the trird man" is 5.
The second step is to compare the Levenshtein distances to see which is the smallest. In this case, the smallest Levenshtein distance is 3, so the humorous edit is "the third pan".

The final answer is the third pan
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'coldplay'?
  choice: coldpnay
  choice: soldplay
  choice: coldglay
  choice: colldplay
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to decide what makes a name humorous. In this case, we might want to look for a name that sounds like it could be a real band or movie, but is actually a made-up word.
Next, we need to find an edit that would create such a word. In this case, we could add an extra letter, like "d" or "g", to create a made-up word.
Finally, we need to choose the best option from the choices given. In this case, we might choose "coldglay" or "colldplay" as the most humorous options.

The final answer is coldglay
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'paul revere and the raiders'?
  choice: paul erevere and the raiders
  choice: paul severe and the raiders
  choice: mpaul revere and the raiders
  choice: paul rfevere and the raiders
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to decide what makes a name humorous. In this case, we might want to look for a name that sounds like it could be real, but is actually quite absurd.
With that in mind, we can eliminate choice C, which changes the name to 'Mpaul Revere and the Raiders'. This change doesn't make the name any more absurd, and in fact, might make it less so.
Next, we can eliminate choice B, 'Paul Severe and the Raiders'. While this name is absurd, it doesn't sound like a real band name in the way that the other choices do.
That leaves us with choices A, 'Paul Erevere and the Raiders', and D, 'Paul Rfevere and the Raiders'. Both of these choices sound like real band names, but choice D is funnier because it is a more extreme change to the original name.

The final answer is paul rfevere and the raiders
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'the smashing pumpkins'?
  choice: the smashing bumpkins
  choice: thez smashing pumpkins
  choice: the rmashing pumpkins
  choice: the smashingq pumpkins
The final answer is one of the choices.
A: Let's think step-by-step.

The first choice is "the smashing bumpkins". This is not a very good edit, because it doesn't change the meaning of the name very much.
The second choice is "thez smashing pumpkins". This is a better edit, because it changes the meaning of the name (from a band to a movie) and it also changes the pronunciation of the name.
The third choice is "the rmashing pumpkins". This is a good edit, because it changes the meaning of the name (from a band to a food) and it also changes the pronunciation of the name.
The fourth choice is "the smashingq pumpkins". This is a good edit, because it changes the meaning of the name (from a band to a letter) and it also changes the pronunciation of the name.

The final answer is the rmashing pumpkins.
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'guns n' roses'?
  choice: guns n' hoses
  choice: guns n'z roses
  choice: gunh n' roses
  choice: guns n' ropes
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to decide what kind of humorous edit we want to make. One option is to change the spelling of the name to something similar to how it might be pronounced. For example, we could change "guns n' roses" to "guns n'z roses."
Another option is to change the meaning of the name by adding or removing a word. For example, we could change "guns n' roses" to "guns n' hoses" to make a pun on the fact that hoses are often used to put out fires.
Finally, we could change the name to something completely different but still related to the original. For example, we could change "guns n' roses" to "gunh n' roses" to make a play on the word "gun."
All of these options are humorous, so it just depends on which one you think is funniest.

The final answer is guns n' hoses
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'the beatles'?
  choice: the bdeatles
  choice: thp beatles
  choice: tce beatles
  choice: the beatless
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to decide what kind of edit would be funny. One option might be to change the spelling of the band's name so that it sounds like a different word. For example, we could change "the beatles" to "the bdeatles" to make it sound like "the beetles."
Another option might be to change the order of the letters in the band's name. For example, we could change "the beatles" to "thp beatles" to make it sound like "the petals."
A third option might be to add or remove a letter from the band's name. For example, we could change "the beatles" to "tce beatles" to make it sound like "the cattle."
Finally, we could change one letter in the band's name to a different letter. For example, we could change "the beatles" to "the beatless" to make it sound like "the beastless."
Based on these options, we think the funniest edit would be to change "the beatles" to "the bdeatles."

The final answer is the bdeatles
----
(Ruin Names) Pick out which simple (measured by Levenshtein distance) edit to the name of a band or movie makes it funny.
Q: Which of the following is a humorous edit of this artist or movie name: 'hellboy'?
  choice: hellbby
  choice: hpllboy
  choice: cellboy
  choice: hellbfoy
The final answer is one of the choices.
A: Let's think step-by-step.

1) "hellboy" -> "hellbby"
This change is not humorous.
2) "hellboy" -> "hpllboy"
This change is not humorous.
3) "hellboy" -> "cellboy"
This change is not humorous.
4) "hellboy" -> "hellbfoy"
This change is humorous.

The final answer is hellbfoy
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
            + """\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
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
    f.write("Formal Fallacies\n\n")
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
            + """%s\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
            % (x)
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
            + """%s\nThe final answer is one of the choices.\nA: Let's think step-by-step.\n"""
            % (x)
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
        label_dict = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)"}
        choices = [
            [ans.strip() for ans in inp.replace("\nA:", "").split("choice: ")][1:] for inp in inputs
        ]
        new_labels = [label_dict[choice.index(label)] for label, choice in zip(labels, choices)]
        joint_labels = [[label, new_label] for label, new_label in zip(labels, new_labels)]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("Q:", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match_v2(joint_labels, preds))
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
