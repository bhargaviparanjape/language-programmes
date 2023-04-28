import argparse
from collections import Counter
import json
import re

import datasets
import numpy as np
from prompt_library import llm_similar_tasks, random_tasks, similar_auto_breakdowns, similar_tasks
from sequential_interpreter import TopDownVisitor
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
import evaluate

rouge = evaluate.load("rouge")

# TODO Only using 10 examples for now
train_dataset = datasets.load_dataset("ccdv/arxiv-summarization", split="train[:10]")
train_inputs = train_dataset["article"]
train_inputs = [t.split('\n') for t in train_inputs] # TODO: concatenate 5 paragraphs
train_labels = train_dataset["abstract"]
dev_dataset = datasets.load_dataset("ccdv/arxiv-summarization", split="validation[:10]")
dev_inputs = dev_dataset["article"]
dev_inputs = [t.split('\n') for t in dev_inputs] # TODO: concatenate 5 paragraphs
dev_labels = dev_dataset["abstract"]

io_pairs = [
    ("""A: <article 1 from arxiv>""", "S: <summary 1 from arxiv>"),
    ("""A: <article 2 from arxiv>""", "S: <summary 2 from arxiv>"),
    ("""A: <article 3 from arxiv>""", "S: <summary 3 from arxiv>"),
]

task_description = "Summarize the given articles"

few_shot_cot_prompt = """In these examples, you are given an Article. Break the input down into subtasks in order to solve the task. You can read and write to an external file memory to help summarize using functions from alexAndKylesAwesomeLibrary.
from alexAndKylesAwesomeLibrary import read
from alexAndKylesAwesomeLibrary import write
from alexAndKylesAwesomeLibrary import erase

Article: <Article 1>
Summary:  <Summary 1>
S1: [read] this is one of the most important parts of the article [write].
"""
def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)["input_ids"]))

    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=200, temperature=temperature, quote="---", n=1
        )
        prompts = ["%s\n%s".format(few_shot_prompt, x) for x in chunk]
        return gpt3(prompts)

    predictions_list = []
    labels_list = []
    runs = 5
    for run in range(runs):
        print(f"run {run+1}/{runs}")
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        predictions_list.append(preds)
        labels_list.append(labels)
    print("Rouge performance")
    print(rouge.compute(predictions=predictions_list, references=labels_list))

def get_few_shot_cot_prompt(task_name: str, description: str, strategy: str) -> str:
    if strategy == "fixed":
        return few_shot_cot_prompt
    elif strategy == "random":
        return random_tasks(N=6)
    elif strategy == "similar":
        return similar_tasks(description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        return similar_auto_breakdowns(description, io_pairs, N=6)
    elif strategy == "llm_similar":
        return llm_similar_tasks(task_name, description, io_pairs, N=6)


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    global few_shot_pot_prompt
    task_name = "Summarize the given articles"
    task_description = "In these examples, you are given an Article. Break the input down into subtasks in order to solve the task. You can read and write to an external file memory to help summarize using functions from alexAndKylesAwesomeLibrary."
    
    prompt = get_few_shot_cot_prompt(task_name, task_description, strategy)

    interpreter = TopDownVisitor(model_name=model_name, temperature=temperature)

    def predict_complete(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        outputs = gpt3(prompts)
        completed_outputs = [
            interpreter.complete_program(prefix, output)
            for prefix, output in zip(prompts, outputs)
        ]
        return completed_outputs

    predictions_list = []
    labels_list = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_labels = [label.split(" ")[0] for label in labels]
        for x in tqdm(chunks(inputs, 10)): # should we replace the chunks function by inputs.split('\n') to use paragraphs ? or use the chunks function with longer chunks?
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        predictions_list.append(preds)
        labels_list.append(new_labels)
    print("Rouge performance")
    print(rouge.compute(predictions=predictions_list, references=labels_list))


auto_cot_cleaned_prompt = """(Physics Questions) Answer these high-school-level physics multiple-choice questions.
Q: Lamar Gant, U.S. powerlifting star, became the first man to deadlift five times his own body weight in 1985. Deadlifting involves raising a loaded barbell from the floor to a position above the head with outstretched arms. Determine the work done by Lamar in deadlifting 260 kg to a height of 0.85 m above the ground.
A: Let's think step-by-step.

The work done by Lamar in deadlifting 260 kg to a height of 0.85 m above the ground is:
$W = \frac{1}{2} \cdot 260 \cdot 9.8 \cdot 0.85 = 12,091 \text{ J}$

The final answer is 12,091 J.
----
(Physics Questions) Answer these high-school-level physics multiple-choice questions.
Q: In the Funny Car competition at the Joliet Speedway in Joliet, Illinois in October of 2004, John Force complete the 1/4-mile dragster race in a record time of 4.437 seconds. Determine the average speed of the dragster in km/hr.
A: Let's think step-by-step.

First, we need to figure out how long it took the dragster to travel 1/4 of a mile. Since we know the dragster's speed in terms of seconds, we can convert 1/4 of a mile into seconds. There are 5280 feet in a mile, so 1/4 of a mile is 1320 feet. There are 3 feet in a yard, so 1/4 of a mile is also 440 yards. There are 3600 seconds in an hour, so 1/4 of a mile is also 1.2 seconds.
Now that we know how long it took the dragster to travel 1/4 of a mile, we can calculate the average speed. The dragster traveled 1/4 of a mile in 1.2 seconds, so the average speed is 1/4 of a mile divided by 1.2 seconds. This is equal to 0.208 miles per second, or 0.208 * 3600 = 750 km/hr.

The final answer is 750 km/hr.
----
(Physics Questions) Answer these high-school-level physics multiple-choice questions.
Q: A bicycle has a momentum of 24 kg*m/s. What momentum would the bicycle have if it had one-half the mass and was moving with thrice the speed?
A: Let's think step-by-step.
The bicycle's momentum is 24 kg*m/s.
The bicycle has a mass of 12 kg and a speed of 6 m/s.
If the bicycle had one-half the mass, it would have a mass of 6 kg.
If the bicycle had thrice the speed, it would have a speed of 18 m/s.
Therefore, the bicycle would have a momentum of 6*18=108 kg*m/s.

The final answer is 108 kg*m/s.
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
    description = "(Physics Questions) Answer these high-school-level physics multiple-choice questions."
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(
            model=model_name, max_length=500, temperature=0.7, quote="---", n=1
        )
        prompt = (
            """%s\n""" % description
            + io_pair[0]
            + """\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Anachronisms\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

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

    predictions_list = []
    labels_list = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)): # should we replace the chunks function by inputs.split('\n') to use paragraphs ? or use the chunks function with longer chunks?
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        predictions_list.append(preds)
        labels_list.append(labels)
    print("Rouge performance")
    print(rouge.compute(predictions=predictions_list, references=labels_list))


def nl_program(
    temperature=0.3,
    model_name="davinci-codex-002-msft",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Article Summarization"
    task_description = "(Physics Questions) Answer these high-school-level physics questions by applying the right physics formula, making substitutions, and storing the result in 'ans'."

    if strategy == "fixed":
        few_shot_cot_prompt = few_shot_cot_prompt
    elif strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(
            task_name, task_description, io_pairs, N=6
        )

    interpreter = TopDownVisitor(
        model_name=model_name, exclude_list=["[generate python code]"]
    )

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
        predictions_list = []
        labels_list = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            new_labels = [label.split(" ")[0] for label in labels]
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
            predictions_list.append(preds)
            labels_list.append(new_labels)
        print("Rouge performance")
        print(rouge.compute(predictions=predictions_list, references=labels_list))
        perf_array = []
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            new_labels = [label.split(" ")[0] for label in labels]
            for x in tqdm(chunks(inputs, 10)): # should we replace the chunks function by inputs.split('\n') to use paragraphs ? or use the chunks function with longer chunks?
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            predictions_list.append(preds)
            labels_list.append(new_labels)
        print("Rouge performance")
        print(rouge.compute(predictions=predictions_list, references=labels_list))


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
    parser.add_argument("--num_dev_examples", type=int, default=len(dev_inputs))
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
    print("Dev examples:", len(dev_inputs))

    # inputs = inputs[:args.num_dev_examples]
    # labels = labels[:args.num_dev_examples] # this lines are quite weird
    inputs = dev_inputs
    labels = dev_labels

    if args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            self_consistency=args.self_consistency,
            strategy=args.selection_strategy,
        )
