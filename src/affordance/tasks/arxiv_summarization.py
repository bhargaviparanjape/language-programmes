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
import evaluate

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

rouge = evaluate.load("rouge")

# TODO Only using 10 examples for now
train_dataset = datasets.load_dataset("ccdv/arxiv-summarization", split="train[:10]")
train_inputs = train_dataset["article"]
train_inputs = [t.split("\n").strip() for t in train_inputs]  # TODO: concatenate 5 paragraphs
train_labels = train_dataset["abstract"]
dev_dataset = datasets.load_dataset("ccdv/arxiv-summarization", split="validation[:10]")
dev_inputs = dev_dataset["article"]
dev_inputs = [t.split("\n").strip() for t in dev_inputs]  # TODO: concatenate 5 paragraphs
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

def nl_program(
    model_name="davinci-codex-002-msft",
    temperature=0.3,
    strategy="fixed",
    run_title = "",
):
    task_name = "Physics Questions"
    task_description = "(Physics Questions) Answer these high-school-level physics questions by applying the right physics formula, making substitutions, and storing the result in 'ans'."

    prompt = get_few_shot_cot_prompt(task_name, task_description, strategy)

    interpreter = TopDownVisitor(model_name)


    n = 1
    runs = 1
    batch_size = 10

    model = OpenAIModel(model_name, quote="---", temperature=temperature, max_length=1000, n=n)

    ### While setting up the inputs, we also need to set up a list of article IDs to pass as part of run titles
    # run_titles = ['article1', 'article3', 'article1',...]

    


    perf_array = []
    with tqdm(total=runs * (len(inputs) // batch_size + (len(inputs) % batch_size > 0))) as pbar:
        for run in range(runs):
            pbar.set_description(f"run {run+1}/{runs}")

            answers = []

            for x in chunks(new_inputs, batch_size):
                prompts = [prompt.format(task_description, ex) for ex in x]
                completions = model(prompts)

                # run the answers for all prompts at once
                completions = interpreter.batch_visit(prompts, completions, )
                answers.extend(get_answer(x) for x in completions)
    

                pbar.update(1)

            perf_array.append(substring_match(new_labels, answers))

    print("FS-CoT Performance:", perf_array)
    print("- mean:", np.mean(perf_array))
    print("- std dev:", np.std(perf_array))
    positive_calls = [len(stack_trace) >= 1 for stack_trace in interpreter.execution_details]
    positive_rate = sum(positive_calls) / len(positive_calls)
    print("- rate of affordance call", positive_rate)


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
    parser.add_argument("--run-title", type=str, default="<TITLE>")
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

    inputs = dev_inputs
    labels = dev_labels

    if args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            strategy=args.selection_strategy,
            run_title = args.run_title
        )
