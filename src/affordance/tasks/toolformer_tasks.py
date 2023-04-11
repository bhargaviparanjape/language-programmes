import argparse
import os

import datasets
import numpy as np
from prompt_library import (
    few_shot_retrieval_prompt,
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
    get_few_shot_prompt,
    substring_match,
    substring_match_v2,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# task_name = "SQuAD"
# task_description = "(SQuAD) Given an incomplete sentence with a word or phrase replaced with [MASK], identify the word or phrase to fill in so that the sentence is factually accurate."

# task_name = "T-ReX"
# task_description = "(T-ReX) Given an incomplete sentence with a word or phrase replaced with [MASK], identify the word or phrase to fill in so that the sentence is factually accurate."

# task_name = "triviaqa"
# task_description = "(Trivia QA) Given a trivia question, generate an answer to the question."

task_name = "natural_questions"
task_description = "(Natural Questions) Answer the following information seeking questions."

dataset = "NQ"
if dataset == "NQ" or dataset == "triviaqa":
    if dataset == "NQ":
        data_files = {
            split: os.path.join(cache_dir, "nq", "v1.0-simplified_nq-dev-all.jsonl")
            for split in ["train", "dev"]
        }
        d = datasets.load_dataset("json", data_files=data_files)
        labels, inputs = [], []
        for x in d["dev"]:
            tokens = [token["token"] for token in x["document_tokens"]]
            answers = []
            for annotation in x["annotations"]:
                if annotation["yes_no_answer"] != "NONE":
                    answers.append(annotation["yes_no_answer"].lower())
                for short_ans in annotation["short_answers"]:
                    answers.append(
                        " ".join(tokens[short_ans["start_token"] : short_ans["end_token"]])
                    )
            if len(answers):
                labels.append(answers)
                inputs.append(x["question_text"])

        inputs = inputs[:500]
        labels = labels[:500]

        train_inputs = inputs
        train_labels = labels
    else:
        d = datasets.load_dataset("trivia_qa", "rc.nocontext", cache_dir=cache_dir)
        inputs = d["validation"]["question"]
        labels = [x["normalized_aliases"] for x in d["validation"]["answer"]]
        train_inputs = d["train"]["question"]
        train_labels = [x["normalized_aliases"] for x in d["train"]["answer"]]
elif dataset == "trex" or dataset == "squad":
    d = datasets.load_dataset("lama", dataset, cache_dir=cache_dir)
    inputs = d["train"]["masked_sentence"]
    labels = d["train"]["obj_label"]
    selected = np.random.choice(len(inputs), size=500, replace=False)
    inputs = [inputs[idx] for idx in selected]
    labels = [labels[idx] for idx in selected]
    train_inputs = d["train"]["masked_sentence"]
    train_labels = [[l] for l in d["train"]["obj_label"]]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def exact_match_v2(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        label_list = [l.lower() for l in label]
        if predict.lower() in label_list:
            correct += 1
        count += 1
    return (1.0 * correct) / count


io_pairs = [(inp, lab) for inp, lab in zip(train_inputs[:5], train_labels[:5])]


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    global dataset
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
        if dataset == "triviaqa":
            perf_array.append(exact_match_v2(labels, preds))
        else:
            perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_cot_prompt = few_shot_retrieval_prompt


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    global dataset

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

    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 3
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [get_answer(x.strip()) for x in answers]
        if dataset == "triviaqa" or dataset == "NQ":
            perf_array.append(substring_match_v2(labels, preds))
        else:
            perf_array.append(substring_match(labels, preds))
        print(perf_array)
        positive_calls = [
            int(len(stack_trace_list) >= 1) for stack_trace_list in interpreter.execution_details
        ]
        positive_rate = sum(positive_calls) / len(interpreter.execution_details)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))
    print("Rate of affordance call", positive_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="astronomy")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "text-davinci-001",
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
    elif args.inference_strategy == "nl_program":
        nl_program(
            args.temperature,
            args.model_name,
            self_consistency=args.self_consistency,
            strategy=args.selection_strategy,
        )
