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
from utils import OpenAIModel, cache_dir, chunks, get_answer, get_few_shot_prompt

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

task_name = "mawps"
task_description = "(Mathematics) Solve the following arithmetic reasoning problems."


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


def process_data(data, split):
    labels = []
    ans_list = []
    for d in data[split]:
        try:
            ans = eval(list(d.values())[1].split("=")[-1].strip())
            labels.append(list(d.values())[1].split("=")[-1].strip())
            if isinstance(ans, int):
                ans_list.append(ans)
            elif (ans).is_integer():
                ans_list.append(int(ans))
            else:
                ans_list.append(float("%.2f" % ans))

        except:
            ans = eval(list(d.values())[1].split("=")[0].strip())
            labels.append(list(d.values())[1].split("=")[-1].strip())
            if isinstance(ans, int):
                ans_list.append(ans)
            elif (ans).is_integer():
                ans_list.append(int(ans))
            else:
                ans_list.append(float("%.2f" % ans))
    return labels, ans_list


data = datasets.load_dataset("omarxadel/MaWPS-ar", "train", cache_dir=cache_dir)
inputs = [list(d.values())[0] for d in data["validation"]]
labels, ans_list = process_data(data, "validation")


train_inputs = [list(d.values())[0] for d in data["train"]]
train_labels, train_ans_list = process_data(data, "train")

io_pairs = [(inp, lab) for inp, lab in zip(train_inputs[:5], train_labels[:5])]


def mawps(model_name="text-davinci-002", temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, temperature=temperature, max_length=200, quote="---", n=1
        )
        prompts = [
            """Amanda wants to diverge 608 Books among 28 friends. How many would each friend acquire?
608 / 28
----
Connie had 141 plum. Carrie took 139 from him. Now How many plum Connie have left?
141 - 139
----
Michael had 280 watermelon. Jasmine gripped some watermelon. Now Michael has 3  watermelon. How many did Jasmine grippeds?
280 - 3
----
Katherine acquire 9 bags of nectarine . how many nectarine in each bag? If total 89 nectarine Katherine  acquire.
89 / 9
----
 A company invited 24 people to a luncheon, but 10 of them didn't show up. If the tables they had held 7 people each, how many tables do they need?
((24.0-10.0)/7.0)
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
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        evaluated_predictions = []
        pred_expressions = []
        for p in preds:
            try:
                evaluated_predictions.append(eval(p))
            except:
                pred_expressions.append(p)
                # evaluated_predictions.append(eval(re.search("[0-9,.]+", p).group(0).replace(",", "")))
                evaluated_predictions.append(p)
        evaluated_labels = []  # [eval(l) for l in labels]
        for l in labels:
            try:
                evaluated_labels.append(eval(l))
            except:
                # evaluated_labels.append(eval(re.search("[0-9,.]+", l).group(0).replace(",", "")))
                evaluated_labels.append(p)
        # sometimes the model actually predicts the final answer
        perf_array.append(
            np.mean([1 if p == l else 0 for p, l in zip(evaluated_predictions, evaluated_labels)])
        )
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# mawps("code-davinci-002")
few_shot_cot_prompt = few_shot_arithmetic_prompt


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
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

    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

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
        new_labels = [str(s) for s in ans_list]
        perf_array.append(exact_match(new_labels, preds))
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
