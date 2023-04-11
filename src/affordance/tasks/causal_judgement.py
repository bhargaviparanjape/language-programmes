import argparse
from collections import Counter
import re

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

d = datasets.load_dataset("bigbench", "causal_judgment", cache_dir=cache_dir)
inputs = d["validation"]["inputs"] + d["train"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"] + d["train"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))


io_pairs = [
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: Billy and Suzy work for a company that has a central computer. If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted. In order to make sure that two people are available to answer phone calls during designated calling hours, the company issued the following official policy: Billy and Suzy are both permitted to log in to the central computer in the mornings, and neither of them are permitted to log in to the central computer in the afternoons. This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time. Immediately, some work emails are deleted. Did Billy cause the emails to be deleted?""",
        "No",
    ),
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: Alice and Zoe work for the same company. They work in different rooms, and both of them sometimes need to access the central computer of the company. Unbeknownst to everybody, if two people are logged in to the central computer at the same time, some work emails containing important customer information are immediately deleted from the central computer. In order to make sure that one person is always available to answer incoming phone calls, the company issued the following official policy: Alice is the only one permitted to log in to the central computer in the mornings, whereas Zoe is the only one permitted to log in to the central computer in the afternoons. One day, violating the official policy, Zoe logs in to the central computer at 9 am. The same day, Alice also logs in at 9 am. Immediately, some work emails containing important customer information are deleted from the central computer. Did Zoe cause some work emails containing important customer information to be deleted from the central computer?""",
        "Yes",
    ),
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: There is a man who gets paid for pumping water into a cistern thereby replenishing the supply of drinking water in a nearby house. Unfortunately for the inhabitants of the house, the water that the man is pumping into the cistern today has been systematically contaminated with a lethal poison whose effects are unnoticeable until they can no longer be cured. Even though the man pumping the water had nothing to do with poisoning the water, he knows that the water has been poisoned. Nevertheless, the man pumps the water into the cistern knowing that it will poison and kill the inhabitants. But, he neither wants to kill them nor does he aim to do so, he simply wants to do his job and get paid. He views the death of the inhabitants as an unfortunate by-product of his pumping water into the cistern. Did the man intentionally poison the inhabitants?\n""",
        "Yes",
    ),
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: Jim, Carol, Bob, and Nancy are researchers in a remote area, and they have a limited supply of electricity. Because of their limited supply, the electricity only comes on in the evenings from 8-9 PM, and they have to restrict who can use power on certain days. If four people turn on their lamps at the same time, the breaker will fail. The breaker will not fail if fewer people turn on their lamps at the same time. Jim, Carol, Bob, and Nancy are all allowed to use their lamps on Thursdays. This Thursday Jim turns on his lamp at 8 PM. Just then Carol turns on her lamp, Bob also turns on his lamp, and Nancy turns on her lamp. Since four people turned on their lamps at the same time, the circuit breaker failed. Did Jim turning on his lamp at 8 PM cause the circuit breaker to fail?""",
        "No",
    ),
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: Billy and Suzy work for a company that has a central computer. If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted. In order to make sure that one person is always available to answer incoming phone calls, the company issued the following official policy: Billy is the only one permitted to log in to the central computer in the afternoons, whereas Suzy is the only one permitted to log in to the central computer in the mornings. Billy is never permitted to log into the central computer in the morning. This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time. Immediately, some work emails are deleted. Did Billy cause the emails to be deleted?""",
        "Yes",
    ),
    (
        """How would a typical person answer each of the following questions about causation?\n\n\nQ: Alice and Zoe work for the same company. They work in different rooms and both of them sometimes need to access the central computer of the company. Unbeknownst to everybody, if two people are logged in to the central computer at the same time, an empty email is immediately sent from the central computer to a non-existent email address. One day, Zoe logs in to the central computer at 9 am. The same day, Alice logs in at 9:30 am, when Zoe is already logged in. Immediately, an empty email is sent from the central computer to a non-existent email address. Did Zoe cause an empty email to be sent from the central computer to a non-existent email address?""",
        "No",
    ),
]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Causal Judgement"
task_description = """(Causal Judgement) Read a short story where multiple cause-effect events are introduced and answer causal questions such as "Did X cause Y?\""""


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

auto_cot_cleaned_prompt = """(Causal Judgement) Read a short story where multiple cause-effect events are introduced and answer causal questions such as "Did X cause Y?"
How would a typical person answer each of the following questions about causation?


Q: Billy and Suzy work for a company that has a central computer. If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted. In order to make sure that two people are available to answer phone calls during designated calling hours, the company issued the following official policy: Billy and Suzy are both permitted to log in to the central computer in the mornings, and neither of them are permitted to log in to the central computer in the afternoons. This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time. Immediately, some work emails are deleted. Did Billy cause the emails to be deleted?
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

Billy and Suzy work for a company that has a central computer.
If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted.
In order to make sure that two people are available to answer phone calls during designated calling hours, the company issued the following official policy: Billy and Suzy are both permitted to log in to the central computer in the mornings, and neither of them are permitted to log in to the central computer in the afternoons.
This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time.
Immediately, some work emails are deleted.
Did Billy cause the emails to be deleted?
Yes, Billy caused the emails to be deleted.

The final answer is Yes
----
(Causal Judgement) Read a short story where multiple cause-effect events are introduced and answer causal questions such as "Did X cause Y?"
How would a typical person answer each of the following questions about causation?


Q: There is a man who gets paid for pumping water into a cistern thereby replenishing the supply of drinking water in a nearby house. Unfortunately for the inhabitants of the house, the water that the man is pumping into the cistern today has been systematically contaminated with a lethal poison whose effects are unnoticeable until they can no longer be cured. Even though the man pumping the water had nothing to do with poisoning the water, he knows that the water has been poisoned. Nevertheless, the man pumps the water into the cistern knowing that it will poison and kill the inhabitants. But, he neither wants to kill them nor does he aim to do so, he simply wants to do his job and get paid. He views the death of the inhabitants as an unfortunate by-product of his pumping water into the cistern. Did the man intentionally poison the inhabitants?

The final answer is either "Yes" or "No".
A: Let's think step-by-step.

The man did not poison the water himself.
The man knew the water had been poisoned.
The man pumped the water into the cistern anyway.
Based on these three points, it seems like the answer should be "No". The man did not intentionally poison the inhabitants - he was just doing his job. However, if we consider the fact that the man knew the water was poisoned and he pumped it into the cistern anyway, we could argue that he did have the intention of poisoning the inhabitants. In this case, the answer would be "Yes".

The final answer is No
----
(Causal Judgement) Read a short story where multiple cause-effect events are introduced and answer causal questions such as "Did X cause Y?"
How would a typical person answer each of the following questions about causation?


Q: Billy and Suzy work for a company that has a central computer. If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted. In order to make sure that one person is always available to answer incoming phone calls, the company issued the following official policy: Billy is the only one permitted to log in to the central computer in the afternoons, whereas Suzy is the only one permitted to log in to the central computer in the mornings. Billy is never permitted to log into the central computer in the morning. This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time. Immediately, some work emails are deleted. Did Billy cause the emails to be deleted?
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

Billy and Suzy work for a company that has a central computer.
If two people log in to the central computer at exactly 9:27 am, some work emails will be immediately deleted.
In order to make sure that one person is always available to answer incoming phone calls, the company issued the following official policy: Billy is the only one permitted to log in to the central computer in the afternoons, whereas Suzy is the only one permitted to log in to the central computer in the mornings. Billy is never permitted to log into the central computer in the morning.
This morning at exactly 9:27 am, Billy and Suzy both log into the central computer at the same time.
Immediately, some work emails are deleted.
Did Billy cause the emails to be deleted?
A: Yes

The final answer is Yes
----
(Causal Judgement) Read a short story where multiple cause-effect events are introduced and answer causal questions such as "Did X cause Y?"
How would a typical person answer each of the following questions about causation?


Q: Alice and Zoe work for the same company. They work in different rooms and both of them sometimes need to access the central computer of the company. Unbeknownst to everybody, if two people are logged in to the central computer at the same time, an empty email is immediately sent from the central computer to a non-existent email address. One day, Zoe logs in to the central computer at 9 am. The same day, Alice logs in at 9:30 am, when Zoe is already logged in. Immediately, an empty email is sent from the central computer to a non-existent email address. Did Zoe cause an empty email to be sent from the central computer to a non-existent email address?
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

Zoe logs in to the central computer at 9 am.
The same day, Alice logs in at 9:30 am, when Zoe is already logged in.
Immediately, an empty email is sent from the central computer to a non-existent email address.
Did Zoe cause an empty email to be sent from the central computer to a non-existent email address?
Yes, Zoe caused an empty email to be sent from the central computer to a non-existent email address.

The final answer is Yes
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
            + """\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""
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
