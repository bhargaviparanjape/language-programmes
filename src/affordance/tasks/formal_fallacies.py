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

d = datasets.load_dataset("bigbench", "formal_fallacies_syllogisms_negation", cache_dir=cache_dir)
inputs = d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Formal Fallacies"
task_description = """(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations."""

io_pairs = [
    (
        """Q:"Here comes a perfectly valid argument: First of all, whatever is not both an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is an ingredient of All Good Things 12g or an ingredient of Restore.next, every ingredient of All Good Things 12g is not an ingredient of Eye brow pencil or not an ingredient of Minnie shampoo. From this follows: Whatever is an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is also an ingredient of Restore."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""",
        "invalid",
    ),
    (
        """Q:"Here comes a perfectly valid argument: First of all, being a granddaughter of Connie is necessary for being a niece of Marisa. Next, some granddaughter of Connie is not a close friend of Erika. All this entails that some close friend of Erika is not a niece of Marisa."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""",
        "invalid",
    ),
    (
        """Q:"Some football fans admire various clubs, others love only a single team. But who is a fan of whom precisely? The following argument pertains to this question: First of all, every opponent to FC Admira Wacker Mödling is a backer of Rosenborg BK and an admirer of HNK Rijeka. Next, Elias is not a backer of Rosenborg BK or not an admirer of HNK Rijeka. We may conclude: Elias is not an opponent to FC Admira Wacker Mödling."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""",
        "valid",
    ),
    (
        """Q:"Is Siri a stepsister of Mary? Is Susan related to Kate? In large families, it is sometimes difficult to keep track of all one\'s relatives. The following argument seeks to clarify some such relations: To begin with, Bonnie is a schoolmate of Miranda. Moreover, whoever is a workmate of Aubrey is not a schoolmate of Miranda. All this entails that Bonnie is not a workmate of Aubrey."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?\n""",
        "valid",
    ),
    (
        """Q:"Here comes a perfectly valid argument: First premise: Being a rare consumer of Tom\'s of Maine soap is sufficient for being a regular user of Dermasil soap. Second premise: Whoever is regular consumer of Crafted soap is at least one of these: an owner of a TIGI shampoo, a rare consumer of Tom\'s of Maine soap or a regular user of Redken shampoo. Third premise: Whoever is an owner of a TIGI shampoo is not a regular user of Dermasil soap. Hence, whoever is a regular consumer of Crafted soap is not a regular user of Dermasil soap or a regular user of Redken shampoo."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""",
        "invalid",
    ),
    (
        """Q:"Is Titanium oxide an ingredient of my washing power? Which chemicals does my perfume contain? It is really difficult to keep track of all chemicals one is regularly exposed to. The following argument seeks to clarify some such relations: First, everything that is an ingredient of Jest-er Kidding is an ingredient of Pink Nail Paint, too. Second, there is no ingredient of Jest-er Kidding who is an ingredient of Pink Nail Paint yet not an ingredient of MATT LIP GLOSS. Therefore, being an ingredient of MATT LIP GLOSS is necessary for being an ingredient of Jest-er Kidding."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""",
        "valid",
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

auto_cot_cleaned_prompt = """(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Here comes a perfectly valid argument: First of all, whatever is not both an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is an ingredient of All Good Things 12g or an ingredient of Restore.next, every ingredient of All Good Things 12g is not an ingredient of Eye brow pencil or not an ingredient of Minnie shampoo. From this follows: Whatever is an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is also an ingredient of Restore."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?
The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

The first premise is: "Whatever is not both an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is an ingredient of All Good Things 12g or an ingredient of Restore.next." This is a perfectly fine premise.
The second premise is: "Every ingredient of All Good Things 12g is not an ingredient of Eye brow pencil or not an ingredient of Minnie shampoo." This is also a perfectly fine premise.
From these two premises, the conclusion that "Whatever is an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is also an ingredient of Restore" does indeed logically follow. Therefore, the argument is valid.

The final answer is valid
----
(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Here comes a perfectly valid argument: First of all, being a granddaughter of Connie is necessary for being a niece of Marisa. Next, some granddaughter of Connie is not a close friend of Erika. All this entails that some close friend of Erika is not a niece of Marisa."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?
The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

The first premise states that being a granddaughter of Connie is necessary for being a niece of Marisa. In other words, all nieces of Marisa are granddaughters of Connie.
The second premise states that some granddaughter of Connie is not a close friend of Erika. This means that there is at least one granddaughter of Connie who is not a close friend of Erika.
From these two premises, we can deduce that some close friend of Erika is not a niece of Marisa. This is because if being a granddaughter of Connie is necessary for being a niece of Marisa, and there is at least one granddaughter of Connie who is not a close friend of Erika, then it follows that there is at least one close friend of Erika who is not a niece of Marisa. Therefore, the argument is valid.

The final answer is valid
----
(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Some football fans admire various clubs, others love only a single team. But who is a fan of whom precisely? The following argument pertains to this question: First of all, every opponent to FC Admira Wacker Mödling is a backer of Rosenborg BK and an admirer of HNK Rijeka. Next, Elias is not a backer of Rosenborg BK or not an admirer of HNK Rijeka. We may conclude: Elias is not an opponent to FC Admira Wacker Mödling."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?
The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

The first premise is "Every opponent to FC Admira Wacker Mödling is a backer of Rosenborg BK and an admirer of HNK Rijeka." This is a categorical syllogism in which the minor premise is the universal affirmative "Every opponent to FC Admira Wacker Mödling is a backer of Rosenborg BK" and the conclusion is the particular affirmative "Elias is a backer of Rosenborg BK." The syllogism is valid, but the first premise is false. The second premise is "Elias is not a backer of Rosenborg BK or not an admirer of HNK Rijeka." This is a disjunctive syllogism in which the first disjunct is the particular negative "Elias is not a backer of Rosenborg BK" and the second disjunct is the particular negative "Elias is not an admirer of HNK Rijeka." The conclusion is the particular negative "Elias is not an opponent to FC Admira Wacker Mödling." The syllogism is valid, but the first premise is false. Therefore, the argument is invalid.

The final answer is invalid
----
(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Is Siri a stepsister of Mary? Is Susan related to Kate? In large families, it is sometimes difficult to keep track of all one's relatives. The following argument seeks to clarify some such relations: To begin with, Bonnie is a schoolmate of Miranda. Moreover, whoever is a workmate of Aubrey is not a schoolmate of Miranda. All this entails that Bonnie is not a workmate of Aubrey."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?

The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

The first premise is "Bonnie is a schoolmate of Miranda." This is true.
The second premise is "Whoever is a workmate of Aubrey is not a schoolmate of Miranda." This is also true.
The conclusion is "Bonnie is not a workmate of Aubrey." This follows logically from the premises, so the argument is valid.

The final answer is valid
----
(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Here comes a perfectly valid argument: First premise: Being a rare consumer of Tom's of Maine soap is sufficient for being a regular user of Dermasil soap. Second premise: Whoever is regular consumer of Crafted soap is at least one of these: an owner of a TIGI shampoo, a rare consumer of Tom's of Maine soap or a regular user of Redken shampoo. Third premise: Whoever is an owner of a TIGI shampoo is not a regular user of Dermasil soap. Hence, whoever is a regular consumer of Crafted soap is not a regular user of Dermasil soap or a regular user of Redken shampoo."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?
The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

The first premise is "Being a rare consumer of Tom's of Maine soap is sufficient for being a regular user of Dermasil soap." This is not a valid premise, because it is possible to be a rare consumer of Tom's of Maine soap and not be a regular user of Dermasil soap.
The second premise is "Whoever is regular consumer of Crafted soap is at least one of these: an owner of a TIGI shampoo, a rare consumer of Tom's of Maine soap or a regular user of Redken shampoo." This is a valid premise.
The third premise is "Whoever is an owner of a TIGI shampoo is not a regular user of Dermasil soap." This is not a valid premise, because it is possible to be an owner of a TIGI shampoo and also be a regular user of Dermasil soap.
Therefore, the argument is invalid.

The final answer is invalid
----
(Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Q:"Is Titanium oxide an ingredient of my washing power? Which chemicals does my perfume contain? It is really difficult to keep track of all chemicals one is regularly exposed to. The following argument seeks to clarify some such relations: First, everything that is an ingredient of Jest-er Kidding is an ingredient of Pink Nail Paint, too. Second, there is no ingredient of Jest-er Kidding who is an ingredient of Pink Nail Paint yet not an ingredient of MATT LIP GLOSS. Therefore, being an ingredient of MATT LIP GLOSS is necessary for being an ingredient of Jest-er Kidding."
 Is the argument, given the explicitly stated premises, deductively valid or invalid?
The final answer is either "valid" or "invalid".
A: Let's think step-by-step.

First, everything that is an ingredient of Jest-er Kidding is an ingredient of Pink Nail Paint, too.
Second, there is no ingredient of Jest-er Kidding who is an ingredient of Pink Nail Paint yet not an ingredient of MATT LIP GLOSS.
From these two premises, the conclusion does not logically follow. The argument is invalid.

The final answer is invalid
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
            + """\nThe final answer is either "valid" or "invalid".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is either "valid" or "invalid".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is either "valid" or "invalid".\nA: Let's think step-by-step.\n"""
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
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
            preds = [get_autocot_answer(x, answer_prompt="The final answer is ") for x in answers]
            perf_array.append(exact_match(labels, preds))
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
            x = [ex.replace("""Q: "Here comes a perfectly valid argument: """, '"') for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# Q:"Here comes a perfectly valid argument: First of all, whatever is not both an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is an ingredient of All Good Things 12g or an ingredient of Restore.next, every ingredient of All Good Things 12g is not an ingredient of Eye brow pencil or not an ingredient of Minnie shampoo. From this follows: Whatever is an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is also an ingredient of Restore."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""", 'invalid'),
# ("""Q:"Here comes a perfectly valid argument: First of all, being a granddaughter of Connie is necessary for being a niece of Marisa. Next, some granddaughter of Connie is not a close friend of Erika. All this entails that some close friend of Erika is not a niece of Marisa."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""", 'invalid'),
# ("""Q:"Some football fans admire various clubs, others love only a single team. But who is a fan of whom precisely? The following argument pertains to this question: First of all, every opponent to FC Admira Wacker Mödling is a backer of Rosenborg BK and an admirer of HNK Rijeka. Next, Elias is not a backer of Rosenborg BK or not an admirer of HNK Rijeka. We may conclude: Elias is not an opponent to FC Admira Wacker Mödling."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""", 'valid'),
# ("""Q:"Is Siri a stepsister of Mary? Is Susan related to Kate? In large families, it is sometimes difficult to keep track of all one\'s relatives. The following argument seeks to clarify some such relations: To begin with, Bonnie is a schoolmate of Miranda. Moreover, whoever is a workmate of Aubrey is not a schoolmate of Miranda. All this entails that Bonnie is not a workmate of Aubrey."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?\n""", 'valid'),
# ("""Q:"Here comes a perfectly valid argument: First premise: Being a rare consumer of Tom\'s of Maine soap is sufficient for being a regular user of Dermasil soap. Second premise: Whoever is regular consumer of Crafted soap is at least one of these: an owner of a TIGI shampoo, a rare consumer of Tom\'s of Maine soap or a regular user of Redken shampoo. Third premise: Whoever is an owner of a TIGI shampoo is not a regular user of Dermasil soap. Hence, whoever is a regular consumer of Crafted soap is not a regular user of Dermasil soap or a regular user of Redken shampoo."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?""", 'invalid'),
# ("""Q:"Is Titanium oxide an ingredient of my washing power? Which chemicals does my perfume contain? It is really difficult to keep track of all chemicals one is regularly exposed to. The following argument seeks to clarify some such relations: First, everything that is an ingredient of Jest-er Kidding is an ingredient of Pink Nail Paint, too. Second, there is no ingredient of Jest-er Kidding who is an ingredient of Pink Nail Paint yet not an ingredient of MATT LIP GLOSS. Therefore, being an ingredient of MATT LIP GLOSS is necessary for being an ingredient of Jest-er Kidding."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?
def human_decomp(model_name="text-davinci-002", temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [
            """Q:"Here comes a perfectly valid argument: First of all, whatever is not both an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is an ingredient of All Good Things 12g or an ingredient of Restore. Next, every ingredient of All Good Things 12g is not an ingredient of Eye brow pencil or not an ingredient of Minnie shampoo. From this follows: Whatever is an ingredient of Eye brow pencil and an ingredient of Minnie shampoo is also an ingredient of Restore."\n Is the argument, given the explicitly stated premises, deductively valid or invalid?

%s
A: Let's think step by step."""
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
            x = [ex.replace("Q:", "") for ex in x]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
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
