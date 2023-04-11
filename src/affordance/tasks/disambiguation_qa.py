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
    substring_match_v2,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "disambiguation_qa", cache_dir=cache_dir)
inputs = d["validation"]["inputs"] + d["train"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["validation"]["targets"] + d["train"]["targets"]
labels = [l[0] for l in labels]
print(len(inputs))

io_pairs = [
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The chief encouraged the writer because she wrote a good book.\n  choice: Ambiguous\n  choice: The chief wrote a good book\n  choice: The writer wrote a good book",
        "The writer wrote a good book",
    ),
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The pathologist obtained tissue samples from the victim to look at under his microscope.\n  choice: Ambiguous\n  choice: It was the pathologist's microscope\n  choice: It was the victim's microscope",
        "It was the pathologist's microscope",
    ),
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The paralegal forgot to file paperwork for the client so she was fired.\n  choice: Ambiguous\n  choice: The client was fired\n  choice: The paralegal was fired",
        "The paralegal was fired",
    ),
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The mechanic greets the receptionist because they are standing in front of the door.\n  choice: Ambiguous\n  choice: The mechanic is standing in front of the door\n  choice: The receptionist is standing in front of the door",
        "Ambiguous",
    ),
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The taxpayer met with the accountant to find out if he could help with tax preparation.\n  choice: Ambiguous\n  choice: The accountant could help\n  choice: The taxpayer could help",
        "The accountant could help",
    ),
    (
        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\nSentence: The patient was referred to the specialist because they had a rare skin condition.\n  choice: Ambiguous\n  choice: The patient had a skin condition\n  choice: The specialist had a skin condition",
        "The patient had a skin condition",
    ),
]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_name = "Disambiguation QA"
task_description = """(Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices."""


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


def human_decomp(temperature=0.3):
    def predict(chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=1000,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [
            """Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The chief told the counselor that they took the day off.
  choice: The chief took the day off
  choice: The counselor took the day off
  choice: Ambiguous
A: Let's think step by step.
Here we need to determine who the pronoun "they" might be referring to. There are two possible referents for "they", namely the chief and the counselor. The verb "told" might be able to help us determine which one is more likely (if either). Let X be the chief and Y the counselor. The sentence is then of the form "X told Y that (X or Y) did something."
Let's consider Y first: "X told Y that Y did something." This case does not make much sense, as Y would already have the information that Y did something, because it is information about themself.
Now, consider X: "X told Y that X did something." This makes sense, because X would be sharing some information about themself that Y might not have known before.
Because in this context, X is the chief and Y is the counselor, the answer should be the chief. So the answer is The chief took the day off.
----
Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: The manager sent a message to the secretary, but he didn't reply yet.
  choice: The secretary didn't reply yet
  choice: The manager didn't reply yet
  choice: Ambiguous
A: Let's think step by step.
Here we need to determine who the pronoun "he" might be referring to. There are two possible referents for "he", namely the manager and the secretary. The verbs "sent" and "reply" might be able to help us determine which one is more likely (if either). Let X be the manager and Y the secretary. The sentence is then of the form "X sent a message to Y, but (X or Y) didn't reply yet."
Let's consider Y first: "X sent a message to Y, but Y didn't reply yet." This case makes sense, because of the implicit causality of the sentence. Y was the receiver of the message, but Y didn't get back to X yet.
Now, consider X: "X sent a message to Y, but X didn't reply yet." This case doesn't make sense, because X was the initial sender of the message, so it is now Y's turn to write back to X.
Because in this context, X is the manager and Y is the secretary, the answer should be the secretary. So the answer is The secretary didn't reply yet.
----
Q: In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.
Sentence: Bailey will plan to meet the director at his office
  choice: It will be Bailey's office
  choice: It will be the director's office
  choice: Ambiguous
A: Let's think step by step.
Here we need to determine who the pronoun "his" might be referring to. There are two possible referents for "his", namely Bailey's and the director's. The verb phrase "plan to meet" might be able to help us determine which one is more likely (if either). Let X be Bailey and Y the director. The sentence is then of the form "X will plan to meet Y at (X or Y)'s office."
Let's consider Y first: "X will plan to meet Y at Y's office." This case makes sense, because X might want to meet up with Y at Y's office.
Now, consider X: "X will plan to meet Y at X's office." This case also makes sense, because X might want to meet up with Y at X's own office.
Because both X and Y are possible at the same time, we conclude that the antecedent of the pronoun is ambiguous. So the answer is Ambiguous.
----
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
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
        preds = [get_autocot_answer(x, answer_prompt="So the answer is ") for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Human decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """"""

auto_cot_cleaned_prompt = """(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The chief encouraged the writer because she wrote a good book.
  choice: Ambiguous
  choice: The chief wrote a good book
  choice: The writer wrote a good book
The final answer is one of the choices.
A: Let's think step-by-step.

1) "The chief encouraged the writer" - we know that the chief encouraged someone, but we don't know who.
2) "because she wrote a good book" - we know that the reason the chief encouraged the writer is because the writer wrote a good book.

From this, we can infer that the antecedent of "she" is the writer.

The final answer is "The writer wrote a good book"
----
(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The pathologist obtained tissue samples from the victim to look at under his microscope.
  choice: Ambiguous
  choice: It was the pathologist's microscope
  choice: It was the victim's microscope
The final answer is one of the choices.
A: Let's think step-by-step.
The pronoun in question is "his." "His" could refer to the pathologist or the victim.
The pathologist is the subject of the sentence and is male, so "his" could refer to the pathologist's microscope.
The victim is also male, so "his" could refer to the victim's microscope.
We need more information to determine which antecedent "his" is referring to.

The final answer is "Ambiguous"
----
(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The paralegal forgot to file paperwork for the client so she was fired.
  choice: Ambiguous
  choice: The client was fired
  choice: The paralegal was fired
The final answer is one of the choices.
A: Let's think step-by-step.

The antecedent of "she" is the most recently mentioned female noun or pronoun, which in this case is "the paralegal."

The final answer is "The paralegal was fired"
----
(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The mechanic greets the receptionist because they are standing in front of the door.
  choice: Ambiguous
  choice: The mechanic is standing in front of the door
  choice: The receptionist is standing in front of the door
The final answer is one of the choices.
A: Let's think step-by-step.

The pronoun in question is "they." We can try to figure out its antecedent by looking at the other nouns in the sentence and seeing which one makes the most sense.
The mechanic greets the receptionist because they are standing in front of the door.
In this sentence, it makes the most sense that "they" is referring to the mechanic and the receptionist. This is because the mechanic is greeting the receptionist, and they are both standing in front of the door.

The final answer is "Ambiguous"
----
(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The taxpayer met with the accountant to find out if he could help with tax preparation.
  choice: Ambiguous
  choice: The accountant could help
  choice: The taxpayer could help
The final answer is one of the choices.
A: Let's think step-by-step.

First, we need to identify the pronoun in question: "he." Next, we need to ask ourselves what this pronoun could be referring to. In this sentence, there are two potential antecedents for "he"--the taxpayer or the accountant. Because both of these nouns are masculine, we can't use pronoun agreement to help us determine the antecedent.
In this case, we need to look at the context of the sentence to try to determine which antecedent is more likely. Based on the context, it seems more likely that the pronoun "he" is referring to the accountant. This is because the sentence is talking about the taxpayer meeting with the accountant to discuss tax preparation. It seems more likely that the pronoun "he" would be referring to the accountant in this context, rather than the taxpayer.

The final answer is "The accountant could help"
----
(Disambiguation QA) Given a sentence with an ambiguous pronoun, choose the person to whom the pronoun refers. When the meaning of the sentence is ambiguous, return the answer "ambiguous".
In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.

Sentence: The patient was referred to the specialist because they had a rare skin condition.
  choice: Ambiguous
  choice: The patient had a skin condition
  choice: The specialist had a skin condition
The final answer is one of the choices.
A: Let's think step-by-step.

"The patient" is the subject of the sentence, so we know that the pronoun "they" is referring to the patient.
"They" is referred to the specialist because "they" had a rare skin condition.
The pronoun "they" refers to the patient.

The final answer is "The patient had a skin condition"
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
                x = [ex.replace("\nPronoun identification:", "") for ex in x]
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
        label_dict = {0: "(A)", 1: "(B)", 2: "(C)"}
        choices = [
            [ans.strip() for ans in inp.replace("\nPronoun identification:", "").split("choice: ")][
                1:
            ]
            for inp in inputs
        ]
        new_labels = [label_dict[choice.index(label)] for label, choice in zip(labels, choices)]
        joint_labels = [[label, new_label] for label, new_label in zip(labels, new_labels)]
        for x in tqdm(chunks(inputs, 10)):
            x = [
                ex.replace(
                    "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\n",
                    "",
                )
                for ex in x
            ]
            x = [ex.replace("\nPronoun identification:", "") for ex in x]
            # answers.extend(predict(task_description, x))
            answers.extend(predict_complete(task_description, x))
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match_v2(joint_labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


few_shot_human_prompt = """Description: (Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices.
Input: Sentence: The technician told the customer that they had completed the repair.
  choice: Ambiguous
  choice: The customer completed the repair
  choice: The technician completed the repair
Q1: [think step-by-step]
#1: Let's think step-by-step. In this sentence, the pronoun "they" refers to "the technician". So the answer is "The technician completed the repair".
----
Description: (Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices.
Input: Sentence: The reporter and the chef discuss their favorite dishes.
  choice: Ambiguous
  choice: They discuss the chef's favorite dishes
  choice: They discuss the reporter's favorite dishes
Q1: [think step-by-step]
#1: Let's think step-by-step. "The reporter and the chef discuss their favorite dishes." In this sentence, "their" refers to "the reporter and the chef". So the answer is "They discuss the reporter's favorite dishes" because "the reporter and the chef" is a plural pronoun and it refers to two people.
Q2: [EOQ]
Ans: They discuss the reporter's favorite dishes
----
Description: (Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices.
Input: Sentence: After meeting with the producers, Sam will go to their office.
  choice: Ambiguous
  choice: It will be Sam's office
  choice: It will be the producers' office
Q1: [think step-by-step]
#1: Let's think step-by-step. "After meeting with the producers, Sam will go to their office." "Their" refers to the producers. So the answer is (It will be the producers' office).
----
Description: (Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices.
Input: Sentence: The student set up a meeting with the architect to review his blueprints.
  choice: Ambiguous
  choice: The blueprints were the architect's
  choice: The blueprints were the student's
Q1: [think step-by-step]
#1: Let's think step-by-step. "The student" is the subject of the sentence and "the architect" is the object. So "his" refers to "the architect". So the answer is (B).
Q2: [EOQ]
Ans: (B)
----
Description: (Disambiguation QA) Given a sentence with a pronoun, choose the person to whom the pronoun refers to. The final answer should be one of the choices.
Input: Sentence: The visitor tells the teacher that they like the cake.
  choice: Ambiguous
  choice: The teacher likes the cake
  choice: The visitor likes the cake
Q1: [think step-by-step]
#1: Let's think step-by-step. The pronoun "they" could refer to either "the visitor" or "the teacher". So the answer is ambiguous.
Q2: [EOQ]
Ans: Ambiguous
----
"""


def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_human_prompt
    interpreter = TopDownVisitorBeta(model_name=model_name)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1500, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    def predict_self_consistency(description, chunk, n=9):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=n
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [
                    ex.replace(
                        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\n",
                        "",
                    )
                    for ex in x
                ]
                x = [ex.replace("\nPronoun identification:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 9)):
                    new_answer = interpreter.batch_visit(
                        [prompts[chunk_no]] * len(ans_chunk), ans_chunk
                    )
                    processed_answers = [get_answer(ex) for ex in new_answer]
                    for pred in enumerate(processed_answers):
                        # Only add to the counter if there is a legitimate answer
                        if pred is not None:
                            result_counter[chunk_no].update([pred])
                answers.extend(result_counter)
            preds = [x.most_common(1)[0][0][1] for x in answers[: len(inputs)]]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))

    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            label_dict = {0: "(A)", 1: "(B)", 2: "(C)"}
            choices = [
                [
                    ans.strip()
                    for ans in inp.replace("\nPronoun identification:", "").split("choice: ")
                ][1:]
                for inp in inputs
            ]
            new_labels = [label_dict[choice.index(label)] for label, choice in zip(labels, choices)]
            joint_labels = [[label, new_label] for label, new_label in zip(labels, new_labels)]
            for x in tqdm(chunks(inputs, 10)):
                x = [
                    ex.replace(
                        "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\n\n",
                        "",
                    )
                    for ex in x
                ]
                x = [ex.replace("\nPronoun identification:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(new_labels, preds))
            # Report on interpreter performance
            print(perf_array)
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / (len(interpreter.execution_details) + 1e-6)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


# human_decomp("text-davinci-002", 0.3)
# human_intervention(0.3, "text-davinci-002")
# human_decomp()

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
