import argparse
from collections import Counter
import pdb
import re

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
    get_autocot_answer,
    get_few_shot_prompt,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

d = datasets.load_dataset("bigbench", "physics", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
# inputs = [x.split('\n')[0] for x in inputs]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

task_description = "Identify the physics formula that would be most useful for finding the answer to each of the following word problems."

io_pairs = [
    (
        """Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t""",
        "F = q * E",
    ),
    (
        """Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt""",
        "Q = m * c * dT",
    ),
    (
        """Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt""",
        "F = m * a",
    ),
    (
        """Q: A 6.6 N object is traveling at a velocity of 3.0 m/s north. What is the object’s momentum?
  choice: p = m * v
  choice: v = v_0 + a * t
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2
  choice: q = Q * dθ / π""",
        "p = m * v",
    ),
    (
        """Q: A 1250 W electric motor is used to lift an 80.0 kg weight to a height of 4.0 m in 3.00 s. What is the efficiency of the motor?
  choice: U = m * g * h
  choice: dt = dx / v
  choice: E = K + U + Q
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2""",
        "U = m * g * h",
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


def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0 * correct) / count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):
    few_shot_prompt = get_few_shot_prompt(inputs, [[l] for l in labels], n=N)
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
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def physics():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t
A:
F = q * E
----
Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt
A:
Q = m * c * dT
----
Identify the physics formula that would be most useful for finding the answer to each of the following word problems.


Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt
A:
F = m * a
----
%s"""
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
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1:
# Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin.
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon.
# Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: [compare] What is the final answer?
# #3: Unknown
# Q4: [EOQ]
# Ans: Unknown
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
# Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
# Q1: [tag] What are the entities in this sentence?
# #1:
# President George H. W. Bush
# Gulf War
# Q2: [search] When was President George H. W. Bush president?
# #2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
# Q3: [search] When was the Gulf War fought?
# #3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
# Q4: Could these entities have co-existed based on thier time periods alone?
# #4: Yes. Their time periods intersect.
# Q5: [generate output] Is this an anachronism?
# #5: No
# Q6: [EOQ]
# Ans: No
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
# Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
# Q1: [tag] What are the entities in this sentence?
# #1:
# Kurt Cobain
# "Twin Peaks"
# Q2: [search] When did television show "Twin Peaks" air?
# #2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
# Q3: [search] When did Kurt Cobain live?
# #3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
# Q4: Could these entities have co-existed based on this information?
# #4: No. Musician  Kurt Cobain could not have starred in Twin Peaks.
# Q5: [generate output] Is this an anachronism?
# #5: Yes
# Q6: [EOQ]
# Ans: Yes
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#   choice: Anjalikastra
#   choice: Narayanastra
#   choice: Agneyastra
#   choice: Brahmastra
# Q1: [search] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
# #1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q2: [compare] Which option is the answer in #3 most similar to?
# #2: Brahmastra
# Q3: [EOQ]
# Ans: Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1:
# Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: [compare] What is the final answer?
# #3: Florida, Missouri
# Q4: [EOQ]
# Ans: Florida, Missouri
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_retrieval_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Physics questions"
    task_description = "(Physics questions) Given a physics word problem, choose which physics formula from among the choices is most helpful in solving the word problem. The final answer should be one of the choices."

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

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    interpreter = TopDownVisitorBeta(model_name=model_name, temperature=temperature)

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
            x = [
                ex.replace(
                    "Identify the physics formula that would be most useful for finding the answer to each of the following word problems.\n\n\n",
                    "",
                )
                for ex in x
            ]
            answers.extend(predict_complete(task_description, x))
            # time.sleep(10)
        preds = [get_answer(x.strip()) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_cleaned_prompt = """Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t
The final answer should be one of the choices.
A: Let's think step-by-step.

We know that the oil droplet is positively charged and that it is being held stationary by a vertical electric field. In order to find the charge on the oil droplet, we need to know the electric field strength.
The electric field strength is given by the equation:
F = q * E
where q is the charge on the oil droplet and E is the electric field strength.

The best answer choice is F = q * E
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt
The final answer should be one of the choices.
A: Let's think step-by-step.

First, we need to find the change in temperature of the ball. We can use the equation:
Q = m * c * dT
where Q is the heat lost, m is the mass of the ball, c is the specific heat capacity, and dT is the change in temperature.

The best answer choice is Q = m * c * dT
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt
The final answer should be one of the choices.
A: Let's think step-by-step.

We are given the mass of the car, the velocity of the car, and the coefficient of friction. We are looking for the minimum friction force.
The formula we are looking for is F = μ * N.
In order to compute normal reaction, we need to use the formula for force given mass and acceleration due to gravity. We should use the formula F = m * a

The best answer choice is F = m * a
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 6.6 N object is traveling at a velocity of 3.0 m/s north. What is the object’s momentum?
  choice: p = m * v
  choice: v = v_0 + a * t
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2
  choice: q = Q * dθ / π
The final answer should be one of the choices.
A: Let's think step-by-step.

We are given the object's mass (6.6 N) and velocity (3.0 m/s north). We want to calculate the object's momentum.

The momentum formula is: p = m * v

The best answer choice is p = m * v
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
A 1250 W electric motor is used to lift an 80.0 kg weight to a height of 4.0 m in 3.00 s. What is the efficiency of the motor?
  choice: U = m * g * h
  choice: dt = dx / v
  choice: E = K + U + Q
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2
The final answer should be one of the choices.
A: Let's think step-by-step.

First, we need to find the amount of work done by the motor. We can use the equation W = F * d, where W is work, F is force, and d is distance. We know the force (it's just the weight of the object being lifted), and we know the distance (it's 4.0 m).
The weight of the object being lifted F = m * g.
So the final formula for work is W = m * g * h

The best answer choice is U = m * g * h
----
"""

auto_cot_corrected_prompt = """Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: In an experiment, a positively charged oil droplet weighing 6.5 * 10 ^ -15 N is held stationary by a vertical electric field. If the electric field strength is 5.3 * 10 ^ 3 N/C, what is the charge on the oil droplet?
  choice: dt = dx / v
  choice: v = λ * f
  choice: F = q * E
  choice: v = v_0 + a * t
The final answer should be one of the choices.
A: Let's think step-by-step.

First, we need to figure out what quantity we're trying to solve for. In this case, it's the charge on the oil droplet.
Next, we need to identify which formula would be most useful for solving for that quantity. In this case, it's the formula for electric force, which is equal to the charge on an object times the electric field strength.
So, we can solve for the charge on the oil droplet by plugging in the given values for the electric field strength and the weight of the oil droplet, and solving for q.

The best answer choice is F = q * E.
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 3.0 kg ball rolls down from the top of a ramp as shown. If the ball is moving at 10.0 m/sat the bottom, how much energy was lost due to friction (thermal energy)?
  choice: Q = m * c * dT
  choice: ɸ = E * A * cos(θ)
  choice: E = F / q
  choice: a = dv / dt
The final answer should be one of the choices.
A: Let's think step-by-step.

1. We want to find the energy lost due to friction, so we're looking for a formula that deals with energy.
2. We know the mass of the ball, the velocity of the ball, and the change in temperature.
3. Q = m * c * dT is the formula we want!

The best answer choice is Q = m * c * dT
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 1200 kg car rounds a flat circular section of road at 20 m/s as shown in the diagram. The coefficient of friction between the car tires and the road surface is 0.65. What minimum friction force is required for the car to follow this curve?
  choice: F = m * a
  choice: dq = ρ * dV
  choice: E = q * σ / (2 * ε)
  choice: P = dE / dt
The final answer should be one of the choices.
A: Let's think step-by-step.

First, we need to calculate the force required to keep the car moving at 20 m/s. We can use the equation F = ma for this. We know the mass of the car (1200 kg) and the acceleration (20 m/s), so we just need to solve for F.
F = 1200 kg * 20 m/s
F = 24000 kg * m/s^2
Now that we know the force required to keep the car moving, we need to calculate the friction force. We know that the friction force is equal to the coefficient of friction times the normal force. Since this is a flat surface, the normal force is just equal to the weight of the car. We also know the coefficient of friction (0.65). So we just need to solve for the friction force.
Friction force = 0.65 * weight of the car
Friction force = 0.65 * (1200 kg * 9.8 m/s^2)
Friction force = 7824 N

The best answer choice is F = m * a.
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 6.6 N object is traveling at a velocity of 3.0 m/s north. What is the object’s momentum?
  choice: p = m * v
  choice: v = v_0 + a * t
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2
  choice: q = Q * dθ / π
The final answer should be one of the choices.
A: Let's think step-by-step.

We want to find the object's momentum, so we'll need a formula that includes both mass and velocity. That narrows it down to either p = m * v or v = v_0 + a * t.
But since we're only given the object's velocity and not its acceleration, we'll use p = m * v.

The best answer choice is p = m * v.
----
Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.
Q: A 1250 W electric motor is used to lift an 80.0 kg weight to a height of 4.0 m in 3.00 s. What is the efficiency of the motor?
  choice: U = m * g * h
  choice: dt = dx / v
  choice: E = K + U + Q
  choice: d = x_0 + v_0 * t + 1/2 * a * t ^ 2
The final answer should be one of the choices.
A: Let's think step-by-step.

What are we trying to find? The efficiency of the motor.
What formula would give us efficiency?
Efficiency = work output / work input
For work output, we can use the formula:
Work output = force * distance
For work input, we can use the formula:
Work input = power * time
Now we just need to plug in the given values and solve for efficiency.

The best answer choice is U = m * g * h.
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
    auto_cot_prompt = ""
    description = """Physics: Identify the physics formula that would be most useful for finding the answer to each of the following word problems."""
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=500, temperature=0.7, quote="---", n=1
        )
        prompt = (
            """%s\n""" % description
            + io_pair[0]
            + """\nThe final answer should be one of the choices.\nA: Let's think step-by-step.\n"""
        )
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    else:
        auto_cot_prompt = auto_cot_cleaned_prompt

    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt", "a+")
    f.write("Physics Questions\n\n")
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
            + """%s\nThe final answer should be one of the choices.\nA: Let's think step-by-step.\n"""
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
            + """%s\n""" % description
            + """%s\nThe final answer should be one of the choices.\nA: Let's think step-by-step.\n"""
            % x
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
                        if re.search("""The best answer choice is """, x):
                            preds.append(
                                x[re.search("""The best answer choice is """, x).span(0)[-1] :]
                            )
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The best answer choice is """, pred):
                            result_counter[chunk_no].update(
                                [pred[re.search("""The best answer choice is """, x).span(0)[-1] :]]
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
                x = [
                    ex.replace(
                        "Identify the physics formula that would be most useful for finding the answer to each of the following word problems.",
                        "",
                    )
                    for ex in x
                ]
                x = [ex.replace("\nA:", "") for ex in x]
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [
                get_autocot_answer(x, answer_prompt="The best answer choice is") for x in answers
            ]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def dynamic_few_shot_cot(temperature=0.3, strategy="random"):
    if strategy == "random":
        few_shot_cot_prompt = random_tasks(N=6)
    elif strategy == "similar":
        few_shot_cot_prompt = similar_tasks(task_description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        few_shot_cot_prompt = similar_auto_breakdowns(task_description, io_pairs, N=6)
    elif strategy == "llm_similar":
        few_shot_cot_prompt = llm_similar_tasks(task_description, io_pairs, N=6)

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model="text-davinci-002",
            max_length=2048,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [
                ex.replace(
                    "Identify the physics formula that would be most useful for finding the answer to each of the following word problems.",
                    "",
                )
                for ex in x
            ]
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt
    task_name = "Physics Formulas"
    task_description = "Identify the physics formula that would be most useful for finding the answer to each of the following word problems."

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

    if self_consistency:
        perf_array = []
        runs = 2
        batch_size = 2
        for run in range(runs):
            print("Run %d" % run)
            answers = []  # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                x = [ex.replace("\n\n\nQ: ", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 9)):
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
            for x in tqdm(chunks(inputs, 10)):
                # x = [ex.replace("Identify the physics formula that would be most useful for finding the answer to each of the following word problems.", "") for ex in x]
                x = [ex.replace("\n\n\nQ: ", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
                pdb.set_trace()
            preds = [get_answer(x.strip()) for x in answers]
            perf_array.append(substring_match(labels, preds))
            print(perf_array)
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / len(interpreter.execution_details)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


few_shot_human_prompt = """Input: Identify the physics formula that would be most useful for finding the answer to each of the following word problems. A proton is located at A, 1.0 m from a fixed +2.2 *10^−6 C charge. If the proton started from rest at A, what would be its speed at B?
  choice: U_E = k * q * q / r
  choice: v = λ * f
  choice: K = 1/2 * m * v ^ 2
  choice: dv = a * dt
Q1: [subquestion] What quantity is being asked for in "A proton is located at A, 1.0 m from a fixed +2.2 *10^−6 C charge. If the proton started from rest at A, what would be its speed at B?"
#1: The speed of the proton.
Q2: [subquestion] What other information is given in "A proton is located at A, 1.0 m from a fixed +2.2 *10^−6 C charge. If the proton started from rest at A, what would be its speed at B?"
#2: Distance (1.0m), charge(+2.2 *10^−6 C), initial speed (0)
Q3: [search] Given Distance, charge and initial speed, what can we compute for proton? Give the physics formula.
#3: Given the distance, charge, and initial speed of a proton, you can compute the total energy of the proton. From Eq. 2.3, the magnitude of the E field due to a point charge q at a distance r is given by. E = k|q| r2. Here we are given E and r, so we can solve for ...
Q4: [compare] What is the physics formula in the choices that is closest to E = k|q| r2.
#4: U_E = k * q * q / r
Q5: [EOQ]
Ans: U_E = k * q * q / r
----
Input: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.A particle of mass 6.8 * 10 ^ −5 g and charge 58 mC moves in a region of space where the electric field is uniform and is 7.2 N/C in the x direction and zero in the y and z direction. If the initial velocity of the particle is given by vy = 1.7 * 10 ^ 6 m/s, vx = vz = 0, what is the speed of the particle at 0.7 s?
  choice: F = m * a
  choice: v = dx / dt
  choice: E = q / (ε * A * cos(θ))
  choice: a = dv / dt
Q1: [subquestion] What quantity is being asked for in "A particle of mass 6.8 * 10 ^ −5 g and charge 58 mC moves in a region of space where the electric field is uniform and is 7.2 N/C in the x direction and zero in the y and z direction. If the initial velocity of the particle is given by vy = 1.7 * 10 ^ 6 m/s, vx = vz = 0, what is the speed of the particle at 0.7 s"
#1: speed of the particle
Q2: [subquestion] What other information is given in "A particle of mass 6.8 * 10 ^ −5 g and charge 58 mC moves in a region of space where the electric field is uniform and is 7.2 N/C in the x direction and zero in the y and z direction. If the initial velocity of the particle is given by vy = 1.7 * 10 ^ 6 m/s, vx = vz = 0, what is the speed of the particle at 0.7 s"
#2: mass of particle (6.8 * 10 ^ −5 g), charge(58 mC), electric field (7.2 N/C), initial velocity (vy = 1.7 * 10 ^ 6 m/s, vx = vz = 0)
Q3: [search] Given mass of the particle, charge and initial velocity of particle, and electric field, what can we compute for the particle? Give the physics formula.
#3:
We can compute acceleration of the particle, as follows:
a = (q/m) * E
This formula is derived from the Lorentz force law, which states that the force (F) on a particle of charge q in an electric field E is given by the equation:
F = ma = qE
Q4: [subquestion] What is the physics formula in the choices that is closest to F = ma = qE, a = (q/m) * E
#4: F = m * a
Q5: [EOQ]
Ans: F = m * a
----
Input: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.An electron passing between parallel plates 0.025 m apart experiences an upward electrostatic force of 5.1*10^−16 N. What is the potential difference between the plates?
  choice: E = q / (ε * A * cos(θ))
  choice: v = λ * f
  choice: E = q * σ / (2 * ε)
  choice: V = k * q / r
Q1: [subquestion] What quantity is being asked for in "An electron passing between parallel plates 0.025 m apart experiences an upward electrostatic force of 5.1*10^−16 N. What is the potential difference between the plates?"
#1: potential difference between the plates
Q2: [subquestion] What other information is given in "An electron passing between parallel plates 0.025 m apart experiences an upward electrostatic force of 5.1*10^−16 N. What is the potential difference between the plates?"
#2: distance between parallel plates(0.025 m), upward electrostatic force (5.1*10^−16 N)
Q3: [search] Given distance between plates and upward electrostatic force, what can we compute for the electron? Give the physics formula.
#3:
The electrostatic force can be computed using Coulomb's Law, which states that the force between two charges is equal to the product of the magnitude of the charges divided by the square of the distance between them, multiplied by a constant, k:
F = k * (q1 * q2) / r2
Q4: [subquestion] What is the physics formula in the choices that is closest to F = k * (q1 * q2) / r2
#4: V = k * q / r
Q5: [EOQ]
Ans: V = k * q / r
----
Input: Identify the physics formula that would be most useful for finding the answer to each of the following word problems.The 1205 kg physics dragster is traveling at 35 km/h east when it hits the gas and accelerates at 12.5 m/s^2 for 3.25 s. What is its change in momentum during this time?
  choice: W = F * d
  choice: v = λ * f
  choice: U_E = k * q * q / r
  choice: v = v_0 + a * t
Q1: [subquestion] What quantity is being asked for in "The 1205 kg physics dragster is traveling at 35 km/h east when it hits the gas and accelerates at 12.5 m/s^2 for 3.25 s. What is its change in momentum during this time?"
#1: change in momentum
Q2: [subquestion] What other information is given in "The 1205 kg physics dragster is traveling at 35 km/h east when it hits the gas and accelerates at 12.5 m/s^2 for 3.25 s. What is its change in momentum during this time?"
#2: mass (1205 kg), velocity (35 km/h east), acceleration (12.5 m/s^2), time (3.25s)
Q3: [search] Given mass, velocity, acceleration, time, what can we compute for the dragster? Give the physics formula.
#3:
With these values, we can compute the dragster's momentum using the formula: momentum = mass * velocity + (acceleration * time^2)/2 Solving for Final Velocity from Distance and Acceleration
A fourth useful equation can be obtained from another algebraic manipulation of previous equations. If we solve v = v0 + at  for t, we get ...
Q4: [subquestion] What is the physics formula in the choices that is closest to v = v0 + at
#4: v = v_0 + a * t
Q5: [EOQ]
Ans: v = v_0 + a * t
----
Description: %s
Input: %s
Q1: """


def human_intervention(
    temperature=0.3,
    model_name="text-davinci-002",
    strategy="fixed",
    self_consistency=False,
):
    global few_shot_cot_prompt

    few_shot_cot_prompt = few_shot_human_prompt
    interpreter = TopDownVisitorBeta(model_name=model_name, exclude_list=["[generate python code]"])

    def predict(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
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
                x = [ex.replace("\nEdited:", "") for ex in x]
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
        runs = 1
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(inputs, 10)):
                x = [ex.replace("\n\n\nQ: ", "") for ex in x]
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            preds = [get_answer(x) for x in answers]
            perf_array.append(substring_match(labels, preds))
            # Report on interpreter performance
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / (len(interpreter.execution_details) + 1e-6)
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
        print("Rate of affordance call", positive_rate)


# human_intervention(0.3, "text-davinci-002")

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
            "auto_cot_corrected",
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
    elif args.inference_strategy == "auto_cot_corrected":
        auto_cot(
            args.temperature,
            args.model_name,
            predict=True,
            use_corrected=True,
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
