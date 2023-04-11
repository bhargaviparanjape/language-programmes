import argparse
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
    get_autocot_answer,
    get_few_shot_prompt,
    get_subset,
    propose_decomposition,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Startegies for unknown questions
# predictions about future times from present day
# Things that ARE HARD TO count or find an exact cnt for
# Things that ARE HARD TO measure or find an exact measure for
# Things that ARE HARD TO DATE or find an exact date for
# Facts that are not in public record because personal details are often unknown


# load data
d = datasets.load_dataset("bigbench", "known_unknowns", cache_dir=cache_dir)
inputs = d["train"]["inputs"] + d["validation"]["inputs"]
labels = d["train"]["targets"] + d["validation"]["targets"]
labels = [l[0] for l in labels]

train_inputs = d["train"]["inputs"]
train_labels = d["train"]["targets"]

# Instructions that Flipped learning comes up with (not known)
task_description = """Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown"."""

io_pairs = [
    (
        """Q: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde""",
        "A: Ronnie Ronalde",
    ),
    (
        """Q: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri""",
        "A: Florida, Missouri",
    ),
    (
        """Q: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat""",
        "A: Bowler hat",
    ),
    (
        """Q: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156""",
        "A: Unknown",
    ),
    (
        """Q: How many legs do ants have?
  choice: Unknown
  choice: 6""",
        "A: 6",
    ),
    (
        """Q: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours""",
        "A: Between 7 and 9 hours",
    ),
    (
        """Q: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories""",
        "A: Unknown",
    ),
    (
        """Q: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings""",
        "A: Unknown",
    ),
    (
        """Q: What will be the name of the great-great-granddaughter of Diana Ross?
  choice: Unknown
  choice: Hazel Ross""",
        "A: Unknown",
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
    few_shot_prompt = get_few_shot_prompt(inputs, [[d] for d in labels], n=N)
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


def known_unknown():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="---", n=1)
        prompts = [
            """Q: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
A:
Ronnie Ronalde
----
Q: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
A:
Florida, Missouri
----
Q: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
A:
Bowler hat
----
Q: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
A:
Unknown
----
Q: How many legs do ants have?
  choice: Unknown
  choice: 6
A:
6
----
Q: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours
A:
Between 7 and 9 hours
----
Q: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
A:
Unknown
----
Q: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings
A:
Unknown
----
Q: What will be the name of the great-great-granddaughter of Diana Ross?
  choice: Unknown
  choice: Hazel Ross
A:
Unknown
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


def automatic_decomposition():
    decomp_prompt = "The task is to answer the question or figure out if the answer is unknown. I want to break this task into individual steps."
    io_pairs = """Input: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
Output: Ronnie Ronalde
----
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Output: Florida, Missouri
----
Input: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
Output: Bowler hat
----
Input: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
Output: Unknown
----
Input: How many legs do ants have?
  choice: Unknown
  choice: 6
Output: 6
----
Input: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours
Output: Between 7 and 9 hours
----
Input: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
Output: Unknown
----
Input: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings
Output: Unknown
----
Input: What will be the name of the great-great-granddaughter of Diana Ross?
  choice: Unknown
  choice: Hazel Ross
Output: Unknown"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_list_reversal_fn(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])

        #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model=model_name, max_length=1000, quote="---", n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """The task is to answer the question or figure out if the answer is unknown. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise. Either provide the answer or say "Unknown"."""
                    % (decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(inputs, labels=labels, n=len(inputs))
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition", z)
        fn = get_list_reversal_fn(decomposition, batch_size=20)
        this_preds = fn(subset)
        #     pp = np.array([1 if 'contains an anachronism' in x.lower() else 0 for x in this_preds])
        pdb.set_trace()
        # pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        pp = [
            pred.strip().lower().endswith("unknown") or pred.strip().lower().endswith("known")
            for pred in this_preds
        ]
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
    print(accs)
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.std(accs))


# few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1: Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
# Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: What is the final answer?
# #3: Unknown
# Q4: [EOC]
# Unknown
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
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
# Q5: Is this an anachronism?
# #5: No
# Q6: [EOC]
# No
# ----
# Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism?
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
# Q5: Is this an anachronism?
# #5: Yes
# Q6: [EOC]
# Yes
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#   choice: Anjalikastra
#   choice: Narayanastra
#   choice: Agneyastra
#   choice: Brahmastra
# Q1: [search] In the Hindu epic Ramayana, who is the main villain?
# #1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q2: [compare] Which option is the answer in #3 most similar to?
# #2: Brahmastra
# Q3: [EOC]
# Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1: Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: What is the final answer?
# #3: Florida, Missouri
# Q4: [EOC]
# Florida, Missouri
# ----
# Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
# Input: In the Hindu epic Ramayana, the main villain was a devotee of which deity?
#   choice: Indra
#   choice: Vishnu
#   choice: Brahma
#   choice: Shiva
# Q1: [subquestion] Can this question be answered step-by-step?
# #1: Yes.
# Q2: [search] In the Hindu epic Ramayana, who is the main villain?
# #2: Ravana is the main antagonist of the Hindu Epic, the Ramayana.
# Q3: [search] Ravana was a devotee of which deity?
# #3: Ravana, was an ardent devotee of Lord Shiva, is depicted and described as a great scholar,a brahman,a capable ruler and a maestro of the Veena.
# Q4: [compare] Which option is the answer in #3 most similar to?
# #4: Shiva
# Q5: [EOC]
# Shiva
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_retrieval_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Known Unknown"
    task_description = """(Known Unknown) Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown"."""

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
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.4, quote="---", n=1)
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
            answers.extend(
                predict_complete(
                    """Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".""",
                    x,
                )
            )
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to identify what is being asked. In this case, the question is asking for a famous whistler.
Second, we need to look at the choices provided. We see that one of the choices is Ronnie Ronalde, so we can infer that he is a famous whistler.

The final answer is Ronnie Ronalde
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to find out what type of hat Charlie Chaplin wore. To do this, we can either search for pictures of him online, or look for information in a biography or other source.
The bowler hat and cane were Charlie Chaplin's trademarks and indispensable props to his famous Tramp guise.

The final answer is Bowler hat.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to find out how many people were living in Cheyenne, Wyoming during the last solar eclipse.
According to the US Census Bureau, the population of Cheyenne, Wyoming was 59,466 in 2010.
Next, we need to find out how many of those people were eating during the last solar eclipse.
We don't have any data on that.

The final answer is Unknown.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How many legs do ants have?
  choice: Unknown
  choice: 6
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to find out how many legs ants have. We can look this up in a reference book, or we can ask someone who knows.
From what we can tell, ants have six legs. Therefore, the answer to the question is six.

The final answer is six.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to establish what we know. We know that Tinyman Zhengel is a cat, and that he eats food every day. However, we don't know how much food he eats every day.
Next, we need to consider our options. We can either estimate how much food Tinyman Zhengel eats every day, or we can ask him directly.
If we ask Tinyman Zhengel directly, he will be able to tell us exactly how much food he eats every day. However, since we don't have the ability to communicate with Tinyman Zhengel, we will not be able to get an accurate answer.
Therefore, the best answer is "Unknown".

The final answer is Unknown.
----
"""

auto_cot_cleaned_prompt = """Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: Who is a famous whistler?
  choice: Unknown
  choice: Ronnie Ronalde
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to determine what a famous whistler is. A famous whistler is someone who is well-known for their ability to whistle.
Next, we need to identify who meets this criteria. There are many people who could be considered famous whistlers, but one name that stands out is Ronnie Ronalde.
Ronnie Ronalde is a famous whistler, so the final answer is Ronnie Ronalde.

The final answer is Ronnie Ronalde.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to identify what information is being asked for. In this case, the question is asking for the location where Mark Twain was born.
Next, we need to find where this information is located in the text. In this case, the text says that Mark Twain was born in Florida, Missouri.
Finally, we need to choose the correct answer from the options given. In this case, the correct answer is "Florida, Missouri".

The final answer is Florida, Missouri.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: What type of hat did Charlie Chaplin wear?
  choice: Unknown
  choice: Bowler hat
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to decide what type of hat Charlie Chaplin wore. Was it a bowler hat or something else?
From what we can tell, Charlie Chaplin wore a bowler hat.

The final answer is bowler hat.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How many people in Cheyenne, Wyoming were eating during the last solar eclipse?
  choice: Unknown
  choice: 156
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, what is a solar eclipse? A solar eclipse is an event that occurs when the moon passes between the sun and the earth, and the moon blocks the sun's light from reaching the earth.
Second, when was the last solar eclipse? The last solar eclipse was on August 21, 2017.
Third, how many people were in Cheyenne, Wyoming on August 21, 2017? According to the U.S. Census Bureau, the population of Cheyenne, Wyoming was 63,624 as of July 1, 2017.
Fourth, how many of those people were eating during the eclipse? This answer is unknown.

The final answer is unknown.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How many legs do ants have?
  choice: Unknown
  choice: 6
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, let's consider the question. It's asking how many legs ants have. We know that ants are insects, and we also know that insects have six legs. Therefore, the answer to the question is six.

The final answer is six.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How many hours of sleep a night is considered healthy for an adult?
  choice: Unknown
  choice: Between 7 and 9 hours
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

The question is asking how many hours of sleep a night is considered healthy for an adult.
The two options provided are "Unknown" and "Between 7 and 9 hours".
Since we don't know the answer, the final answer should be "Unknown".

The final answer is unknown.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: How much food does the cat Tinyman Zhengel eat every day?
  choice: Unknown
  choice: 250 calories
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

First, we need to find out how much food Tinyman Zhengel eats every day. To do this, we need to find a source that provides this information. Unfortunately, we were unable to find any sources that provided this information.
Next, we need to try to estimate how much food Tinyman Zhengel eats every day. We know that the average cat eats about 250 calories per day. However, we don't know if Tinyman Zhengel is an average cat. Therefore, we cannot say for sure how much food he eats every day.

The final answer is unknown.
----
Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown".
Q: Who was the first person to whistle?
  choice: Unknown
  choice: T.J. Cummings
The final answer is either the answer phrase or the word "Unknown".
A: Let's think step-by-step.

The first person to whistle was T.J. Cummings.

The final answer is T.J. Cummings.
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
    for io_pair in io_pairs:
        gpt3 = OpenAIModel(model=model_name, max_length=1000, temperature=0.7, quote="---", n=1)
        prompt = (
            """%s\n""" % task_description
            + io_pair[0]
            + """\nThe final answer is either the answer phrase or the word "Unknown".\nA: Let's think step-by-step.\n"""
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
    f.write("Anachronisms\n\n")
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
            + """%s\nThe final answer is either the answer phrase or the word "Unknown".\nA: Let's think step-by-step.\n"""
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
            + """%s\nThe final answer is either the answer phrase or the word "Unknown".\nA: Let's think step-by-step.\n"""
            % (x)
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
            # time.sleep(10)
        preds = [get_autocot_answer(x) for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Auto-CoT Performance:")
    print("Perf Array", perf_array)
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance(temperature=0.3):
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

    def search_query(answer):
        lines = answer.strip().split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if "[search]" in line:
                query_no = re.search("[0-9]+", line.split()[0]).group(0)
                new_lines.append(line)
                query = line[re.search("\[search\]", line).span(0)[1] :].strip()
                search_snippet = search(query, top_k=1)
                new_lines.append("#%s: " % (query_no) + search_snippet)
                # skip a next line or add #2
                skip = True
            else:
                if skip:
                    skip = False
                    # If the GPT-3 answer needs to be added as well, remove #[0-9]+ from the answer
                    # pdb.set_trace()
                    new_lines.append(line)
                    continue
                new_lines.append(line)
        return "\n".join(new_lines)

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(
            model=model_name, max_length=1000, temperature=temperature, quote="---", n=1
        )
        prompts = [few_shot_cot_prompt % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers = predict(
                "Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string 'Unknown'",
                x,
            )
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            # affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            affordance_outputs = [search_query("Q1:" + a) for a in answers]
            new_x = []
            for ex, a in zip(x, affordance_outputs):
                last_question = re.findall("Q[0-9]+: \[search\]", a)
                if len(last_question) > 0:
                    last_question = last_question[-1]
                else:
                    # No search attempted.
                    new_x.append(ex)
                    continue
                query_no = re.search("[0-9]+", last_question.split()[0]).group(0)
                q = re.search(r"Q%s: " % str(int(query_no) + 1), a)
                if q:
                    new_prompt = ex + "\n" + a[: q.span(0)[1]]
                    if len(tokenizer(new_prompt)["input_ids"]) + 1000 > 4097:
                        pdb.set_trace()
                        # Too much input.
                        new_x.append(ex)
                    else:
                        new_x.append(ex + "\n" + a[: q.span(0)[1]])
                else:
                    # No next question beyond the last search questions. So continue to generate.
                    new_x.append(ex + "\n" + a)
            new_answers.extend(
                predict_with_affordance(
                    "Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string 'Unknown'",
                    new_x,
                )
            )
        preds = [x.strip() for x in new_answers]
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
    task_name = "Known Unknown"
    task_description = """(Known Unknown) Answer the question by choosing one of the two options provided. If the answer can't be found, the final answer should be the string "Unknown"."""

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

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
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
