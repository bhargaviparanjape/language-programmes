import argparse
from collections import Counter
import os
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
    get_subset,
    propose_decomposition,
    substring_match,
)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Musique
data_files = {
    split: os.path.join(cache_dir, "musique", "data", "musique_full_v1.0_%s.jsonl" % split)
    for split in ["train", "dev"]
}
d = datasets.load_dataset("json", data_files=data_files)
dev_inputs = [ex["question"] for ex in d["dev"]][:500]
dev_labels = [ex["answer"] for ex in d["dev"]][:500]

train_inputs = ["Q: " + ex["question"] for ex in d["train"]][:500]
train_labels = [[ex["answer"]] for ex in d["train"]][:500]

io_pairs = [
    (
        "Question: When did the country where Beterverwagting is located become a member of caricom",
        "Answer: 1 August 1973",
    ),
    (
        "Question: When did the roof gardens above the district where George Palao is born, open to the public",
        "Answer: 1980s",
    ),
    (
        "Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun",
        "Answer: Ming general Qu Neng",
    ),
    (
        "Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located",
        "Answer: near Salem",
    ),
    (
        "Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6",
        "Answer: Hassan Gouled Aptidon",
    ),
    (
        "Question: What subject was studied in David Sassoon's birthplace",
        "Answer: Islamic mathematics",
    ),
    (
        "Question: Who is the spouse of the creator of The Neverwhere",
        "Answer: Amanda Palmer",
    ),
    (
        "Question: When did the torch arrive in the birthplace of Han Kum-Ok",
        "Answer: April 28",
    ),
    (
        "Question: Who sings Meet Me in Montana with the performer of I Only Wanted You",
        "Answer: Dan Seals",
    ),
    (
        "Question: When did the torch reach the mother country of Highs and Lows",
        "Answer: May 2",
    ),
]

task_name = "Multi-step question answering"
task_description = "(Multi-step question answering) Answer complex questions that require decompositions into sub-questions and composing intermediate answers."


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
            model=model_name, max_length=500, temperature=temperature, quote="---", n=1
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
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
            # time.sleep(10)
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(dev_labels, preds))
        print(perf_array)
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def musique():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
        prompts = [
            """Question: When did the country where Beterverwagting is located become a member of caricom?
Answer: 1 August 1973
----
Question: When did the roof gardens above the district where George Palao is born, open to the public?
Answer: 1980s
----
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Answer: Ming general Qu Neng
----
Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Answer: near Salem
----
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Answer: Hassan Gouled Aptidon
----
Question: What subject was studied in David Sassoon's birthplace?
Answer: Islamic mathematics
----
Question: Who is the spouse of the creator of The Neverwhere?
Answer: Amanda Palmer
----
Question: When did the torch arrive in the birthplace of Han Kum-Ok?
Answer: April 28
----
Question: Who sings Meet Me in Montana with the performer of I Only Wanted You?
Answer: Dan Seals
----
Question: When did the torch reach the mother country of Highs and Lows?
Answer: May 2
----
Question: %s
Answer: """
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(preds, dev_labels)])
        perf_array.append(pp.mean())
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


auto_cot_corrected_prompt = """Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: When did the country where Beterverwagting is located become a member of caricom
A: Let's think step-by-step.

1. What country Beterverwagting is located in.
A quick Google search tells us that it is located in Guyana.

2. When Guyana became a member of CARICOM.
According to this website, Guyana became a member of CARICOM in 1973. The Caribbean Community and Common Market (CARICOM) was established by the Treaty of Chaguaramas, which was signed by Barbados, Jamaica, Guyana and Trinidad & Tobago and came into effect on August 1, 1973.

The final answer is 1 August 1973.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: When did the roof gardens above the district where George Palao is born, open to the public
A: Let's think step-by-step.

1. When and where was George Palao born?
Born in Kensington, London on 4 October 1940, George Palao became part of an entire generation of Gibraltarians who were born away from the Rock after their parents were evacuated from Gibraltar along with most of the civilian population during World War II.

2. When did the Kensington roof gardens open to the public?
For 80 years, the Kensington Roof Gardens, or The Roof Gardens as it’s known today, has stood as a green oasis in the concrete jungle of London.
In 1981, the rooftop was leased by Virgin, which opened up large sections of the garden to any visitors wanting to marvel at the 100+ trees present.

The final answer is 1980s.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun
A: Let's think step-by-step.

1. What is the birthplace of Li Shuoxun?
Li Shuoxun was born in Sichuan, China.

2. Who was ordered to force a Tibetan assault into Sichuan?
Yet invasion went both ways; the Ming general Qu Neng, under the command of Lan Yu, was ordered to repel a Tibetan assault into Sichuan in 1390.

The final answer is Ming general Qu Neng.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6
A: Let's think step-by-step.

1. The country whose southwestern portion is crossed by National Highway 6
The RN-6 National Highway is a national highway in southwestern Djibouti. Therefore, the country in question is Djibouti.

2. Identify the first president of Djibouti.
This can be found using a simple Google search, which tells us that the first president of India was Hassan Gouled Aptidon.

The final answer is Hassan Gouled Aptidon.
----
"""

auto_cot_cleaned_prompt = """Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: When did the country where Beterverwagting is located become a member of caricom
A: Let's think step-by-step.

1) What is the country where Beterverwagting is located?
A: The country where Beterverwagting is located is Guyana.
2) When did Guyana become a member of caricom?
A: Guyana became a member of caricom in 1973.

The final answer is 1973.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: When did the roof gardens above the district where George Palao is born, open to the public
A: Let's think step-by-step.

First, we need to find out when the roof gardens were built. George Palao was born in 1892, so the roof gardens must have been built after that. A quick Google search reveals that the gardens were built in 1898.
Next, we need to find out when the gardens were open to the public. Again, a quick Google search reveals that the gardens were open to the public in 1900.
Therefore, the answer to the question is that the roof gardens were open to the public in 1900, two years after they were built.

The final answer is 1900.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun
A: Let's think step-by-step.

Who was ordered to force a Tibetan assault?
The Chinese government.
Who was the Chinese government ordered to force a Tibetan assault on?
The Tibetan people.
Why was the Chinese government ordered to force a Tibetan assault?
To quell a Tibetan uprising.
Where did the Tibetan uprising take place?
The birthplace of Li Shuoxun.

The final answer is The Chinese government.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located
A: Let's think step-by-step.

First, we need to identify the state in which the Washington University is located. According to the story, it is located in the city of A City Decides. A quick Google search tells us that A City Decides is located in the state of Missouri.
Now that we know the state, we can find out where the Merrimack River starts in Missouri. A quick search tells us that the Merrimack River starts in the town of Monticello.

The final answer is Monticello.
----
Multi-hop question answering: Answer complex questions that require decompositions into sub-questions and composing intermediate answers.
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6
A: Let's think step-by-step.

1) Which country has National Highway 6 crossing its southwestern portion?
2) Who was the first president of that country?
1) National Highway 6 crosses the southwestern portion of India.
2) The first president of India was Rajendra Prasad.

The final answer is Rajendra Prasad.
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
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(
            model="text-davinci-002", max_length=1000, temperature=0.7, quote="---", n=1
        )
        prompt = """%s\n""" % task_description + io_pair[0] + """\nA: Let's think step-by-step.\n"""
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
    f.write("Musique\n\n")
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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
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
            + """%s\nA: Let's think step-by-step.\n""" % (x)
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
            for x in tqdm(chunks(dev_inputs, batch_size)):
                x = [ex.replace("\nA:", "") for ex in x]
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
            perf_array.append(substring_match(dev_labels, preds))
        print("FS-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))
    else:
        perf_array = []
        runs = 5
        for run in range(runs):
            print("Run %d" % run)
            answers = []
            for x in tqdm(chunks(dev_inputs, 10)):
                answers.extend(predict(x))
                # time.sleep(10)
            preds = [get_autocot_answer(x) for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
            print(perf_array)
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def subquestion_decomposition():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
        prompts = [
            """Question: When did the country where Beterverwagting is located become a member of caricom?
Follow up question: Beterverwagting >> country
Intermediate answer: Guyana
Follow up question: when did Guyana became a member of caricom
Intermediate answer: 1 August 1973
So the final answer is 1 August 1973
----
Question: When did the roof gardens above the district where George Palao is born, open to the public?
Follow up question: George Palao >> place of birth
Intermediate answer: Kensington
Follow up question: when did the roof gardens above Kensington open to the public
Intermediate answer: 1980s
So the final answer is 1980s
----
Question: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Follow up question: Li Shuoxun >> place of birth
Intermediate answer: Sichuan
Follow up question: Who was ordered to force a Tibetan assault into Sichuan ?
Intermediate answer: Ming general Qu Neng
So the final answer is Ming general Qu Neng
----
Question: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Follow up question: Which place is A City Decides in?
Intermediate answer: St. Louis
Follow up question: Where is Washington University in St. Louis located?
Intermediate answer: Missouri
Follow up question: where does the merrimack river start in Missouri
Intermediate answer: near Salem
So the final answer is near Salem
----
Question: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Follow up question: National Highway 6 >> country
Intermediate answer: Djibouti
Follow up question: Who was the first president of Djibouti ?
Intermediate answer: Hassan Gouled Aptidon
So the final answer is Hassan Gouled Aptidon
----
Question: What subject was studied in David Sassoon's birthplace?
Follow up question: David Sassoon >> place of birth
Intermediate answer: Baghdad
Follow up question: What subject was studied in Baghdad ?
Intermediate answer: Islamic mathematics
So the final answer is Islamic mathematics
----
Question: Who is the spouse of the creator of The Neverwhere?
Follow up question: The Neverwhere was made by whom?
Intermediate answer: Neil Gaiman
Follow up question: Neil Gaiman >> spouse
Intermediate answer: Amanda Palmer
So the final answer is Amanda Palmer
----
Question: When did the torch arrive in the birthplace of Han Kum-Ok?
Follow up question: Han Kum-Ok >> place of birth
Intermediate answer: Pyongyang
Follow up question: When did the torch arrive in Pyongyang ?
Intermediate answer: April 28
So the final answer is April 28
----
Question: Who sings Meet Me in Montana with the performer of I Only Wanted You?
Follow up question: I Only Wanted You >> performer
Intermediate answer: Marie Osmond
Follow up question: who sings meet me in montana with Marie Osmond
Intermediate answer: Dan Seals
So the final answer is Dan Seals
----
Question: When did the torch reach the mother country of Highs and Lows?
Follow up question: What is the country Highs and Lows is from?
Intermediate answer: Hong Kong
Follow up question: When did the torch arrive in Hong Kong ?
Intermediate answer: May 2
So the final answer is May 2
----
Question: %s"""
            % x
            for x in chunk
        ]
        return gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(preds, dev_labels)])
        perf_array.append(pp.mean())
        # perf_array.append(exact_match(dev_labels, preds))
    print("Subquestion decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.mean(perf_array))


def automatic_decomposition():
    decomp_prompt = (
        "I want to break down the task of answering complex questions into individual steps."
    )
    io_pairs = """Input: When did the country where Beterverwagting is located become a member of caricom?
Output: 1 August 1973
----
Input: When did the roof gardens above the district where George Palao is born, open to the public?
Output: 1980s
----
Input: Who was ordered to force a Tibetan assault into the birthplace of Li Shuoxun?
Output: Ming general Qu Neng
----
Input: Where does the Merrimack River start in the state where the Washington University in the city where A City Decides takes place is located?
Output: near Salem
----
Input: Who was the first president of the country whose southwestern portion is crossed by National Highway 6?
Output: Hassan Gouled Aptidon"""
    decompositions = propose_decomposition(decomp_prompt, io_pairs, 10)

    for decomp in decompositions:
        print(decomp)
        print("----")

    def get_musique_fn(decomposition, batch_size=10):
        decomposition = "1." + decomposition
        last_n = int(re.findall(r"(\d+)\.", decomposition)[-1])

        #     decomposition += '\n%s. Output YES if there is an anachronism, and NO otherwise' % (last_n + 1)
        def decomposition_fn(sentences):
            gpt3 = OpenAIModel(model="text-davinci-002", max_length=1000, quote="---", n=1)
            out = []
            for chunk in chunks(sentences, batch_size):
                prompts = [
                    """Answer the following complex questions. Using the following steps will help.
Steps:
%s
----
%s
How did you arrived at this answer step-wise."""
                    % (decomposition, x)
                    for x in chunk
                ]
                out.extend(gpt3(prompts))
            return out

        return decomposition_fn

    labs, subset = get_subset(dev_inputs, labels=dev_labels, n=100)
    preds = []
    pps = []
    accs = []
    for z, decomposition in enumerate(decompositions):
        print("Decomposition", z)
        fn = get_musique_fn(decomposition, batch_size=20)
        this_preds = fn(subset)
        #     pp = np.array([1 if 'contains an anachronism' in x.lower() else 0 for x in this_preds])
        pp = np.array([1 if ref.lower() in x.lower() else 0 for x, ref in zip(this_preds, labs)])
        preds.append(this_preds)
        pps.append(pp)
        # accs.append(exact_match(labs, pp))
        accs.append(pp.mean())
    print(accs)
    print("Automatic decomposition Performance:")
    print("Mean", np.mean(accs))
    print("Std. Dev", np.mean(accs))


# few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
#   choice: Unknown
#   choice: Five million
# Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
# #1:
# Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
# Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
# Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: No. The question is too specific
# Q3: [compare] What is the final answer?
# Unknown
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
# #4: [subquestion] Could these entities have co-existed and interacted based on this information?
# Yes. Their time periods intersect.
# Q5: [generate answer] Is this an anachronism?
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
# Q4: [subquestion] Could these entities have co-existed and interacted based on this information?
# No. Musician  Kurt Cobain could not have starred in Twin Peaks.
# Q5: [generate answer] Is this an anachronism?
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
# Q1: [subquestion] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
# #1: No.
# Q2: [search] In the Hindu epic Ramayana, who is the main villain?
# #2: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
# Q3: [compare] Which option is the answer in #3 most similar to?
# #3: Brahmastra
# Q4: [EOQ]
# Ans: Brahmastra
# ----
# Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
# Input: Where was Mark Twain born?
#   choice: Unknown
#   choice: Florida, Missouri
# Q1: [search] Where was Mark Twain born?
# #1:
# Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
# Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
# #2: Yes. The answer is Florida, Missouri
# Q3: [generate answer] What is the final answer?
# Florida, Missouri
# Q4: [EOQ]
# Ans: Florida, Missouri
# ----
# Desciption: %s
# Input: %s
# Q1:"""

few_shot_cot_prompt = few_shot_retrieval_prompt


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "Multi-hop questions"
    task_description = "(Multi-hop questions) Answer complex questions that require decompositions into sub-questions and composing intermediate answers."

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
        for x in tqdm(chunks(dev_inputs, 10)):
            answers.extend(
                predict_complete(
                    "Answer complex questions that require decompositions into sub-questions and composing intermediate answers.",
                    x,
                )
            )
            # time.sleep(10)
        preds = [get_answer(x) for x in answers]
        perf_array.append(substring_match(dev_labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
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
                if search_snippet != "":
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
            model="text-davinci-002",
            max_length=1000,
            temperature=temperature,
            quote="---",
            n=1,
        )
        prompts = [few_shot_cot_prompt_short % (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs):
        print("Run %d" % run)
        answers = []
        new_answers = []
        running_out_cnt = 0
        for x in tqdm(chunks(dev_inputs, 10)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers = predict(
                "Answer complex questions that require decompositions into sub-questions and composing intermediate answers.",
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
                    if (
                        len(tokenizer(few_shot_cot_prompt_short + new_prompt)["input_ids"]) + 1000
                        > 3900
                    ):
                        # pdb.set_trace()
                        # Too much input.
                        running_out_cnt += 1
                        new_x.append(ex)
                    else:
                        new_x.append(ex + "\n" + a[: q.span(0)[1]])
                else:
                    # No next question beyond the last search questions. So continue to generate.
                    new_x.append(ex + "\n" + a)
            new_answers.extend(
                predict_with_affordance(
                    "Answer complex questions that require decompositions into sub-questions and composing intermediate answers.",
                    new_x,
                )
            )
        preds = [x.strip() for x in new_answers]
        perf_array.append(substring_match(dev_labels, preds))
        print(perf_array)
        print("Running out", running_out_cnt, len(dev_inputs))
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
    task_name = "Multi-hop question answering."
    task_description = "Answer the following complex multi-hop questions that require answering several intermediate questions."

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
            for x in tqdm(chunks(dev_inputs, batch_size)):
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
            preds = [x.most_common(1)[0][0] for x in answers[: len(dev_inputs)]]
            perf_array.append(substring_match(dev_labels, preds))
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
            for x in tqdm(chunks(dev_inputs, 10)):
                x = [ex.replace("\nA:", "") for ex in x]
                prompts, answer = predict(task_description, x)
                new_answer = interpreter.batch_visit(prompts, answer)
                answers.extend(new_answer)
            pdb.set_trace()
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(dev_labels, preds))
            positive_calls = [
                int(len(stack_trace_list) >= 1)
                for stack_trace_list in interpreter.execution_details
            ]
            positive_rate = sum(positive_calls) / len(interpreter.execution_details)
            print(perf_array)
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

    dev_inputs = dev_inputs[: args.num_dev_examples]
    dev_labels = dev_labels[: args.num_dev_examples]

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
