import argparse
import ast
import json
import pdb
import re
import time
from re import L
from turtle import pd

import datasets
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, cache_dir, chunks, get_answer,
                   get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
import urllib.request
from collections import Counter

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
from prompt_library import (llm_similar_tasks, random_tasks,
                            similar_auto_breakdowns, similar_tasks)
from sequential_interpreter import TopDownVisitor, TopDownVisitorBeta

# Get data
d = datasets.load_dataset('bigbench', 'anachronisms')
inputs = d['validation']['inputs'] + d['train']['inputs']
labels = d['validation']['targets'] + d['train']['targets']
labels = [l[0] for l in labels]
# inputs = [x.split('\n')[0] for x in inputs]
# labels = np.array([int(x[0] == 'Yes') for x in d['train']['targets'] + d['validation']['targets']])

train_inputs = d['train']['inputs']
train_labels = d['train']['targets']

task_description = "Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."

io_pairs = [("President Syngman Rhee sent a letter commending Hugo Chavez's election victory.",
"Yes"),
("The recognition of Christianity as the official religion of both Ethiopia and the Roman Empire within the same decade is notable.",
"Yes"),
("President Woodrow Wilson rallied Americans to support the U.S. joining the International Atomic Energy Agency.",
"Yes"),
("The T. rex was running toward the herd of Wagyu cattle grazing outside.",
"Yes"),
("Sun Tzu dedicated an entire chapter to describing the failure of Babylon.",
"No"),
("Igor Stravinsky's favorite musical piece was the Symphonie Fantastique.",
"No"),
("The Hagia Sophia has seen drastic transformations to its interior from its inception, including becoming a church, a mosque, a museum, and back to a mosque again. ",
"No"),
("Ponce De Leon used a telegram to report his findings to the king.",
"Yes"),
("Jason connected his new TRS80 color computer to the TV and played Pyramid 2000.",
"No"),
("Richard III used LEDs to light his throne room.",
"Yes")]


def exact_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() == predict.lower():
            correct += 1
        count += 1
    return (1.0*correct)/count


def token_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in [p.lower() for p in predict]:
            correct += 1
        count += 1
    return (1.0*correct)/count


def few_shot(N=10, temperature=0.3, model_name="text-davinci-002"):

    few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=N)
    print(len(tokenizer(few_shot_prompt)['input_ids']))

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=200, temperature=temperature, quote='---', n=1)
        prompts = ["""%s\
%s""" % (few_shot_prompt, x) for x in chunk]
        return gpt3(prompts)
    
    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = [x.strip() for x in answers]
        perf_array.append(exact_match(labels, preds))
    print("No decomposition Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


# Human Decomp 
def anachronism():
    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=200, quote='---', n=1)
        prompts = ['''Given a sentence and the time periods of each entity in it, tell me if it could have happened or not.
    Sentence: I wrote about shakespeare
    Entities and dates:
    I -> 21st century
    Shakespeare -> 16th century
    Could the sentence be true based on the dates alone: Yes
    ----
    Sentence: Shakespeare wrote about me

    Entities and dates:
    Shakespeare -> 16th century
    I -> 21st century

    Could the sentence be true based on the dates alone: No
    ----
    Sentence: %s''' % x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 2
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers.extend(predict(x))
        preds = np.array([int(x.endswith('No')) for x in answers])
        perf_array.append((preds == labels).mean())
    print("Human Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))

few_shot_cot_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.  
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
  choice: Unknown
  choice: Five million
Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon? 
#1: 
Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldri...
Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon ...
Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: No. The question is too specific
Q3: [compare] What is the final answer?
Unknown
Q4: [EOQ]
Ans: Unknown
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
Q1: [tag] What are the entities in this sentence?
#1: 
President George H. W. Bush
Gulf War
Q2: [search] When was President George H. W. Bush president?
#2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
Q3: [search] When was the Gulf War fought?
#3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
Q4: [subquestion] Could these entities have co-existed and interacted based on this information?
Yes. Their time periods intersect.
Q5: [generate answer] Is this an anachronism?
#5: No
Q6: [EOQ]
Ans: No
----
Description: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
Q1: [tag] What are the entities in this sentence?
#1: 
Kurt Cobain
"Twin Peaks"
Q2: [search] When did television show "Twin Peaks" air?
#2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
Q3: [search] When did Kurt Cobain live?
#3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
Q4: [subquestion] Could these entities have co-existed and interacted based on this information?
No. Musician  Kurt Cobain could not have starred in Twin Peaks.
Q5: [generate answer] Is this an anachronism?
#5: Yes
Q6: [EOQ]
Ans: Yes
----
Description: Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [subquestion] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#1: No.
Q2: [search] In the Hindu epic Ramayana, who is the main villain? 
#2: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q3: [compare] Which option is the answer in #3 most similar to?
#3: Brahmastra
Q4: [EOQ]
Ans: Brahmastra
----
Description: Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". 
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Q1: [search] Where was Mark Twain born?
#1: 
Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: Yes. The answer is Florida, Missouri
Q3: [generate answer] What is the final answer?
Florida, Missouri
Q4: [EOQ]
Ans: Florida, Missouri
----
Desciption: %s 
Input: %s
Q1:"""


def few_shot_cot(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):

    global few_shot_cot_prompt
    task_name = "Anachronisms"
    task_description = "(Anachronisms) An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'yes' or 'no'."

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
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            answers.extend(predict("An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'yes' or 'no'", x))
        preds = [[y.strip() for y in x.split("\n")] for x in answers]
        perf_array.append(token_match(labels, preds))
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


auto_cot_corrected_prompt = """Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'.
President Syngman Rhee sent a letter commending Hugo Chavez's election victory.
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

First, we need to determine whether the sentence is talking about a person, thing, or event. In this case, it is talking about a person: President Syngman Rhee.

Next, we need to determine whether this person is in their proper time. President Syngman Rhee was a real person who lived from 1875-1965. The sentence mentions him sending a letter, so we need to determine if the time frame in which he is sending the letter is correct.

The sentence mentions Hugo Chavez's election victory. Hugo Chavez was a real person who lived from 1954-2013. Therefore, the time frame in which President Syngman Rhee is sending the letter is not correct, and the sentence contains an anachronism.

The final answer is "Yes".
----
Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'.
The recognition of Christianity as the official religion of both Ethiopia and the Roman Empire within the same decade is notable.
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

1. "The recognition of Christianity as the official religion of both Ethiopia and the Roman Empire within the same decade is notable."

2. Christianity was not recognized as the official religion of Ethiopia until the 4th century, and it was not recognized as the official religion of the Roman Empire until the Edict of Milan in 313.

3. Therefore, this sentence contains an anachronism.

The final answer is "Yes".
----
Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'.
President Woodrow Wilson rallied Americans to support the U.S. joining the International Atomic Energy Agency.
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

First, we need to determine whether the sentence contains a mistake in chronology. President Woodrow Wilson was in office from 1913-1921, so the sentence could be referring to an event that happened during his presidency. The International Atomic Energy Agency wasn't founded until 1957, so if the sentence is referring to an event that happened during Wilson's presidency, it would be a mistake in chronology.

Next, we need to determine whether the sentence contains a person, thing, or event that is out of its proper time. The International Atomic Energy Agency wasn't founded until 1957, so if the sentence is referring to the Agency, it would be out of its proper time.

Based on this information, we can conclude that the sentence does contain an anachronism.

The final answer is "Yes".
----

Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'.
Igor Stravinsky's favorite musical piece was the Symphonie Fantastique.
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

First, we need to determine what the proper time for Igor Stravinsky's favorite musical piece is. In other words, when was the Symphonie Fantastique composed?

The Symphonie Fantastique was composed in 1830.

Next, we need to determine if Igor Stravinsky was alive in 1830. Igor Stravinsky was born in 1882 and died in 1971.

Since it is possible for Igor Stravinsky to have liked music that was composed before his time, the sentence "Igor Stravinsky's favorite musical piece was the Symphonie Fantastique" is not anachronistic.

The final answer is "No".
----
Anachronisms: An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'.
Jason connected his new TRS80 color computer to the TV and played Pyramid 2000.
The final answer is either "Yes" or "No".
A: Let's think step-by-step.

The TRS80 color computer was released in 1980, so that's not an anachronism.

Pyramid 2000 was released in 1983, so that's not an anachronism either.

The sentence does not contain any anachronisms.

The final answer is "No"
----
"""


def auto_cot(temperature=0.3, model_name="text-davinci-002", predict=True, use_corrected=False, self_consistency=False):
    global auto_cot_corrected_prompt
    auto_cot_prompt = ""
    for io_pair in io_pairs[:10]:
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=0.7, quote='---', n=1)
        prompt = """%s\n"""% task_description + io_pair[0] + \
            """\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n""" 
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
        # Add the final answer with special format so evaluation is easier.

    if use_corrected:
        auto_cot_prompt = auto_cot_corrected_prompt
    
    print(auto_cot_prompt)
    f = open("auto_cot_demonstrations.txt","a+")
    f.write("Anachronisms\n\n")
    f.write(auto_cot_prompt)

    if not predict:
        return

    def predict_self_consistency(description, chunk, n=5):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=n)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    def predict(chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=500, temperature=temperature, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n"""%task_description + \
            """%s\nThe final answer is either "Yes" or "No".\nA: Let's think step-by-step.\n"""% (x) for x in chunk]
        return gpt3(prompts)

    if self_consistency:
        perf_array = []
        runs = 1
        batch_size = 2
        for run in range(runs): 
            print("Run %d"%run)
            answers = [] # List of counters
            for x in tqdm(chunks(inputs, batch_size)):
                answer_set = predict_self_consistency(task_description, x)
                result_counter = [Counter() for i in range(batch_size)]
                for chunk_no, ans_chunk in enumerate(chunks(answer_set, 5)):
                    preds = []
                    for x in ans_chunk:
                        if re.search("""The final answer is """, x):
                            preds.append(x[re.search("""The final answer is """, x).span(0)[-1]:])
                        else:
                            preds.append(x.strip())
                    for enum, pred in enumerate(ans_chunk):
                        # Only add to the counter if there is a legitimate answer
                        if re.search("""The final answer is """, pred):
                            result_counter[chunk_no].update([pred[re.search("""The final answer is """, x).span(0)[-1]:]])
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
            print("Run %d"%run)
            answers = []
            for x in tqdm(chunks(inputs, 20)):
                answers.extend(predict(x))
            preds = [x.strip() for x in answers]
            perf_array.append(substring_match(labels, preds))
        print("Auto-CoT Performance:")
        print("Mean", np.mean(perf_array))
        print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def search_query(answer):
        lines = answer.strip().split("\n")
        new_lines = []
        skip=False
        for line in lines:
            if "[search]" in line:
                query_no = re.search('[0-9]+', line.split()[0]).group(0)
                new_lines.append(line)
                query = line[re.search("\[search\]", line).span(0)[1]:].strip()
                search_snippet = search(query, top_k=1)
                new_lines.append("#%s: "%(query_no) + search_snippet)
                # skip a next line or add #2
                skip=True
            else:
                if skip:
                    skip=False
                    continue
                new_lines.append(line)
        return "\n".join(new_lines)


    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            answers = predict("An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'yes' or 'no'", x)
            
            # Replace all instances of search with actual search outputs.
            # affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [search_query(a) for a in answers]
            # Find the last search term and resume execution after that.
            last_questions = [re.findall("Q[0-9]+: \[search\]", a)[-1] for a in affordance_outputs]
            query_nos = [re.search('[0-9]+', question.split()[0]).group(0) for question in last_questions]
            next_questions = [re.search(r"Q%s: "%str(int(q) + 1), a) for q, a in zip(query_nos,affordance_outputs)]
            x = [ex + "\n" + a[:q.span(0)[1]] for ex, a, q in zip(x, affordance_outputs, next_questions)]
            pdb.set_trace()
            new_answers.extend(predict_with_affordance("An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'yes' or 'no'", x))
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed", self_consistency=False):
    
    global few_shot_cot_prompt
    task_name = "Anachronisms"
    task_description = "(Anachronisms) An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'yes' or 'no'."

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

    interpreter = TopDownVisitorBeta()

    def predict(description, chunk):
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 1
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_labels = ["Ans: " + label for label in labels]
        for x in tqdm(chunks(inputs, 10)):
            x = [ex.replace("\nDoes the preceding sentence contain non-contemporaneous (anachronistic) elements?", "") for ex in x]
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(new_labels, preds))
        print(perf_array)
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


if __name__ == "__main__":
    parser  = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["text-davinci-002", "text-davinci-003", "code-davinci-002", "code-cushman-001"], default="text-davinci-002")
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--inference_strategy", type=str, choices=["dummy", "few_shot", "auto_cot", "cot_rollout", "few_shot_cot", "nl_program"], default="few_shot")
    parser.add_argument("--num_train_examples", type=int, default=10)
    parser.add_argument("--num_dev_examples", type=int, default=len(inputs))
    parser.add_argument("--self_consistency", default=False, action='store_true')

    args = parser.parse_args()

    print("Dataset statistics")
    print(task_description)
    print("Training examples:", len(train_inputs))
    print("Dev examples:", len(inputs))

    inputs = inputs[:args.num_dev_examples]
    labels = labels[:args.num_dev_examples]

    if args.inference_strategy == "few_shot":
        few_shot_prompt = get_few_shot_prompt(train_inputs, train_labels, n=args.num_train_examples)
        print("Length of few-shot prompt", len(tokenizer(few_shot_prompt)['input_ids']))
        few_shot(args.num_train_examples, args.temperature, args.model_name)
    elif args.inference_strategy == "auto_cot":
        auto_cot(args.temperature, args.model_name, predict=True, use_corrected=True, self_consistency=False)
    elif args.inference_strategy == "few_shot_cot":
        few_shot_cot(args.temperature, args.model_name)
    elif args.inference_strategy == "nl_program":
        nl_program(args.temperature, args.model_name, self_consistency=args.self_consistency)