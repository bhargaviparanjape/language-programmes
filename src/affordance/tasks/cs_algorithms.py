from re import L
from turtle import pd
from utils import gpt3, propose_decomposition, propose_instruction, chunks, get_subset, OpenAIModel, cache_dir, substring_match
from prompt_library import random_tasks, similar_tasks, llm_similar_tasks, similar_auto_breakdowns
from sequential_interpreter import TopDownVisitorBeta, TopDownVisitor
import datasets
import numpy as np
from tqdm import tqdm
import json, pdb
import re
import time

task_description = """(CS Algorithms) Solve the following simple programming tasks using Python."""


io_pairs=[("""Given two strings, determine the length of the longest common subsequence.

Strings: VIRVRHRSTQBLLSYPZDVYCFPSQRXNA SPLYVHLWMLDJVYMQTOZMVOJF
Length of longest common subsequence:""",
"8"),
("""Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:""",
"0"),
("""Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:""",
"4"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: } { ( [
Valid/Invalid?""",
"Invalid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: [ [ [ { [ [ ] { { } ( ) } [ ] ] } ] ] ]
Valid/Invalid?""",
"Valid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?""",
"Valid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?""",
"Invalid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?""",
"Valid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?""",
"Invalid"),
("""Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ] { } { [ ] [ ] } [ ] ( { ( ) } )
Valid/Invalid?""",
"Valid")]

d = datasets.load_dataset('bigbench', 'cs_algorithms', cache_dir=cache_dir)
inputs = d['validation']['inputs']
# inputs = [x.split('\n')[0] for x in inputs]
labels = d['validation']['targets']
labels = [l[0] for l in labels]

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

def cs_algorithms():
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=200, quote='---', n=1)
        prompts = ["""Given two strings, determine the length of the longest common subsequence.

Strings: VIRVRHRSTQBLLSYPZDVYCFPSQRXNA SPLYVHLWMLDJVYMQTOZMVOJF
Length of longest common subsequence:
8
----
Given two strings, determine the length of the longest common subsequence.

Strings: SCZFZGCCQQLB OJDXI
Length of longest common subsequence:
0
----
Given two strings, determine the length of the longest common subsequence.

Strings: RLXEHVGPC LDOOBAOCQPRJKZWOKUPPEHEAZIZPLSB
Length of longest common subsequence:
4
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: } { ( [
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ [ [ { [ [ ] { { } ( ) } [ ] ] } ] ] ]
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ]
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: ) } { [ ) } [ } { )
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: ( ) [ ( ) ] ( { } )
Valid/Invalid?
Valid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ {
Valid/Invalid?
Invalid
----
Determine whether the given sequence of parentheses is properly matched.

Sequence: [ { } ] { } { [ ] [ ] } [ ] ( { ( ) } )
Valid/Invalid?
Valid
----
%s
""" % x for x in chunk]
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




few_shot_cot_prompt="""In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used. 
Description: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
Input: 
```
if x < 5:
	pass
```
What error does this program surface?
Q1: [code execute] Execute the following python code snippet.
if x < 5:
	pass
#1:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Q2: [get error message] What is the final error message?
#2: NameError: name 'x' is not defined
Q3: [EOQ]
Ans: NameError: name 'x' is not defined
----
Desciption: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to get the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [code execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOQ]
Ans: {1, 2, 3}
----
Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet. 
Input:
Python code:
try:
    n = int(input())
    m = int(input())
    integer_sum = int(n) + int(m)
    print(integer_sum)
except:
    print('error')

  choice: prints number between 5 and 6
  choice: try input and except error
  choice: inputs the string 'try'
  choice: prints sum of two input numbers only if they are integers otherwise raises error
Q1: [code generate] prints number between 5 and 6
#1:
import random

print(random.uniform(5,6))
Q2: [code generate] try input and except error
#2
try:
    #open file
    file = open(file_name, "r")
    #read file
    data = file.read()
    #close file
    file.close()
    #split data
    data = data.split("\n")
    #remove empty string
Q3: [code generate] inputs the string 'try'
#3: print('try')
Q4: [code generate] prints sum of two input numbers only if they are integers otherwise raises error
#4:
#!/usr/bin/python

a=raw_input("enter first number: ")
b=raw_input("enter second number: ")
try:
    sum=int(a)+int(b)
    print "sum is: ",sum
except:
    print "enter integer values only"
Q5: [compare] Which of the generated code snippets are most like the original one?
#5: prints sum of two input numbers only if they are integers otherwise raises error
Q6: [EOQ]
Ans: prints sum of two input numbers only if they are integers otherwise raises error
----
Description: (Code Description) Which of the following choices best describes the functionality of the given python code snippet. 
Input:
Python code:
numbers_list = [2, 6, 8, 10, 11, 4, 12, 7, 13, 17, 0, 3, 21]
filtered_list = list(filter(lambda num: (num > 7), numbers_list))
print(filtered_list)

  choice: prints lambda
  choice: returns a filtered list
  choice: prints a list of numbers greater than 7 from numbers_list
  choice: prints numbers from 2 to 6
Q1: [code generate] prints lambda
#1:
print("lambda")
Q2: [code generate] returns a filtered list
#2
def filter_list(l):
  'return a new list with the strings filtered out'
  return [x for x in l if not isinstance(x, str)]
Q3: [code generate] prints a list of numbers greater than 7 from numbers_list
#3:
numbers_list = [1,2,3,4,5,6,7,8,9,10]
for number in numbers_list:
    if number > 7:
        print(number)
Q4: [code generate] prints numbers from 2 to 6
#4:
for i in range(2,7):
    print(i)
Q5: Which of the generated code snippets are most like the original one?
#5: prints a list of numbers greater than 7 from numbers_list
Q6: [EOQ]
Ans: prints a list of numbers greater than 7 from numbers_list
----
Desciption: %s
Input: %s
Q1:"""




def auto_cot():
    auto_cot_prompt = ""
    description = "CS Algorithms: Solve the following simple programming tasks."
    for io_pair in io_pairs[:5]:
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompt = """%s\n"""%description + io_pair[0] + \
            """\nA: Let's think step-by-step.\n"""
        auto_cot_prompt += prompt
        cot = gpt3(prompt)
        auto_cot_prompt += cot[0] + "\n----\n"
    print(auto_cot_prompt)
    def predict(chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.2, quote='---', n=1)
        prompts=[auto_cot_prompt + """%s\n""" %description + \
            """%s\nA: Let's think step-by-step.\n"""%x for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(x))
            pdb.set_trace()
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("Auto-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))


def affordance():
    def predict(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    def string_index(sequence, position):
        char_list = []
        for word in sequence:
            character  = word[position]
            char_list.append(character)
        return char_list

    def predict_with_affordance(description, chunk):
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=0.4, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        new_answers = []
        for x in tqdm(chunks(inputs, 20)):
            answers = predict("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x)
            pdb.set_trace()
            affordance_inputs = [json.loads(a.strip().split("\n")[1].replace("#1: ", "")) for a in answers]
            affordance_outputs = [string_index(inp, 2) for inp in affordance_inputs]
            x = [ex + a[:re.search("#2: ", a).span(0)[1]] + json.dumps(o) for ex, a, o in zip(x, answers, affordance_outputs)]
            new_answers.extend(predict_with_affordance("Take the letters at position 3 of the words in a list of words and concatenate them using a space.", x))
        preds = [[y.strip() for y in x.split("\n")] for x in new_answers]
        perf_array.append(token_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
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
        gpt3 = OpenAIModel(model="text-davinci-002",  max_length=2048, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 20)):
            # x = [ex.replace("\nA:", "") for ex in x]
            answers.extend(predict(task_description, x))
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
        print(perf_array)
    print("Few-shot COT performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))



def nl_program(temperature=0.3, model_name="text-davinci-002", strategy="fixed"):
    global few_shot_cot_prompt
    task_name = "CS Algorithms"
    task_description = "(CS Algorithms) Solve the following simple programming tasks using Python, executing the code to get the final answer."

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
        gpt3 = OpenAIModel(model=model_name,  max_length=1000, temperature=temperature, quote='---', n=1)
        prompts=[few_shot_cot_prompt% (description, x) for x in chunk]
        return prompts, gpt3(prompts)

    perf_array = []
    runs = 5
    for run in range(runs): 
        print("Run %d"%run)
        answers = []
        for x in tqdm(chunks(inputs, 10)):
            prompts, answer = predict(task_description, x)
            new_answer  = interpreter.batch_visit(prompts, answer)
            answers.extend(new_answer)
            time.sleep(60)
        preds = [x.strip() for x in answers]
        perf_array.append(substring_match(labels, preds))
    print("FS-CoT Performance:")
    print("Mean", np.mean(perf_array))
    print("Std. Dev", np.std(perf_array))




# auto_decomp(10, 0.3)
# affordance(temperature=0.3)
# dynamic_few_shot_cot(temperature=0.3, strategy="random")
# few_shot_cot()
# few_shot(N=5, temperature=0.3)
# auto_cot()
nl_program(temperature=0.3)