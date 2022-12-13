import ast
import json
import pdb
import re
import time
from enum import auto
from re import L
from turtle import pd

import datasets
import numpy as np
import openai
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import (OpenAIModel, OpenAIModelLogProb, chunks, get_few_shot_prompt, get_subset, gpt3,
                   propose_decomposition, propose_instruction, substring_match,
                   parse_program)
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import jaccard_score

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TASKS = {"date_understanding" : {
"name": "Date Understanding", 
"description": "Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.",
"instances": [
"""Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
#1: 06/01/2021
Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: [arithmetic] What date is 24 hours later than today?  
#3: 05/01/2021
Q4: [EOQ]
Ans: 05/31/2021""", 
"""Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007 in MM/DD/YYYY 
#1: 01/01/2007
Q2: [arithmetic] What date is one week from 01/01/2007? 
#2: 01/08/2007
Q3: [EOQ]
Ans: 01/08/2007"""]
},
"language_games":{
    "name": "Language Games",
    "description": "Translate English into Pig Latin.",
    "instances": ["""Input: (English) Sami made his way across the bar and hugged Layla.
Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
#1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
Q4: [EOQ]
Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.""",
"""Input: (English) Tom is the most popular boy in school.
Q1: [string split] What are the words in "Tom is the most popular boy in school."?
#1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
Q4: [EOQ]
Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay."""]
},
"k_letter_concatenation":{
    "name": "K'th letter concatenation",
    "description": "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
    "instances": ["""Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
#1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["v", "e", "m", "t", "d"]
Q3: [string merge] Concatenate #2 with spaces
#3: "v e m t d"
Q4: [EOQ]
Ans: v e m t d""", 
"""Input: Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
#1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["r", "a", "i", "o", "u"]
Q3: [string merge] Concatenate #2 with spaces
#3: "r a i u o"
Q4: [EOQ]
Ans: r a i u o"""]
},
"known_unknown":{
    "name": "Known Unknown",
    "description": """Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".""",
    "instances": ["""Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
  choice: Unknown
  choice: Five million
Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon? 
#1:
Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin.
Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon.
Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: No. The question is too specific
Q3: [compare] What is the final answer?
#3: Unknown
Q4: [EOC]
Ans: Unknown""",
"""Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Q1: [search] Where was Mark Twain born?
#1:
Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
Q2: [check answer type] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: Yes. The answer is Florida, Missouri
Q3: [compare] What is the final answer?
#3: Florida, Missouri
Q4: [EOC]
Ans: Florida, Missouri"""]
},
"anachronisms":{
    "name": "Anachronisms",
    "description": "An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism? Answer Yes/No.",
    "instances": ["""Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
Q1: [tag] What are the entities in this sentence?
#1:
President George H. W. Bush
Gulf War
Q2: [search] When was President George H. W. Bush president?
#2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
Q3: [search] When was the Gulf War fought?
#3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
#4: Could these entities have co-existed based on thier time periods alone?
Yes. Their time periods intersect.
Q5: [generate output] Is this an anachronism?
#5: No
Q6: [EOC]
Ans: No""",
"""Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
Q1: [tag] What are the entities in this sentence?
#1:
Kurt Cobain
"Twin Peaks"
Q2: [search] When did television show "Twin Peaks" air?
#2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
Q3: [search] When did Kurt Cobain live?
#3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
Q4: Could these entities have co-existed based on this information?
No. Musician  Kurt Cobain could not have starred in Twin Peaks.
Q5: [generate output] Is this an anachronism?
#5: Yes
Q6: [EOC]
Ans: Yes"""]
},
"hindu_knowledge":{
    "name" : "Hindu Knowledge",
    "description": "Answer questions about Hindu mythology by choosing the option that best answers the question.",
    "instances": ["""Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [search] In the Hindu epic Ramayana, who is the main villain? 
#1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q2: [compare] Which option is the answer in #3 most similar to?
#2: Brahmastra
Q3: [EOC]
Ans: Brahmastra""",
"""Input: In the Hindu epic Ramayana, the main villain was a devotee of which deity?
  choice: Indra
  choice: Vishnu
  choice: Brahma
  choice: Shiva
Q1: [subquestion] Can this question be answered step-by-step?
#1: Yes.
Q2: [search] In the Hindu epic Ramayana, who is the main villain? 
#2: Ravana is the main antagonist of the Hindu Epic, the Ramayana. 
Q3: [search] Ravana was a devotee of which deity?
#3: Ravana, was an ardent devotee of Lord Shiva, is depicted and described as a great scholar,a brahman,a capable ruler and a maestro of the Veena.
Q4: [compare] Which option is the answer in #3 most similar to?
#4: Shiva
Q5: [EOC]
Ans: Shiva"""]
},
"auto_debugging":{
    "name": "Auto Debugging",
    "description": "Debug the following code snippets by finding the answer or the error message.",
    "instances": ["""Input: 
```
if x < 5:
	pass
```
What error does this program surface?
Q1: [execute] Execute the following python code snippet.
if x < 5:
	pass
#1:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Q2: [generate answer] What is the final error message?
#2: NameError: name 'x' is not defined
Q3: [EOC]
Ans: NameError: name 'x' is not defined""",
"""Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to get the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOC]
Ans: {1, 2, 3}"""]
},
"code_description":{
    "name": "Code Description",
    "description": "Given a python code snippet, choose the option that is the best description of the code snippet.",
    "instances": ["""Input:
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
#2:
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
Q6: [EOC]
Ans: prints sum of two input numbers only if they are integers otherwise raises error""",
"""Input:
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
Q6: [EOC]
Ans: prints a list of numbers greater than 7 from numbers_list"""]
},
"aqua_rat":{
    "name": "Arithmetic about Ratios",
    "description": "Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as needed.",
    "instances": ["""Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number?
    choice: 29 
    choice: 30 
    choice: 31 
    choice: 32 
    choice: None
Q1: [algebra] Let the greater part be x. What is the lesser part?
#1: 49-x
Q2: [algebra] What is the greater part if increased by 6
#2: x+6
Q2: [algebra] What is the lesser part if decreased by 11
#2: 49-x-11
Q3: [algebra] What is the ratio of greater to lesser after the change?
#3: (x+6):(49-x-11)
Q4: [algebra] Write down the equation to solve for x
#4: (x+6):(49-x-11) = 9:2
Q5: [solve] Solve for x: (x+6):(49-x-11) = 9:2
#5: x = 30
Q6: [compare] Which option is closes to this answer?
#6: 30
Q7: [EOC]
Ans: 30""",
"""Input: A salesman makes a 10%% commission on the selling price for each light switch he sells. If he sells 220 switches and the selling price of each switch is $6, what is his total commission?
    choice: 66
    choice: 88
    choice: 100
    choice: 120
    choice: 132
Q1: [arithmetic] What is the total selling price of all light bulbs.
#1: 220*6=1320
Q2: [arithmetic] What is 10%% of #1?
#2: 0.10*1320=132
Q3: [EOC]
Ans: 132
"""]
},
"gsm8k":{
    "name" : "Middle school arithmetic problems",
    "description": "Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.",
    "instances": ["""Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
Q1: [find inputs] What is total number of toys that need to be made?
#1: 400
Q2: [find inputs] How many workers in total?
#2: 5
Q3: [arithmetic] How many toys do 2 of the workers produce?
#3: 2 workers working for 10 hours making 6 toys per hour produce 2*10*6 = 120 toys.
Q4: [arithmetic] How many toys do another 2 of workers produce?
#4: 2 workers working for 10 hours making 4 toys per hour produce 2*10*4 = 80 toys.
Q5: [arithmetic] How many toys did the last worker make?
#5: Out of 400 toys, 120+80=200 toys were made by the first 4 workers. The 5th worker didn't finish the job, since 20 toys were still remaining to be made. So they made 400-200-20=180 toys.
Q6: [arithmetic] How many toys per hour does the fifth worker produce if he worked for 10 hours?
#6: The 5th worker made 180/10 = 18 toys per hour.
Q7: [EOC]
Ans: 18""",
"""Input: If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?
Q1: [arithmetic] What is the speed of the second train? 
#1: 60/2=30 miles an hour
Q2: [arithmetic] What is distance travelled by first train?
#2: 60*3=180 miles
Q3: [arithmetic] What is distance travelled by first train?
#3: 30*3=90 miles
Q4: [subquestion] Are the trains moving towards or away from each other?
#4: Away from each other.
Q5: [arithmetic] How far apart are the trains after 3 hours?
#5: 180+90=270 miles
Q6: [EOC]
Ans: 270 miles"""]
},
"elemetary_math_qa":{
    "name": "Elementary school arithmetic problems",
    "description": "What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as needed.",
    "instances": ["""Input: add 70 to 90, subtract 20 from result, subtract result from 200.
 choice:130
 choice:80
 choice:150
 choice:100
 choice:60
Q1: [arithmetic] add 70 to 90
#1: 70+90=160
Q2: [arithmetic] subtract 20 from 160
#2: 160-20=140
Q3: [arithmetic] subtract result 140 from 200
#3: 200-140=60
Q4: [compare] Which option does the final answer match?
#4: 60
Q5: [EOC]
Ans: 60""",
"""Input: machine r takes 2 more hours than machine b to make 20 widgets . if working together , the machines can make 25 widgets in 3 hours , how long will it take machine r to make 40 widgets ?
 choice:12
 choice:8
 choice:5
 choice:10
 choice:6
Q1: [algebra] Let the number of hours machine b takes to made 20 widgets be x. How many hours does machine r take?
#1: machine r takes x+2 hours to make 20 widgets.
Q2: [algebra] How many widgets does machine b make in 3 hours?
#2: (3*20)/(x)=60/x
Q2: [algebra] How many widgets does machine r make in 3 hours?
#2: (3*20)/(x+2)=60/(x+2)
Q2: [algebra] How many widgets do machine b and r make in 3 hours?
#2: 25
Q3: [algebra] Write down the equation to solve for x
#3: 60/x + 60/(x+2) = 25
Q4: [Solve] Solve for x: 60/x + 60/(x+2) = 25
#4: x = 4 
Q5: [arithmetic] How long does it take machine r to make 20 widgets?
#5: x+2=6 
Q6: [arithmetic] How long does it take machine r to make 40 widgets?
#6: 2*6=12 
Q5: [compare] Which option is closes to this answer?
#5: 12
Q6: [EOC]
Ans: 12"""]
},
"mawps":{
    "name": "Arithmetic word problems",
    "description": "Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.",
    "instances": ["""Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
Q1: [arithmetic] How much bread does Viola have if he had 167 loafs before and Nancy too 137 of them?
#1: 167-137=30
Q2: [EOC]
Ans: 30""",
"""Input: Rodney had 35 pear . He dice each pear into 13 slices . How many pear slices did Rodney make?
Q1: [arithmetic]  How many pear slices did Rodney make if he started with 35 pears and diced each into 13 slices?
#1: 35*13=455
Q2: [EOC]
Ans: 455"""]
},
"navigation":{
    "name": "Navigation",
    "description": """Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No".""",
    "instances": ["""Input: Take 7 steps. Turn right. Take 8 steps. Turn around.
Q1: [subquestion] Does this question require vector arithmetic?
#1: Yes.
Q2: [subquestion] Which way are you facing when you start? If unknown, assume you face forward?
#2: Face forward
Q3: [vector arithmetic] What is the distance moved forward?
#3: 7 steps
Q4: [vector arithmetic] What is the distance moved right?
#4: 8 steps
Q5: [vector arithmetic] What is the distance moved backward?
#5: 0 steps
Q6: [vector arithmetic] What is the distance moved left?
#6: 0 steps
Q7: [vector arithmetic] What is total distance moved from starting point?
#7: 7 steps vertically, 8 steps horizontally 
Q8: [subquestion] Is the total distance moved, both vertically and horizontally, 0? 
#8: No
Q9: [EOC]
Ans: No""",
"""Input: Always face forward. Take 6 steps right. Take 10 steps left. Take 3 steps left. Take 1 step forward. Take 4 steps left. Take 10 steps left.
Q1: [subquestion] Does this question require vector arithmetic?
#1: Yes.
Q2: [subquestion] Which way are you facing when you start? If unknown, assume you face forward?
#2: Face forward
Q3: [vector arithmetic] What is the distance moved forward?
#3: 1
Q4: [vector arithmetic] What is the distance moved right?
#4: 6
Q5: [vector arithmetic] What is the distance moved backward?
#5: 8
Q6: [vector arithmetic] What is the distance moved left?
#6: 3+10+4+10=27
Q7: [vector arithmetic] What is total distance moved from starting point?
#7: 8-1=7 steps vertically, 27-6=21 steps horizontally 
Q8: [subquestion] Is the total distance moved, both vertically and horizontally, 0? 
#8: No
Q9: [EOC]
Ans: No"""]
},
}



def random_tasks(N=5):
    random_tasks = np.random.choice(len(TASKS), N, replace=False)
    selected_tasks = [list(TASKS.keys())[t] for t in random_tasks]
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks:
        prompt += TASKS[task]["instances"][0] + "\n----\n"
    prompt += "Description: %s\nInput: %s\nQ1:"
    # create a prompt consisting of N random tasks
    return prompt

def similar_tasks(task_description, io_pairs, N=5):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts_to_compare = []
    for key, task in TASKS.items():
        print(key)
        doc = "\n".join(task["instances"][0].split("\n")[:2]) + "\n" + task["instances"][1].split("\n")[1]
        texts_to_compare.append(doc)
    doc_embeddings = model.encode(texts_to_compare) # (13, 384)
    query = "Description:" + task_description +  "\nInput: " + io_pairs[0][0] + "\nInput: " + io_pairs[1][0]
    query_embeddings = model.encode([query]) #(1, 384)
    similarity_matrix = util.pytorch_cos_sim(query_embeddings, doc_embeddings)
    # Choose top N 
    top_tasks = (-similarity_matrix.squeeze(0).numpy()).argsort()[:N]
    selected_tasks = [list(TASKS.keys())[t] for t in top_tasks]
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks:
        prompt += TASKS[task]["instances"][0] + "\n----\n"
    prompt += "Description: %s\nInput: %s\nQ1:"
    return prompt


def similar_auto_breakdowns(task_description, io_pairs, N=5):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.3, quote='---', n=1)
    texts_to_compare = []
    for key, task in TASKS.items():
        print(key)
        doc = "\n".join(task["instances"][0].split("\n")[:2]) + "\n" + task["instances"][1].split("\n")[1]
        doc = doc + "\nA: Let's think step-by-step.\n"
        doc = doc + gpt3(doc)[0]
        texts_to_compare.append(doc)
    doc_embeddings = model.encode(texts_to_compare) # (13, 384)
    
    query = "Description:" + task_description +  "\nInput: " + io_pairs[0][0] + "\nInput: " + io_pairs[1][0]
    query = query + "\nA: Let's think step-by-step.\n"
    query = query + gpt3(query)[0]

    query_embeddings = model.encode([query]) #(1, 384)
    similarity_matrix = util.pytorch_cos_sim(query_embeddings, doc_embeddings)
    # Choose top N 
    top_tasks = (-similarity_matrix.squeeze(0).numpy()).argsort()[:N]
    selected_tasks = [list(TASKS.keys())[t] for t in top_tasks]
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks:
        prompt += TASKS[task]["instances"][0] + "\n----\n"
    prompt += "Description: %s\nInput: %s\nQ1:"
    return prompt


task_similarity_prompt = """Give two tasks with their descriptions and examples of inputs and outputs for the tasks, determine if they are similar. Two tasks are similar if require common subtasks like string operations, web search, translation, arithmetic, code execution, etc.
----
Task1: [Date understanding] Find the required date in MM/DD/YYYY using information about related events and dates in the input. Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY? The final answer is 05/01/2021.
Task2: [Language Games] Translate English into Pig Latin. Input: English sentence is "Sami made his way across the bar and hugged Layla". The final answer is "Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay."
Are these similar? Yes. They both require answering in a spcific string format. 
----
Task1: [K'th letter concatenation] Take the letters at position 3 of the words in a list of words and concatenate them using a space. Input: What are the words in "Savita Saeed Ramos Sato Yadav"? The final answer is "v e m t d".
Task2: [Language Games] Translate English into Pig Latin. Input: English sentence is "Sami made his way across the bar and hugged Layla". The final answer is "Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay."
Are these similar? Yes. They both require accessing and manipulating characters in strings.
----
Task1: [K'th letter concatenation] Take the letters at position 3 of the words in a list of words and concatenate them using a space. Input: What are the words in "Savita Saeed Ramos Sato Yadav"? The final answer is "v e m t d".
Task2: [Known Unknown] Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". Input: How many hairs were on Neil Armstrong's head when he landed on the moon? The final answer is "Unknown".
Are these similar? No. Task 1 requires manipulating strings and Task 2 requires answering a question by possibly looking up information on the web. 
----
Task1: [Anachronisms] An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism? Input: Kurt Cobain starred in the 1980 television show "Twin Peaks". The final answer is "Yes".
Task2: [Known Unknown] Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown". Input: Where was Mark Twain born? The final answer is Florida, Missouri. 
Are these similar? Yes. They both require searching information about entities mentioned in the text, like Kurt Cobain or Mark Twain.
----
Task1: [Hindu Knowledge] Answer questions about Hindu mythology by choosing the option that best answers the question. Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon? Choices: Anjalikastra, Narayanastra, Agneyastra, Brahmastra. The final answer is Brahmastra.
Task2: [Code Debugging] Debug the following code snippets by finding the answer or the error message. Input: 
if x < 5:
	pass
The final answer is
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'x' is not defined
Are these similar? No. Task 1 is about asnswering a question and requires searching information about entities mentioned in the text. Task 2 is a question about debugging code and may require a Python interpreter.
Task 1: %s
Task 2: %s
Are these similar? 
"""

"""""
Reason: 
----
Task1: arithmetic 
Task2: code
Similar? Yes
Reason: 
"""


def llm_similar_tasks(task_name, task_description, io_pairs, N=5):
    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.5, quote='---', n=1)

    query = "[" + task_name + "] "  + task_description +  " Input: " + io_pairs[0][0] + "\nThe final answer is " + io_pairs[0][1]

    selected_tasks = []
    selection_reasons = []
    for key, task in TASKS.items():
        print(key)
        parsed_program = parse_program(task["instances"][0])
        doc = "[" + task["name"] + "] "  + task["description"] + " Input: " + parsed_program.input_node.text + parsed_program.answer_node.text.replace("Ans: ", "The final answer is ")
        filled_prompt = task_similarity_prompt % (query, doc)
        
        similar = gpt3(filled_prompt)[0]
        selection_reasons.append(similar)
        print(similar)
        if "yes" in similar.strip().lower():
            selected_tasks.append(key)
    
    
    if len(selected_tasks) >= N:
        random_tasks = np.random.choice(len(selected_tasks), N, replace=False)
        selected_tasks = [selected_tasks[t] for t in random_tasks]
    
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks:
        prompt += TASKS[task]["instances"][0] + "\n----\n"
    
    prompt += "Description: %s\nInput: %s\nQ1:"
    return prompt



def task_selection_accuracy(reference_tasks, task_name, task_description, io_pairs, N=3):

    #define Jaccard Similarity function
    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    gpt3 = OpenAIModel(model="text-davinci-002",  max_length=500, temperature=0.3, quote='---', n=1, logprobs=10)

    query = "[" + task_name + "] "  + task_description +  " Input: " + io_pairs[0][0] + "\nThe final answer is " + io_pairs[0][1]

    selected_tasks = []
    all_reasons = []
    selection_reasons = []
    for key, task in TASKS.items():
        parsed_program = parse_program(task["instances"][0])
        doc = "[" + task["name"] + "] "  + task["description"] + " Input: " + parsed_program.input_node.text + parsed_program.answer_node.text.replace("Ans: ", "The final answer is ")
        filled_prompt = task_similarity_prompt % (query, doc)
        
        similar = gpt3(filled_prompt)[0]
        all_reasons.append(similar)
        if "yes" in similar.strip().lower():
            selected_tasks.append(key)
            selection_reasons.append(similar)
    
    
    if len(selected_tasks) >= N:
        random_tasks = np.random.choice(len(selected_tasks), N, replace=False)
        selected_tasks = [selected_tasks[t] for t in random_tasks]
        selection_reasons = [selection_reasons[t] for t in random_tasks]

    # Select N tasks with highest log_prob assigned to "Yes"

    print([(task, reason) for task, reason in zip(selected_tasks, selection_reasons)])
    print("Jaccard:", jaccard(reference_tasks, selected_tasks))


    