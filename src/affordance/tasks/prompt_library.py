import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2Tokenizer
from utils import OpenAIModel, OpenAIModelLogProb, parse_program

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

TASKS = {
    "date_understanding": {
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
Ans: 01/08/2007""",
        ],
    },
    "language_games": {
        "name": "Language Games",
        "description": "Translate English into Pig Latin.",
        "instances": [
            """Input: (English) Sami made his way across the bar and hugged Layla.
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
Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.""",
        ],
    },
    "k_letter_concatenation": {
        "name": "K'th letter concatenation",
        "description": "Take the letters at position 3 of the words in a list of words and concatenate them using a space.",
        "instances": [
            """Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
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
Ans: r a i u o""",
        ],
    },
    "known_unknown": {
        "name": "Known Unknown",
        "description": """Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".""",
        "instances": [
            """Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
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
Q4: [EOQ]
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
Q4: [EOQ]
Ans: Florida, Missouri""",
        ],
    },
    "anachronisms": {
        "name": "Anachronisms",
        "description": "An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism? Answer Yes/No.",
        "instances": [
            """Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
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
Q6: [EOQ]
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
Q6: [EOQ]
Ans: Yes""",
        ],
    },
    "hindu_knowledge": {
        "name": "Hindu Knowledge",
        "description": "Answer questions about Hindu mythology by choosing the option that best answers the question.",
        "instances": [
            """Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [search] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q2: [compare] Which option is the answer in #3 most similar to?
#2: Brahmastra
Q3: [EOQ]
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
Q5: [EOQ]
Ans: Shiva""",
        ],
    },
    "auto_debugging": {
        "name": "Auto Debugging",
        "description": "Debug the following code snippets by finding the answer or the error message.",
        "instances": [
            """Input:
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
Q3: [EOQ]
Ans: NameError: name 'x' is not defined""",
            """Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to print the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOQ]
Ans: {1, 2, 3}""",
        ],
    },
    "code_description": {
        "name": "Code Description",
        "description": "Given a python code snippet, choose the option that is the best description of the code snippet.",
        "instances": [
            """Input:
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
Q6: [EOQ]
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
Q6: [EOQ]
Ans: prints a list of numbers greater than 7 from numbers_list""",
        ],
    },
    "aqua_rat": {
        "name": "Arithmetic about Ratios",
        "prompt_prefix": """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.
""",
        "description": "Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.",
        "instances": [
            """Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number? A) 29 B) 30 C) 31 D) 32 E) None
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
num1 = Symbol('num1', positive=True)
num2 = 49 - num1
num1_after_change = num1 + 6
num2_after_change = num2 - 11
solution = solve_it(num1_after_change / num2_after_change - 9 / 2, num1)
ans = solution[num1]
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 30
Q3: [compare] Which of the options among A) 29 B) 30 C) 31 D) 32 E) None is most similar to the answer?
#3: B
Q4: [EOQ]
Ans: B""",
            """Input:A salesman makes a 10%% commission on the selling price for each light switch he sells. If he sells 220 switches and the selling price of each switch is $6, what is his total commission? A) 66 B) 88 C) 100 D) 120 E) 132
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
commission_percentage = 10
number_of_switches_sold = 220
selling_price_of_each_switch = 6
total_commission = ( commission_percentage / 100 ) * number_of_switches_sold * selling_price_of_each_switch
ans = total_commission
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 132
Q3: [compare] Which of the options among A) 66 B) 88 C) 100 D) 120 E) 132 is most similar to the answer?
#3: E
Q4: [EOQ]
Ans: E""",
        ],
    },
    "gsm8k": {
        "name": "Middle school arithmetic problems",
        "description": "Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.",
        "instances": [
            """Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_toys_ordered = 400
num_workers = 5
toys_produced_per_hour_by_worker1 = 6
toys_produced_per_hour_by_worker2 = 6
toys_produced_per_hour_by_worker3 = 4
toys_produced_per_hour_by_worker4 = 4
toys_produced_per_hour_by_worker5 = Symbol('toys_produced_per_hour_by_worker5', positive=True)
hours_worked = 10
toys_produced = num_toys_ordered-20
toys_produced_by_all_workers = ( toys_produced_per_hour_by_worker1 + toys_produced_per_hour_by_worker2 + toys_produced_per_hour_by_worker3 + toys_produced_per_hour_by_worker4 + toys_produced_per_hour_by_worker5) * hours_worked
solution = solve_it(toys_produced_by_all_workers - toys_produced, toys_produced_per_hour_by_worker5)
ans = solution[toys_produced_per_hour_by_worker5]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [EOQ]
Ans: 18""",
            """Input: If two trains depart from a station in opposite directions, and one train is traveling 60 miles an hour while the other is traveling half that distance per hour, how far apart are they from each other after 3 hours?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
speed_of_first_train = 60
speed_of_second_train = 30
distance_apart = speed_of_first_train * 3 + speed_of_second_train * 3
ans = distance_apart
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 270
Q3: [add unit] Add the appropriate unit to the final answer.
#3: 270 miles
Q4: [EOQ]
Ans: 270 miles""",
        ],
    },
    "elemetary_math_qa": {
        "name": "Elementary school arithmetic problems",
        "description": "What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.",
        "instances": [
            """Input: add 70 to 90, subtract 20 from result, subtract result from 200.
 choice:130
 choice:80
 choice:150
 choice:100
 choice:60
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
70 + 90 = 160
160 - 20 = 140
200 - 140 = 60
ans = 60
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 60
Q3: [EOQ]
Ans: 60""",
            """Input: machine r takes 2 more hours than machine b to make 20 widgets . if working together , the machines can make 25 widgets in 3 hours , how long will it take machine r to make 40 widgets ?
 choice:12
 choice:8
 choice:5
 choice:10
 choice:6
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
time_to_make_20_widgets_by_machine_b = Symbol('time_to_make_20_widgets_by_machine_b', positive=True)
time_to_make_20_widgets_by_machine_r = time_to_make_20_widgets_by_machine_b + 2
num_widgets_produced_by_b = 20 / time_to_make_20_widgets_by_machine_b
num_widgets_produced_by_r = 20 / time_to_make_20_widgets_by_machine_r
sum_widgets = num_widgets_produced_by_b + num_widgets_produced_by_r
time_to_make_25_widgets_together = 3
num_widgets_produced_together = 25 / time_to_make_25_widgets_together
ans = solve_it(sum_widgets - num_widgets_produced_together, time_to_make_20_widgets_by_machine_b)
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: {time_to_make_20_widgets_by_machine_b: 4.00000000000000}
Q3: [generate python code] How long will it take machine r to make 40 widgets. Write python code to solve for this.
#3:
time_to_make_20_widgets_by_machine_b = 4.0
time_to_make_20_widgets_by_machine_r = time_to_make_20_widgets_by_machine_b + 2
time_to_make_40_widgets_by_machine_r = time_to_make_20_widgets_by_machine_r*2
ans = time_to_make_40_widgets_by_machine_r
print(ans)
Q4: [code execute] Execute the python code and get the value of "ans"
#4: 12
Q5: [EOQ]
Ans: 12""",
        ],
    },
    "mawps": {
        "name": "Arithmetic word problems",
        "description": "Solve the following arithmetic word problems, writing out intermediate arithmetic calculations arithmetic calculations as python code. Store your result as a variable named 'ans'.",
        "instances": [
            """Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
viola_bought = 167
nancy_took = 137
ans = viola_bought - nancy_took
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 30
Q3: [EOQ]
Ans: 30""",
            """Input: Rodney had 35 pear . He dice each pear into 13 slices . How many pear slices did Rodney make?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
pear = 35
slices = 13
ans = pear * slices
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 455
Q3: [EOQ]
Ans: 455""",
        ],
    },
    "svamp": {
        "name": "Math word problems",
        "description": """Solve the following arithmetic word problems, writing out intermediate arithmetic calculations arithmetic calculations as python code. Store your result as a variable named 'ans'.""",
        "instances": [
            """Input: James bought 93 red and 10 blue stickers, he used 31 red sticker on his fridge and 7 blue stickers on his laptop. How many red stickers does James have?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
original_red_stickers = 93
used_red_stickers = 31
ans = original_red_stickers - used_red_stickers
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 62
Q3: [EOQ]
Ans: 62""",
            """"Input: There are 5 girls and 4 boys in the Masquerade, 12 more girls and 7 more boys joined. How many more girls than boys are in the Masquerade?
Q1: [generate python code] write down the arithmetic or algebra equations as python code
#1:
num_girls_before = 5
num_girls_joined = 12
num_boys_before = 4
num_boys_joined = 7
total_girls = num_girls_before + num_girls_joined
total_boys = num_boys_before + num_boys_joined
ans = total_girls - total_boys
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 6
Q3: [EOQ]
Ans: 6""",
        ],
    },
    "navigation": {
        "name": "Navigation",
        "description": """Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No".""",
        "instances": [
            """Input: Take 7 steps. Turn right. Take 8 steps. Turn around. Do you return to the starting point? Answer "Yes" or "No"
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
Q9: [EOQ]
Ans: No""",
            """Input: Always face forward. Take 6 steps right. Take 10 steps left. Take 3 steps left. Take 1 step forward. Take 4 steps left. Take 10 steps left.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
steps_forward = 1
steps_backward = 0
steps_right = 6
steps_left = 10 + 3 + 4 + 10
vertical = steps_forward - steps_backward
horizontal = steps_right - steps_left
print((vertical,horizontal))
Q2: [code execute] Execute the python code snippet.
#2: (1,-23)
Q3: [get answer] The final displacement is (0,0). True or False?
#3: False
Q4: [EOQ]
Ans: False""",
        ],
    }
    # "musique":{
    #     "name" : "Multi-step question answering",
    #     "description": "Answer complex questions that require decompositions into sub-questions and composing intermediate answers.",
    #     "instances": [
    #         """Input: """,
    #         """Input: """,
    #     ]
    # },
    # "finqa":{
    #     "name": "Financial Question Answering",
    #     "description": """Read the following text and table. To answer a question, write out intermediate arithmetic calculations arithmetic calculations as python code. Store your result as a variable named 'ans'.""",
    #     "instances": ["""Input: the aggregate commitment under the liquidity asset purchase agreements was approximately $ 23.59 billion and $ 28.37 billion at december 31 , 2008 and 2007 , respectively .
    # ( dollars in billions ) | 2008 amount | 2008 percent of total conduit assets | 2008 amount | percent of total conduit assets
    # united states | $ 11.09 | 46% ( 46 % ) | $ 12.14 | 42% ( 42 % )
    # australia | 4.30 | 17 | 6.10 | 21
    # great britain | 1.97 | 8 | 2.93 | 10
    # spain | 1.71 | 7 | 1.90 | 7
    # italy | 1.66 | 7 | 1.86 | 7
    # portugal | 0.62 | 3 | 0.70 | 2
    # germany | 0.57 | 3 | 0.70 | 2
    # netherlands | 0.40 | 2 | 0.55 | 2
    # belgium | 0.29 | 1 | 0.31 | 1
    # greece | 0.27 | 1 | 0.31 | 1
    # other | 1.01 | 5 | 1.26 | 5
    # total conduit assets | $ 23.89 | 100% ( 100 % ) | $ 28.76 | 100% ( 100 % )
    # what is percentage change in total conduit asset from 2007 to 2008?
    # Q1: [generate python code] write down the arithmetic or algebra equations as python code
    # #1:
    # total_conduit_assets_2007 = 28.76
    # total_conduit_assets_2008 = 23.89
    # net_change_in_total_conduit_assets = total_conduit_assets_2008 - total_conduit_assets_2007
    # percent_change_in_total_conduit_assets = net_change_in_total_conduit_assets / total_conduit_assets_2007
    # ans = percent_change_in_total_conduit_assets
    # print(ans)
    # Q2: [code execute] Execute the python code and get the value of "ans"
    # #2: -0.1693
    # Q3: [EOQ]
    # Ans: -0.1693""",
    # """Input:american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $ 75.0 million and network location intangibles of approximately $ 72.7 million . the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years .
    # - | preliminary purchase price allocation
    # current assets | $ 8763
    # non-current assets | 2332
    # property and equipment | 26711
    # intangible assets ( 1 ) | 21079
    # other non-current liabilities | -1349 ( 1349 )
    # fair value of net assets acquired | $ 57536
    # goodwill ( 2 ) | 5998
    # for acquired customer-related and network location intangibles , what is the expected annual amortization expenses , in millions?
    # Q1: [generate python code] write down the arithmetic or algebra equations as python code
    # #1:
    # customer_related_intangibles = 75
    # network_location_intangibles = 72.7
    # straight_line_basis = 20
    # amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
    # ans = amortization_expenses
    # print(ans)
    # Q2: [code execute] Execute the python code and get the value of "ans"
    # #2: 7.385
    # Q3: [EOQ]
    # Ans: 7.385"""]
    # }
}


additional_arithmetic_tasks = {
    "aqua_rat": {
        "name": "Arithmetic about Ratios",
        "description": "Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as needed.",
        "instances": [
            """Input: Divide the number 49 into two parts, such that if the greater part is increased by 6 and the lesser part is decreased by 11, their ratio is 9 to 2. What is the greater number?
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
Q7: [EOQ]
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
Q3: [EOQ]
Ans: 132
""",
        ],
    },
    "gsm8k": {
        "name": "Middle school arithmetic problems",
        "description": "Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.",
        "instances": [
            """Input: A toy manufacturer receives an order for 400 toys. 5 workers are available to work on the order. 2 of the workers produce 6 toys an hour, and another 2 workers produce 4 toys an hour. They all work on the order during their 10-hour shift, and by the end of their shift the manufacturer still needs another 20 toys to be able to ship the order. How many toys per hour does the fifth worker produce?
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
Q7: [EOQ]
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
Q6: [EOQ]
Ans: 270 miles""",
        ],
    },
    "elemetary_math_qa": {
        "name": "Elementary school arithmetic problems",
        "description": "What is the result of the following arithmetic operations? Write out intermediate arithmetic calculations as needed.",
        "instances": [
            """Input: add 70 to 90, subtract 20 from result, subtract result from 200.
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
Q5: [EOQ]
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
Q6: [EOQ]
Ans: 12""",
        ],
    },
    "mawps": {
        "name": "Arithmetic word problems",
        "description": "Solve the following arithmetic word problems, writing out intermediate arithmetic calculations as needed.",
        "instances": [
            """Input: Viola had 167 Bread. Nancy took 137 from him. Now How many Bread Viola have difference?
Q1: [arithmetic] How much bread does Viola have if he had 167 loafs before and Nancy too 137 of them?
#1: 167-137=30
Q2: [EOQ]
Ans: 30""",
            """Input: Rodney had 35 pear . He dice each pear into 13 slices . How many pear slices did Rodney make?
Q1: [arithmetic]  How many pear slices did Rodney make if he started with 35 pears and diced each into 13 slices?
#1: 35*13=455
Q2: [EOQ]
Ans: 455""",
        ],
    },
    "navigation": {
        "name": "Navigation",
        "description": """Description: Determine if following the given navigation instructions, you return to the starting point. If yes, say "Yes", otherwise, say "No".""",
        "instances": [
            """Input: Take 7 steps. Turn right. Take 8 steps. Turn around.
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
Q9: [EOQ]
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
Q9: [EOQ]
Ans: No""",
        ],
    },
}


few_shot_retrieval_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use search functions like Google search in one or more of your substeps, if there in insufficient information. Other functions like arithmetic and logical operations can also be used.
Description: (Knwon or Unknwon) Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
Input: How many hairs were on Neil Armstrong's head when he landed on the moon?
  choice: Unknown
  choice: Five million
Q1: [search] How many hairs were on Neil Armstrong's head when he landed on the moon?
#1:
Apollo 11 (July 16–24, 1969) was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin.
Neil Alden Armstrong (August 5, 1930 – August 25, 2012) was an American astronaut and aeronautical engineer who became the first person to walk on the Moon.
Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: No. The question is too specific
Q3: [compare] What is the final answer?
#3: Unknown
Q4: [EOQ]
Ans: Unknown
----
Description: (Anachronisms) An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism? Answer Yes/No.
Input: President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
Q1: [tag] What are the entities in this sentence?
#1:
President George H. W. Bush
Gulf War
Q2: [search] When was President George H. W. Bush president?
#2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
Q3: [search] When was the Gulf War fought?
#3: The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
#4: [subquestion] Could these entities have co-existed based on thier time periods alone?
Yes. Their time periods intersect.
Q5: [generate output] Is this an anachronism?
#5: No
Q6: [EOQ]
Ans: No
----
Description: (Hindu Knowledge) Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
  choice: Anjalikastra
  choice: Narayanastra
  choice: Agneyastra
  choice: Brahmastra
Q1: [search] In the Mahabharata, Karna is cursed to forget the incantations needed to use which weapon?
#1: As a result, he cursed Karna, saying that HIS MARTIAL SKILLS, including the use of BRAHMASTRA, would abandon him when he needed them most. Indra, the King of Gods, stung Karna in the form of a bee to get him cursed by Parshuram. Karna walked through the woods in despair, feeling dejected by the curse. A skilled & devoted warrior...
Q2: [compare] Which option is the answer in #3 most similar to?
#2: Brahmastra
Q3: [EOQ]
Ans: Brahmastra
----
Description: (Anachronisms) An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Does the sentence contain an anachrornism? Answer Yes/No.
Input: Kurt Cobain starred in the 1980 television show "Twin Peaks".
Q1: [tag] What are the entities in this sentence?
#1:
Kurt Cobain
"Twin Peaks"
Q2: [search] When did television show "Twin Peaks" air?
#2: Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
Q3: [search] When did Kurt Cobain live?
#3: Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
Q4: [subquestion] Could these entities have co-existed based on this information?
No. Musician  Kurt Cobain could not have starred in Twin Peaks.
Q5: [generate output] Is this an anachronism?
#5: Yes
Q6: [EOQ]
Ans: Yes
----
Description: (Knwon or Unknwon) Choose the option that best answers the question. If the question does not have a known answer, choose "Unknown".
Input: Where was Mark Twain born?
  choice: Unknown
  choice: Florida, Missouri
Q1: [search] Where was Mark Twain born?
#1:
Mark Twain. Samuel Langhorne Clemens was born in Florida, Missouri, and later moved with his family to Hannibal, Missouri, where he grew up.
Q2: [subquestion] Does the information help answer the question? There could be no definitive answer because the question is too specific, about personal details not in public record, because the answer is not yet known, or the question is opinion-based.
#2: Yes. The answer is Florida, Missouri
Q3: [compare] What is the final answer?
#3: Florida, Missouri
Q4: [EOQ]
Ans: Florida, Missouri
----
Description: (Hindu Knowledge) Answer questions about Hindu mythology by choosing the option that best answers the question.
Input: In the Hindu epic Ramayana, the main villain was a devotee of which deity?
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
Q5: [EOQ]
Ans: Shiva
----
Desciption: %s
Input: %s
Q1:"""


few_shot_string_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use string operations like splitting, reformatting, editing or merging. You can also use other operations like arithmetic and logic.
Description: (Date Understanding) Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
#1: 06/01/2021
Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: [arithmetic] What date is 24 hours later than today?
#3: 05/01/2021
Q4: [EOQ]
Ans: 05/31/2021
----
Description: (Language games) Translate English into Pig Latin.
Input: (English) Sami made his way across the bar and hugged Layla.
Q1: [string split] What are the words in "Sami made his way across the bar and hugged Layla."?
#1: ["Sami", "made", "his", "way", "across", "the",  "bar", "and", "hugged", "Layla", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Amisay", "ademay", "ishay", "ayway", "acrossyay", "ethay", "arbay", "andyay", "uggedhay", "Aylalay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
Q4: [EOQ]
Ans: Amisay ademay ishay ayway acrossyay ethay arbay andyay uggedhay Aylalay.
----
Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Savita Saeed Ramos Sato Yadav" and concatenate them using a space.
Q1: [string split] What are the words in "Savita Saeed Ramos Sato Yadav"?
#1: ["Savita", "Saeed", "Ramos",  "Sato",  "Yadav"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["v", "e", "m", "t", "d"]
Q3: [string merge] Concatenate #2 with spaces
#3: "v e m t d"
Q4: [EOQ]
Ans: v e m t d
----
Description: (Kth letter concatenation) Take the letters at position 3 of the words in a list of words and concatenate them using a space.
Input: Take the letters at position 3 of the words in "Ibrahim Francois Pei Shu Ngo" and concatenate them using a space.
Q1: [string split] What are the words in "Ibrahim Francois Pei Shu Ngo"?
#1: ["Ibrahim", "Francois", "Pei", "Shu", "Ngo"]
Q2: [string index] What is the third letter of words in the list in #1?
#2: ["r", "a", "i", "o", "u"]
Q3: [string merge] Concatenate #2 with spaces
#3: "r a i u o"
Q4: [EOQ]
Ans: r a i u o
----
Description: (Language games) Translate English into Pig Latin.
Input: (English) Tom is the most popular boy in school.
Q1: [string split] What are the words in "Tom is the most popular boy in school."?
#1: ["Tom", "is", "the", "most", "popular", "boy",  "in", "school", "."]
Q2: [string edit] Transfer the initial consonant of each word to the end of the word and adding "ay" after it.
#2: ["Omtay", "isyay", "ethay", "ostmay", "opularpay", "oybay",  "inyay", "oolschay", "."]
Q3: [string merge] Concatenate #2 into a full sentence.
#3: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
Q4: [EOQ]
Ans: Omtay isyay ethay ostmay opularpay oybay inyay oolschay.
----
Description: (Date Understanding) Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007 in MM/DD/YYYY
#1: 01/01/2007
Q2: [arithmetic] What date is one week from 01/01/2007?
#2: 01/08/2007
Q3: [EOQ]
Ans: 01/08/2007
----
Descripton: %s
Input: %s
Q1:
"""


few_shot_code_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use a python code generation and execution function in one or more of your substeps, if required. Other functions like arithmetic and logical operations can also be used.
Description: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
Input:
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
Q3: [EOQ]
Ans: NameError: name 'x' is not defined
----
Description: (Auto Debugging) Debug the following code snippets by finding the answer or the error message.
Input:
```
x = set([1, 1, 2, 3])
```
What is the value of x after this program executes?
Q1: [code edit] Edit the following code to print the value of x
x = set([1, 1, 2, 3])
#1:
x = set([1, 1, 2, 3])
print(x)
Q2: [execute] Execute the following python code snippet.
x = set([1, 1, 2, 3])
print(x)
#2: {1, 2, 3}
Q3: [EOQ]
Ans: {1, 2, 3}
----
Description: (Code description) Given a python code snippet, choose the option that is the best description of the code snippet.
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
Q6: [EOQ]
Ans: prints sum of two input numbers only if they are integers otherwise raises error
----
Description: (Code description) Given a python code snippet, choose the option that is the best description of the code snippet.
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
Descripton: %s
Input: %s
Q1:"""


few_shot_arithmetic_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can generate python code to solve arithmetic and algebra equations in using functions from sympy.
from sympy import Symbol
from sympy import simplify
import math
from sympy import solve_it
# solve_it(equations, variable): solving the equations and return the variable value.

Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input:  In a flight of 600 km, an aircraft was slowed down due to bad weather. Its average speed for the trip was reduced by 200 km/hr and the time of flight increased by 30 minutes. The duration of the flight is:  A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
duration = Symbol('duration', positive=True)
delay = 30 / 60
total_disntace = 600
original_speed = total_disntace / duration
reduced_speed = total_disntace / (duration + delay)
solution = solve_it(original_speed - reduced_speed - 200, duration)
ans = solution[duration]
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2:
1.0
Q3: [compare] Which of the options among A)1 hour B)2 hours C)3 hours D)4 hours E)5 hours is most similar to the answer?
#3: A
Q4: [EOQ]
Ans: A
----
Description: Solve the following arithmetic problems on ratios and fractions, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: M men agree to purchase a gift for Rs. D. If 3 men drop out how much more will each have to contribute towards the purchase of the gift?  A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these
Q1: [generate python code] write python code to solve the problem, using math and sympy.
#1:
M = Symbol('M')
D = Symbol('D')
cost_before_dropout = D / M
cost_after_dropout = D / (M - 3)
ans=simplify(cost_after_dropout - cost_before_dropout)
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 3*D/(M*(M - 3))
Q3: [compare] Which of the options among A)D/(M-3) B)MD/3 C)M/(D-3) D)3D/(M**2-3M) E)None of these is most similar to the answer?
#3: D
Q4: [EOQ]
Ans: D
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
total_eggs = 16
eaten_eggs = 3
baked_eggs = 4
sold_eggs = total_eggs - eaten_eggs - baked_eggs
dollars_per_egg = 2
ans = sold_eggs * dollars_per_egg
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 18
Q3: [EOQ]
Ans:18
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy.  She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed.  In the afternoon, she gives her chickens another 25 cups of feed.  How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
numb_of_chickens = 20
cups_for_each_chicken = 3
cups_for_all_chicken = num_of_chickens * cups_for_each_chicken
cups_in_the_morning = 15
cups_in_the_afternoon = 25
ans = cups_for_all_chicken - cups_in_the_morning - cups_in_the_afternoon
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 20
Q3: [EOQ]
Ans: 20
----
Description: Solve the following middle-school arithmetic problems, writing out intermediate arithmetic calculations as python code. Store your result as a variable named 'ans'.
Input: Joseph and Getty went to buy ice creams, they together bought 36 ice creams. On the way back, Joseph ate 12 of the ice creasm, and he has 2 ice creams left now.
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
num_ice_creams_bought_by_joseph = 2 + 12
total_ice_creams = 36
ans = total_ice_creams - num_ice_creams_bought_by_joseph
print(ans)
Q2: [code execute] Execute the python code in #1 and get the value of "ans"
#2: 22
Q3: [EOQ]
Ans: 22
----
Description: Solve the following problems on financial data in the form of text and tables, writing out intermediate calculations as python code. Store your result as a variable named 'ans'.
Input: american tower corporation and subsidiaries notes to consolidated financial statements ( 3 ) consists of customer-related intangibles of approximately $75.0 million and network location intangibles of approximately $72.7 million. the customer-related intangibles and network location intangibles are being amortized on a straight-line basis over periods of up to 20 years. For acquired customer-related and network location intangibles, what is the expected annual amortization expenses, in millions?
Q1: [generate python code] write down the arithmetic or algebra equations as python code, storing the answer as 'ans'
#1:
customer_related_intangibles = 75
network_location_intangibles = 72.7
straight_line_basis = 20
amortization_expenses = ( customer_related_intangibles + network_location_intangibles ) / straight_line_basis
ans = amortization_expenses
print(ans)
Q2: [code execute] Execute the python code and get the value of "ans"
#2: 7.385
Q3: [EOQ]
Ans: 7.385
----
Descripton: %s
Input: %s
Q1:"""


few_shot_free_prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. Thinking though the problem explicitly can be one of the substeps you use.
Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible. The final answer should be "yes" or "no".
Input: Is the following sentence plausible? "Santi Cazorla scored a touchdown."
Q1: [think step-by-step]
#1: Let's think step-by-step. Santi Cazorla is a soccer player. Touchdown is part of American football and rugby. So the answer is no.
Q2: [EOQ]
Ans: no
----
Description: (Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Input: Which sentence has the correct adjective order:
Options:
(A) repulsive small Brazilian exercise ship
(B) Brazilian repulsive exercise small ship
Q1: [think step-by-step]
#1: Let's think step-by-step. When there is more than one adjective before a noun, the adjectives need to respect the following order before a noun: "[1. opinion] [2. size] [3. age] [4. shape] [5. color] [6. origin] [7. material] [8. purpose] noun".
Option (A): "repulsive small Brazilian exercise ship". (1) "repulsive" falls into the opinion category. (2) "small" falls into the size category. (3) "Brazilian" falls into the origin category. (4) "exercise" falls into the purpose category. Option (A) has the following adjective order: [1. opinion] [2. size] [6. origin] [8. purpose] (or, in numeric terms, 1 2 6 8). Because 1 < 2 < 6 < 8 is correct, (A) has the correct ordering.
Option (B): "Brazilian repulsive exercise small ship". Option (B) has the following adjective order: [6. origin] [1. opinion] [8. purpose] [2. size] (or, in numeric terms, 6 1 8 2). Because 6 < 1 < 8 < 2 is not correct, (B) does not have the correct ordering. So the answer is (A).
Q2: [EOQ]
Ans: (A)
----
Description: (Sports Understanding) Determine whether an artificially constructed sentence relating to sports is plausible. The final answer should be "yes" or "no".
Input: Is the following sentence plausible? "Bam Adebayo scored a reverse layup in the Western Conference Finals."
Q1: [think step-by-step]
#1: Let's think step-by-step. Bam Adebayo is an American basketball player. Scoring a reverse layup in the Western Conference Finals is part of the NBA Finals. So the answer is yes.
Q2: [EOQ]
Ans: yes
----
Description: (Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Input: "It is not always easy to see who is related to whom -- and in which ways. The following argument pertains to this question: To begin with, Lesley is a close friend of Fernando. Moreover, being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy. It follows that Lesley is a great-grandfather of Leroy."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid
Q1: [think step-by-step]
#1:
Let's think step-by-step.
(1) Lesley is a close friend of Fernando: Lesley = friend(Fernando).
(2) Being a close friend of Fernando or a schoolmate of Lowell is sufficient for being a great-grandfather of Leroy: If X = friend(Fernando) OR SCHOOLMATE(Lowell), then X = great-grandfather(Leroy).
Hypothesis: Does it follow that Lesley is a great-grandfather of Leroy: Lesley = great-grandfather(Leroy)?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (1), we have Lesley = friend(Fernando). By (2), we have if Lesley = friend(Fernando), then Lesley = great-grandfather(Leroy).
So, it is true that Lesley is a great-grandfather of Leroy. So the answer is valid.
Q2: [EOQ]
Ans: valid
----
Description: (Hyperbation) Identify correct adjective ordering from the two choices. This involves selecting what would be considered the more inexplicably "intuitive" sentence by a native English speaker.
Input:  Which sentence has the correct adjective order:
Options:
(A) rubber terrible ship
(B) terrible rubber ship
Q1: [think step-by-step]
#1: Let's think step-by-step. When there is more than one adjective before a noun, the adjectives need to respect the following order before a noun: "[1. opinion] [2. size] [3. age] [4. shape] [5. color] [6. origin] [7. material] [8. purpose] noun".
Option (A): "rubber terrible ship". (1) rubber" falls into the material category. (2) "terrible" falls into the opinion category. Option (A) has the following adjective order: [7. material] [1. opinion] (or, in numeric terms, 7 1). Because 7 < 1 is not correct, (A) does not have the correct ordering.
Option (B): "terrible rubber ship". Option (B) has the following adjective order: [1. opinion] [7. material] (or, in numeric terms, 1 7). Because 1 < 7 is correct, (B) has the correct ordering. So the answer is (B).
Q2: [EOQ]
Ans: (B)
----
Description: (Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input.
Input: On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?
Q1: [think step-by-step]
#1: Let's think step-by-step. According to this question, the color of the stress ball is blue. So the answer is blue.
Q2: [EOQ]
Ans: blue
----
Description: (Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Input: "It is not always easy to grasp who is consuming which products. The following argument pertains to this question: Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both. No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, in the same time, a loyal buyer of Caress soap. It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid
Q1: [think step-by-step]
#1: Let's think step by step.
(1) Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both: If X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress).
(2): No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and a loyal buyer of Caress soap at the same time. If X = regular-consumer(Lush), then X = NOT (rare-consumer(Nioxin) AND loyal-buyer(Caress)).
Hypothesis: Does it follow that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap: If X = infrequent-user(Paul Mitchell), then X = NOT (regular-consumer(Lush))?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (1), we have if X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress). We need to consider both cases separately:
The case X = rare-consumer(Nioxin) does not appear in (2).
The case X = loyal-buyer(Caress) does not appear in (2), either.
So, from (1) and (2), we cannot necessarily deduce the Hypothesis. So the answer is invalid.
Q2: [EOQ]
Ans: invalid
----
Description: (Movie Recommendations) Propose relevant movie predictions to a user, based on an input prompt sentence enumerating 4 liked movies.
Input: Find a movie similar to Minority Report, Total Recall, Inside Out, Forrest Gump:
Options:
(A) Phenomena
(B) Lilting
(C) Catwoman
(D) Edge of Tomorrow
Q1: [think step-by-step]
#1: Let's think step-by-step. What genre are all these movies about and when were they released.
- Minority Report (action, crime, mystery; 2002)
- Total Recall (action, adventure, science-fiction; 2012)
- Inside Out (animation, family, comedy; 2015)
- Forrest Gump (comedy, drama, romance; 1994)
These are all famous movies produced in the past few decades.Amongst all the options, the only movie similar to these ones seems to be Edge of Tomorrow (action, adventure, crime, mystery; 2014), as it is also a science-fiction movie and features Tom Cruise. So the answer is (D).
Q2: [EOQ]
Ans: (D)
----
Descripton: %s
Input: %s
Q1:"""

additional_free_tasks = """Description: (Formal Fallacies) Distinguish deductively valid syllogistic arguments from formal fallacies, paying specific attention to negations.
Input: "It is not always easy to grasp who is consuming which products. The following argument pertains to this question: Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both. No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and, in the same time, a loyal buyer of Caress soap. It follows that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap."
Is the argument, given the explicitly stated premises, deductively valid or invalid?
Options:
- valid
- invalid
Q1: [think step-by-step]
#1: Let's think step by step.
(1) Every infrequent user of Paul Mitchell shampoo is either a rare consumer of Nioxin shampoo or a loyal buyer of Caress soap, or both: If X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress).
(2): No regular consumer of Lush soap is a rare consumer of Nioxin shampoo and a loyal buyer of Caress soap at the same time. If X = regular-consumer(Lush), then X = NOT (rare-consumer(Nioxin) AND loyal-buyer(Caress)).
Hypothesis: Does it follow that whoever is an infrequent user of Paul Mitchell shampoo is not a regular consumer of Lush soap: If X = infrequent-user(Paul Mitchell), then X = NOT (regular-consumer(Lush))?
Let’s see whether the Hypothesis can be deduced from the arguments (1) and (2) by logical reasoning?
By (1), we have if X = infrequent-user(Paul Mitchell), then X = rare-consumer(Nioxin) OR X = loyal-buyer(Caress). We need to consider both cases separately:
The case X = rare-consumer(Nioxin) does not appear in (2).
The case X = loyal-buyer(Caress) does not appear in (2), either.
So, from (1) and (2), we cannot necessarily deduce the Hypothesis. So the answer is invalid.
Q2: [EOQ]
Ans: invalid
----
Description: (Movie Recommendations) Propose relevant movie predictions to a user, based on an input prompt sentence enumerating 4 liked movies.
Input: Find a movie similar to Twister, The Silence of the Lambs, Independence Day, Braveheart:
Options:
(A) They Shoot Horses
(B) Don't They
(C) Forrest Gump
(D) The Salton Sea
(E) Extreme Days
Q1: [think step-by-step]
#1: Let's think step-by-step. What genre are all these movies about and when were they released.
- Twister (action, adventure, thriller; 1996)
- The Silence of the Lambs (crime, drama, thriller; 1991)
- Independence Day (action, science-fiction, drama; 1996)
- Braveheart (biography, drama, epic; 1995)
These are all famous Hollywood movies produced around the 1990s. Amongst all the options, the only movie similar to these ones seems to be Forrest Gump (comedy, drama, romance; 1994). So the answer is (C).
Q2: [EOQ]
Ans: (C)
----
Description: (Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input.
Input: On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?
Q1: [think step-by-step]
#1: Let's think step-by-step. According to this question, the color of the stress ball is blue. So the answer is blue.
Q2: [EOQ]
Ans: blue
----
Description: (Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input.
Input: On the table, you see a bunch of objects arranged in a row: a purple paperclip, a pink stress ball, a brown keychain, a green scrunchiephone charger, a mauve fidget spinner, and a burgundy pen. What is the color of the object directly to the right of the stress ball?
Q1: [think step-by-step]
#1: Let's think step by step.
According to this question, the objects are arranged in a row, from left to right, as follows: (1) a purple paperclip, (2) a pink stress ball, (3) a brown keychain, (4) a green scrunchiephone charger, (5) a mauve fidget spinner, (6) a burgundy pen.
The stress ball is the second object on the list, namely (2). The object that is to the right of the stress ball corresponds to (3), which is a brown keychain.
The color of the keychain is brown. So the answer is brown.
Q2: [EOQ]
Ans: brown
----
Description: (Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input.
Input: On the nightstand, there is a red pencil, a purple mug, a burgundy keychain, a fuchsia teddy bear, a black plate, and a blue stress ball. What color is the stress ball?
Q1: [think step-by-step]
#1: Let's think step-by-step. According to this question, the color of the stress ball is blue. So the answer is blue.
Q2: [EOQ]
Ans: blue
----
Description: (Reasoning about colored objects) Given a collection of colored objects in the text input, answer the question at the end of the input.
Input: On the table, you see a bunch of objects arranged in a row: a purple paperclip, a pink stress ball, a brown keychain, a green scrunchiephone charger, a mauve fidget spinner, and a burgundy pen. What is the color of the object directly to the right of the stress ball?
Q1: [think step-by-step]
#1: Let's think step by step.
According to this question, the objects are arranged in a row, from left to right, as follows: (1) a purple paperclip, (2) a pink stress ball, (3) a brown keychain, (4) a green scrunchiephone charger, (5) a mauve fidget spinner, (6) a burgundy pen.
The stress ball is the second object on the list, namely (2). The object that is to the right of the stress ball corresponds to (3), which is a brown keychain.
The color of the keychain is brown. So the answer is brown.
Q2: [EOQ]
Ans: brown
----"""


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
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts_to_compare = []
    for key, task in TASKS.items():
        doc = (
            "\n".join(task["instances"][0].split("\n")[:2])
            + "\n"
            + task["instances"][1].split("\n")[1]
        )
        texts_to_compare.append(doc)
    doc_embeddings = model.encode(texts_to_compare)  # (13, 384)
    query = (
        "Description:"
        + task_description
        + "\nInput: "
        + io_pairs[0][0]
        + "\nInput: "
        + io_pairs[1][0]
    )
    query_embeddings = model.encode([query])  # (1, 384)
    similarity_matrix = util.pytorch_cos_sim(query_embeddings, doc_embeddings)
    # Choose top N
    top_tasks = (-similarity_matrix.squeeze(0).numpy()).argsort()[:N]
    selected_tasks = [list(TASKS.keys())[t] for t in top_tasks]
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks:
        print(TASKS[task]["name"])
        prompt += TASKS[task]["description"] + "\n" + TASKS[task]["instances"][0] + "\n----\n"
    prompt += "Description: %s\nInput: %s\nQ1:"
    return prompt


def similar_auto_breakdowns(task_description, io_pairs, N=5):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gpt3 = OpenAIModel(model="text-davinci-002", max_length=500, temperature=0.3, quote="---", n=1)
    texts_to_compare = []
    for key, task in TASKS.items():
        print(key)
        doc = (
            "\n".join(task["instances"][0].split("\n")[:2])
            + "\n"
            + task["instances"][1].split("\n")[1]
        )
        doc = doc + "\nA: Let's think step-by-step.\n"
        doc = doc + gpt3(doc)[0]
        texts_to_compare.append(doc)
    doc_embeddings = model.encode(texts_to_compare)  # (13, 384)

    query = (
        "Description:"
        + task_description
        + "\nInput: "
        + io_pairs[0][0]
        + "\nInput: "
        + io_pairs[1][0]
    )
    query = query + "\nA: Let's think step-by-step.\n"
    query = query + gpt3(query)[0]

    query_embeddings = model.encode([query])  # (1, 384)
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


def llm_similar_tasks(task_name, task_description, io_pairs, N=5):
    gpt3 = OpenAIModelLogProb(
        model="text-davinci-002",
        max_length=500,
        temperature=0.5,
        quote="---",
        n=1,
        logprobs=10,
    )

    query = (
        "["
        + task_name
        + "] "
        + task_description
        + " Input: "
        + io_pairs[0][0]
        + "\nThe final answer is "
        + io_pairs[0][1]
    )

    selected_tasks = []
    selection_reasons = []
    selection_scores = []
    for key, task in TASKS.items():
        selected_tasks.append(key)
        parsed_program = parse_program(task["instances"][0])
        doc = (
            "["
            + task["name"]
            + "] "
            + task["description"]
            + " Input: "
            + parsed_program.input_node.text
            + parsed_program.answer_node.text.replace("Ans: ", "The final answer is ")
        )
        filled_prompt = task_similarity_prompt % (query, doc)

        # similar = gpt3(filled_prompt)[0]
        results = gpt3(filled_prompt)
        results["choices"][0]["logprobs"]["tokens"]
        results["choices"][0]["logprobs"]["token_logprobs"]
        top_logprobs = results["choices"][0]["logprobs"]["top_logprobs"]
        yes_score = -100000.0
        for token_dict in top_logprobs:
            alist = [(k, v) for k, v in token_dict.items()]
            alist = sorted(alist, key=lambda x: x[1], reverse=True)
            for k, v in alist:
                if k.strip().lower() == "yes":
                    if v > yes_score:
                        yes_score = v
        selection_reasons.append(results["choices"][0]["text"])
        selection_scores.append(yes_score)

    selected_tasks = sorted(
        [
            (task_name, task_score, reason)
            for task_name, task_score, reason in zip(
                selected_tasks, selection_scores, selection_reasons
            )
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    # if len(selected_tasks) >= N:
    #     random_tasks = np.random.choice(len(selected_tasks), N, replace=False)
    #     selected_tasks = [selected_tasks[t] for t in random_tasks]
    prompt = """In these examples, you are given a task description and an input. Break the input down into subtasks in order to solve the task. You can use affordances like string operations, search engines, arithmetic functions, or code generation. Be sure to use "[]" to specify affordances in subtasks.\n----\n"""
    for task in selected_tasks[:N]:
        print(task[0], task[-1])
        prompt += (
            "Description: (" + TASKS[task[0]]["name"] + ") " + TASKS[task[0]]["description"] + "\n"
        )
        prompt += (
            TASKS[task[0]]["instances"][0]
            + "\n----\n"
            + TASKS[task[0]]["instances"][1]
            + "\n----\n"
        )

    prompt += "Description: %s\nInput: %s\nQ1:"

    return prompt


def task_selection_accuracy(reference_tasks, task_name, task_description, io_pairs, N=3):
    # define Jaccard Similarity function
    def jaccard(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    gpt3 = OpenAIModel(
        model="text-davinci-002",
        max_length=500,
        temperature=0.3,
        quote="---",
        n=1,
        logprobs=10,
    )

    query = (
        "["
        + task_name
        + "] "
        + task_description
        + " Input: "
        + io_pairs[0][0]
        + "\nThe final answer is "
        + io_pairs[0][1]
    )

    selected_tasks = []
    all_reasons = []
    selection_reasons = []
    for key, task in TASKS.items():
        parsed_program = parse_program(task["instances"][0])
        doc = (
            "["
            + task["name"]
            + "] "
            + task["description"]
            + " Input: "
            + parsed_program.input_node.text
            + parsed_program.answer_node.text.replace("Ans: ", "The final answer is ")
        )
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
