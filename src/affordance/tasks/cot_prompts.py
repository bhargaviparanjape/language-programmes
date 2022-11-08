import openai


TASKS = {"date_understanding" : {
    "description": "Date understanding task: Find the required date in MM/DD/YYYY using information about related events and dates in the input.",
    "instances": [
        """Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?
Q1: [string reformat] Jun 1, 2021 in MM/DD/YYYY
#1: 06/01/2021
Q2: [arithmetic] 06/01/2021 is 2 days away from now. What date is today?
#2: Today is 04/01/2021
Q3: [arithmetic] What date is 24 hours later than today?  
#3: 05/01/2021
Q4: [EOQ]
05/31/2021""", 
        """Description: Find the required date in MM/DD/YYYY using information about related events and dates in the input. Clue: First find what day is today.
Input: Today is the first day of 2007. What is the date one week from today in MM/DD/YYYY?
Q1: [string reformat] first day of 2007 in MM/DD/YYYY 
#1: 01/01/2007
Q2: [arithmetic] What date is one week from 01/01/2007? 
#2: 01/08/2007
Q3: [EOQ]
01/08/2007
----"""]
}}