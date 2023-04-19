import json
import re
import subprocess
import sys

import adatest
import numpy as np
import openai
import parsimonious
import requests
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

subscription_key = "28dc412935f24fb9974d80c30915483a"
search_url = "https://api.bing.microsoft.com/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": subscription_key}

with open("openai.key", "r") as file:
    openai.api_key = file.read().strip()
print("OpenAI API key read successfully")

cache_dir = "./data"


class HuggingFaceModel(adatest.Model):
    def __init__(
        self,
        model="google/flan-t5-small",
        quote='"',
        temperature=0.7,
        top_p=1,
        max_length=30,
        n=1,
        logprobs=None,
    ):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs

    def __call__(self, strings):
        inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
        outputs = self.model.generate(**inputs)
        resp = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return resp


class OpenAIModel(adatest.Model):
    def __init__(
        self,
        model="text-davinci-002",
        quote='"',
        temperature=0.7,
        top_p=1,
        max_length=30,
        n=1,
        logprobs=None,
    ):
        self.model_name = model
        if "google" in model:
            self.model = HuggingFaceModel(model, quote, temperature, top_p, max_length, n, logprobs)
        else:
            self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs

    def __call__(self, strings):
        if "google" in self.model_name:
            resp = self.model(strings)
        else:
            resp = openai.Completion.create(
                model=self.model,
                prompt=strings,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.quote,
                logprobs=self.logprobs,
            )
        return [x["text"] for x in resp["choices"]]


class OpenAIModelLogProb(adatest.Model):
    def __init__(
        self,
        model="text-davinci-002",
        quote='"',
        temperature=0.7,
        top_p=1,
        max_length=30,
        n=1,
        logprobs=None,
    ):
        self.model = model
        self.api_key = openai.api_key
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n
        self.logprobs = logprobs

    def __call__(self, strings):
        resp = openai.Completion.create(
            model=self.model,
            prompt=strings,
            max_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            n=self.n,
            stop=self.quote,
            logprobs=self.logprobs,
        )
        return resp


gpt3 = OpenAIModel(model="text-davinci-002", max_length=200, quote="", n=1)


def propose_decomposition(decomp_prompt, io_pairs, n=20):
    # Default of 0.7 is being used.
    gpt3 = OpenAIModel(model="text-davinci-002", max_length=500, quote="---", n=n)
    prompt = """%s. Here are examples of input-output pairs for the task I'm trying to break down.
----
%s
----
Steps:
1.""" % (
        decomp_prompt,
        io_pairs,
    )
    return gpt3(prompt)


def propose_instruction(instruct_prompt, io_pairs, n=20):
    gpt3 = OpenAIModel(model="text-davinci-002", max_length=400, quote="---", n=n)
    prompt = """%s. Here are examples of input-output pairs for this task.
----
%s
----
I can do this task by""" % (
        instruct_prompt,
        io_pairs,
    )
    return gpt3(prompt)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_subset(inputs, labels, n=100):
    idxs = np.random.choice(len(inputs), n, replace=False)
    labs = np.array([labels[i] for i in idxs])
    subset = [inputs[i] for i in idxs]
    return labs, subset


def substring_match(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower():
            correct += 1
        count += 1
    return (1.0 * correct) / count


def substring_match_v2(labels, predictions):
    correct = 0
    count = 0
    for label, predict in zip(labels, predictions):
        for l in label:
            if l.lower() in predict.lower():
                correct += 1
                break
        count += 1
    return (1.0 * correct) / count


class Command:
    def __init__(self, command_name, command_input=None, command_output=None):
        self.command_name = command_name
        self.command_input = command_input
        self.command_output = command_output

    def __str__(self):
        command = ""
        command += self.command_name + "\n---\n"
        if (self.command_input) and (self.command_output):
            command += self.command_input + "\n---\n"
            command += self.command_output
        return command

    @classmethod
    def convert_to_nlprogram(cls, rank, command, input_only=False):
        # TODO: Add feature for multiline (not all multiline inputs and outputs are in new lines in the retrieval)

        if not command.command_input:
            # The program parse identified a command name but no command input or command output.
            # Based on strict grammer parsing rules, this command node is [EOQ]
            return "Q{0}: {1}".format(rank, command.command_name)

        input_sep = "\n" if "\n" in command.command_input else " "
        output_sep = "\n" if "\n" in command.command_output else " "
        if not input_only:
            if rank == 1:
                return " {1}{2}{3}\n#{4}:{5}{6}".format(
                    rank,
                    command.command_name,
                    input_sep,
                    command.command_input,
                    rank,
                    output_sep,
                    command.command_output,
                )
            return "Q{0}: {1}{2}{3}\n#{4}:{5}{6}".format(
                rank,
                command.command_name,
                input_sep,
                command.command_input,
                rank,
                output_sep,
                command.command_output,
            )
        else:
            if rank == 1:
                return " {1}{2}{3}".format(
                    rank, command.command_name, input_sep, command.command_input
                )
            return "Q{0}: {1}{2}{3}".format(
                rank, command.command_name, input_sep, command.command_input
            )


class StacktraceItem:
    def __init__(
        self,
        command,
        affordance,
        affordance_output,
        affordance_details,
        rerun_program,
        rerun_program_parse,
    ):
        self.command = command
        self.affordance = affordance
        self.affordance_output = affordance_output
        self.affordance_details = affordance_details
        self.rerun_program = rerun_program
        self.rerun_program_parse = rerun_program_parse


class Program:
    def __init__(self, input_node, command_node_list, answer_node):
        self.node_list = []

        for node in command_node_list:
            command_name = node[0].text.split(" ", 1)[1]
            if len(node) > 1:
                command_input = node[1].text
                command_output = node[3].text
                self.node_list.append(Command(command_name, command_input, command_output))
            else:
                self.node_list.append(Command(command_name))

        self.input_node = input_node
        self.answer_node = answer_node

    def __str__(self):
        program = ""
        program += "Program\n======\n"
        program += "Input\n------\n"
        program += self.input_node.__str__()
        program += "\nBreakdown\n------"
        for node in self.node_list:
            program += node.__str__()
            program += "\n------\n"
        program += "\nAnswer\n------\n"
        program += self.answer_node.__str__()
        return program


class Node:
    def __init__(self, expr_name, text):
        self.expr_name = expr_name
        self.text = text

    def __str__(self):
        return json.dumps({"expr_name": self.expr_name, "text": self.text}, indent=2)


def parse_incomplete_program(program=None):
    incomplete_grammar = parsimonious.grammar.Grammar(
        r"""
program = program_start*node*partial_node
program_start = input_start~r"( |\n)"text~r"\n"
input_start = ~r"Input:"
text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
node = command_node~r"\n"output_node~r"\n"
command_node = command_start~r"( |\n)"command_instruction
output_node = begin_answer~r"( |\n)"output
command_instruction = ~r"(?<=\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"
command_start = ~r"Q[0-9]+: \[[A-Za-z_ ]+\]"
begin_answer = ~r"\#[0-9]+:"
output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
partial_node = partial_command_answer / partial_command
partial_command = command_start~r"( |\n)"~r"(?<=\]( |\n))(.|\n|\t)*?$"
partial_command_answer = command_node~r"\n"partial_answer
partial_answer = begin_answer~r"( |\n)"~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?$"
"""
    )
    incomplete_parsed_program = incomplete_grammar.parse(program)
    return incomplete_parsed_program


def parse_program(program=None):
    def recursive_node_visit(node, selection_criterion, node_list):
        for child in node.children:
            recursive_node_visit(child, selection_criterion, node_list)
        if node.expr_name in selection_criterion:
            node_list.append(Node(node.expr_name, node.text))
            return

    #     grammar = parsimonious.grammar.Grammar(
    # r"""
    # program = program_start*node*partial_command*final_answer
    # program_start = input_start~r" "text~r"\n"
    # input_start = ~r"Input:"
    # text = ~r"(?<=Input: )[^\#]+?(?=\nQ[0-9]+:)"
    # node = command_node~r"\n"output_node~r"\n"
    # command_node = command_start~r" "command_instruction
    # output_node = begin_answer~r" "output
    # command_instruction = ~r"(?<=\] )[^\#]+?(?=\n\#[0-9]+)"
    # command_start = ~r"Q[0-9]+: \[[A-Za-z_ ]+\]"
    # begin_answer = ~r"\#[0-9]+:"
    # output = ~r"(?<=\#[0-9]+: )[^\#]+?(?=\nQ[0-9]+:)"
    # partial_command = command_start~r"\n"
    # final_answer = ~r"Ans: (.|\n)*$"
    # """)

    grammar = parsimonious.grammar.Grammar(
        r"""
program = program_start*node*partial_command*final_answer
program_start = input_start~r"( |\n)"text~r"\n"
input_start = ~r"Input:"
text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
node = command_node~r"\n"output_node~r"\n"
command_node = command_start~r"( |\n)"command_instruction
output_node = begin_answer~r"( |\n)"output
command_instruction = ~r"(?<=\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"
command_start = ~r"Q[0-9]+: \[[A-Za-z_ ]+\]"
begin_answer = ~r"\#[0-9]+:"
output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"
partial_command = command_start~r"\n"
final_answer = ~r"Ans:( |\n)(.|\n)*$"
"""
    )

    parsed_program = grammar.parse(program)

    # A full program has 4 children.: Input, full commans, final partial command (EOC) and answer
    # Access all children to get "input_start" and "text"
    input_node = parsed_program.children[0]
    input_node_list = []
    recursive_node_visit(input_node, ["input_start", "text"], input_node_list)
    input_text = input_node_list[1]
    command_nodes = parsed_program.children[1]
    command_node_list = []
    for node in command_nodes.children:
        # Access all children and focus on getting "command_start", "command_instruction", "begin_answer" and "output"
        child_node_list = []
        recursive_node_visit(
            node,
            ["command_start", "command_instruction", "begin_answer", "output"],
            child_node_list,
        )
        command_node_list.append(child_node_list)
    # Access text of answer node that starts with Ans:

    partial_command = []
    recursive_node_visit(parsed_program.children[2], ["command_start"], partial_command)
    command_node_list.append(partial_command)

    answer_node = parsed_program.children[3]
    answer = Node(answer_node.expr_name, answer_node.text)

    nl_program = Program(input_text, command_node_list, answer)
    return nl_program


def fix_program(program):
    # Given various incomplete program parses, figure out a few common mistakes GPT-3 makes and fix them.
    # This is a more all-encompassing version of complete_program
    pass


import re

# as per recommendation from @freylis, compile once only
CLEANR = re.compile("<.*?>")


def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, "", raw_html)
    return cleantext


def search(query, top_k=1):
    params = {"q": query, "textDecorations": True, "textFormat": "HTML"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    if "webPages" in search_results:
        snippets = [
            cleanhtml(v["name"] + " " + v["snippet"]) for v in search_results["webPages"]["value"]
        ][:top_k]
    else:
        return ""
    return "\n".join(snippets)


def execute_code(code_snippet):
    # scope of variables to be defined here?
    # output = exec(code_snippet)
    result = subprocess.run([sys.executable, "-c", code_snippet], capture_output=True, text=True)
    return result


def generate_code(instruction, code_input):
    # Call Codex 002
    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input="x = " + code_input,
        instruction="Python code for " + instruction,
        temperature=0,
        top_p=1,
    )
    return response


def get_few_shot_prompt(inputs, labels, n=150):
    idx = np.random.randint(0, len(inputs), n)
    prompt_string = ""
    for id_, (inp, label) in enumerate(zip(inputs, labels)):
        if id_ not in idx:
            continue
        prompt_string += inp + "\n"
        prompt_string += label[0] + "\n"
        prompt_string += "----\n"
    return prompt_string


def edit_code(instructions, current_code):
    # Call Codex 001 (Edit) Model
    # Call Codex 002
    response = openai.Edit.create(
        model="code-davinci-edit-001",
        input="x = " + current_code,
        instruction="Python code for " + instructions,
        temperature=0,
        top_p=1,
    )
    return response


def get_answer(x, return_original=False):
    search_result = re.search("Ans: ", x)
    if search_result:
        return x[search_result.span(0)[1] :].strip()
    else:
        # Program did not complete
        matches = [match for match in re.finditer("\#[0-9]+:", x)]
        # If any match is made, return the last match output
        if len(matches):
            return x[matches[-1].start() :].strip()
        if return_original:
            return x.strip()
        return ""


def get_autocot_answer(x, answer_prompt="The final answer is ", return_original=False):
    if re.search(answer_prompt, x):
        return x[re.search(answer_prompt, x).span(0)[-1] :].strip()
    else:
        if return_original:
            return x.strip()
        return ""


program = """Input:
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
Q5: [compare]
Which of the generated code snippets are most like the original one?"""

# 5: prints sum of two input numbers only if they are integers otherwise raises error"""

"""
Q6: [EOC]
Ans:
prints sum of two input numbers only if they are integers otherwise raises error
"""

# print(parse_incomplete_program(program))
