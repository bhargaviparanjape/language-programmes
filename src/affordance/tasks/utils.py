from dataclasses import dataclass
import re
import subprocess
import sys
from typing import List, Union

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
    ):
        self.model_name = model
        if "google" in model:
            self.model = HuggingFaceModel(model, quote, temperature, top_p, max_length, n)
        else:
            self.model = model
        self.quote = quote
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.n = n

    def __call__(self, strings: Union[str, List[str]]) -> List[str]:
        if "turbo" in self.model_name:  # chat models
            if isinstance(strings, str):
                strings = [strings]

            out = []
            for string in strings:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful, tool-augmented assistant.",
                        },
                        {"role": "user", "content": string},
                    ],
                    max_tokens=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    n=self.n,
                    stop=self.quote,
                )

                finish_reason = resp["choices"][0]["finish_reason"]
                if finish_reason != "stop":
                    print(f"WARNING: incomplete model response: finish reason '{finish_reason}'")

                out.append(resp["choices"][0]["message"]["content"])
            return out

        if "google" in self.model_name:
            resp = self.model(strings)
        else:  # text completion models
            resp = openai.Completion.create(
                model=self.model,
                prompt=strings,
                max_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                n=self.n,
                stop=self.quote,
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
    """Yield successive n-sized chunks from l.

    Ex:
    >>> list(chunks(list(range(8)), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7]]
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_subset(inputs, labels, n=100):
    idxs = np.random.choice(len(inputs), n, replace=False)
    labs = np.array([labels[i] for i in idxs])
    subset = [inputs[i] for i in idxs]
    return labs, subset


def substring_match(labels: List[str], predictions: List[str]) -> float:
    """Return the rate at which the label appears as a substring of the prediction."""
    correct = 0
    for label, predict in zip(labels, predictions):
        if label.lower() in predict.lower():
            correct += 1

    return correct / min(len(labels), len(predictions))


def substring_match_v2(labels, predictions):
    correct = 0
    for label, predict in zip(labels, predictions):
        for l in label:
            if l.lower() in predict.lower():
                correct += 1
                break
    return correct / min(len(labels), len(predictions))


def _sanitize_multiline(s: Union[str, None]) -> str:
    """Sanitize a str (possible None) with an arrow for printing if it's a multi-line string."""
    out = str(s)

    if "\n" in out:
        out = "↓\n" + out

    return out


@dataclass
class Command:
    """This class represents a command with optional input and output. The command forms a part of
    a program. See the docstring for Program."""

    name: str
    input: Union[str, None] = None
    output: Union[str, None] = None

    def __str__(self):
        return f"""{self.name}
-  input: {_sanitize_multiline(self.input)}
- output: {_sanitize_multiline(self.output)}"""

    def to_program_chunk(self, rank: int, input_only=False) -> str:
        """Convert the program back to a string containing the query and output, e.g.:

        Q2: [search] Albert Einstein
        #2: Albert Einstein was born ...
        """
        # TODO: Add feature for multiline (not all multiline inputs and outputs are in new lines in
        # the retrieval)

        if not self.input:
            # The program parse identified a command name but no command input or command output.
            # Based on strict grammer parsing rules, we know this command node is [EOQ].
            return f"Q{rank}: {self.name}"

        input_sep = "\n" if "\n" in self.input else " "
        output_sep = "\n" if "\n" in self.output else " "

        out = " {}{}{}".format(self.name, input_sep, self.input)
        if rank != 1:
            out = f"Q{rank}:" + out

        if not input_only:
            out += "\n#{}:{}{}".format(rank, output_sep, self.output)

        return out


@dataclass
class Program:
    """This class stores a program, which involves an input text, a series of Commands, and an
    answer text.

    Each command is represented as a query string (starting with e.g. "Q#:"), which has a command
    name and input, and an output string (starting with e.g. "#1:").

    If any text is multi-line (i.e. contains "\\n"), it goes on a separate line from the previous
    text.

    Here's an example:
        Input: Q: Dominic is the star discus thrower on South's varsity track and field team. In
        last year's regional competition, Dominic whirled the 1.8 kg discus in a circle with a
        radius of 1.2 m, ultimately reaching a speed of 45 m/s before launch. Determine the net
        force acting upon the discus in the moments before launch.
        Q1: [generate python code] write down the arithmetic or algebra equations as python code,
        storing the answer as 'ans'
        #1:
        F = mv**2/r
        ans = F
        print(ans)
        Q2: [code execute] Execute the python code in #1 and get the value of "ans"
        #2:
        2160
        Q3: [EOQ]
        Ans: 2160
    """

    input: str
    commands: List[Command]
    answer: str

    def __str__(self):
        command_str = "\n---\n".join(str(c) for c in self.commands)

        return f"""Program
======
Input: {_sanitize_multiline(self.input)}
------
Commands:
{command_str}
------
Answer: {_sanitize_multiline(self.answer)}"""

    _grammar = parsimonious.grammar.Grammar(
        r"""
# A program consists of a program start, nodes, partial commands, and a final answer.
program = program_start* node* partial_command* final_answer

# A program start consists of an input start, whitespace or newline, and text followed by a newline.
program_start = input_start ~r"( |\n)" text ~r"\n"

# An input start is defined by the string "Input:".
input_start = ~r"Input:"

# Text is any sequence of characters between "Input:" and a newline followed by a command (e.g.
# "Q2:").
text = ~r"(?<=Input:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"

# A node consists of a command node followed by a newline and an output node followed by a newline.
node = command_node ~r"\n" output_node ~r"\n"

# A command node consists of a command start, whitespace or newline, and a command instruction.
command_node = command_start ~r"( |\n)" command_instruction

# A command start is a question number followed by a command description in square brackets.
command_start = ~r"Q[0-9]+: \[[A-Za-z_ ]+\]"

# A command instruction is any sequence of characters between "] " and a newline followed by an
# output start (e.g. "#2:").
command_instruction = ~r"(?<=\]( |\n))(.|\n|\t)*?(?=\n\#[0-9]+)"

# An output node consists of a begin answer, whitespace or newline, and output.
output_node = begin_answer ~r"( |\n)" output

# A begin answer is defined by a number sign and a number followed by a colon.
begin_answer = ~r"\#[0-9]+:"

# Output is any sequence of characters between a begin_answer and a newline followed by a command
# start.
output = ~r"(?<=\#[0-9]+:( |\n))(.|\n|\t)*?(?=\nQ[0-9]+:)"

# A partial command is a command node missing an instruction.
partial_command = command_start ~r"\n"

# A final answer is defined by the string "Ans:", whitespace or newline, and any sequence of
# characters until the end of the string.
final_answer = ~r"Ans:( |\n)(.|\n)*$"
"""
    )

    @classmethod
    def from_str(cls, s: str) -> "Program":
        """Parses the program components from the text completion."""
        parsed = cls._grammar.parse(s)

        def match_children(node, matches: List[str], node_list: List[str]):
            """Depth-first search on the node to find all nodes whose expr_name is in matches. These
            nodes are appended to node_list."""
            for child in node.children:
                match_children(child, matches, node_list)
            if node.expr_name in matches:
                node_list.append(node.text)

        # A full program has 4 children: input, full command, final partial command (EOQ), and
        # answer.
        input_node = parsed.children[0]
        command_node = parsed.children[1]
        final_partial_command_node = parsed.children[2]
        answer_node = parsed.children[3]

        # get the input text
        input_elements = []
        match_children(input_node, ["input_start", "text"], input_elements)
        input_text = input_elements[1]

        # get the commands
        commands = []
        for node in command_node.children + [final_partial_command_node]:
            elements = []
            match_children(
                node,
                ["command_start", "command_instruction", "begin_answer", "output"],
                elements,
            )

            # build the command
            name = elements[0].split(" ", 1)[1]
            if len(elements) > 1:
                c = Command(name, elements[1], elements[3])
            else:
                c = Command(name)

            commands.append(c)

        # get the answer
        answer_text = answer_node.text[len("Ans: ") :]

        return cls(input_text, commands, answer_text)


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


_answer_regex = re.compile(r"Ans: (.+)", flags=re.DOTALL)
_output_regex = re.compile(r"\#\d:( |\n)(.*?)(?=Q\d|$)", flags=re.DOTALL)


def get_answer(x: str) -> Union[str, None]:
    """Extract the answer from the end of a natural language program.

    None is returned if no answer could be parsed from the text.
    """
    result = _answer_regex.search(x)
    if result:
        return result.group(1).strip()

    # program did not complete; use the last program output
    matches = _output_regex.findall(x)
    # if any match is made, return the last match output
    if len(matches) > 0:
        return matches[-1][1].strip()

    return None


def get_autocot_answer(x, answer_prompt="The final answer is ", return_original=False):
    if re.search(answer_prompt, x):
        return x[re.search(answer_prompt, x).span(0)[-1] :].strip()
    else:
        if return_original:
            return x.strip()
        return ""
