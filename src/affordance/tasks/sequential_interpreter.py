from dataclasses import dataclass
import textwrap
from typing import List, Sequence, Tuple, Union

import parsimonious
from tools import Tool, get_tool
from tqdm import tqdm
from utils import Command, OpenAIModel, Program


def trim_demonstrations(s: str) -> str:
    """Trim the few-shot demonstrations from prompt text."""
    return s.rsplit("----\n", 1)[1].split("\n", 1)[1]


@dataclass
class StacktraceItem:
    command: Command
    tool: str
    tool_output: str
    tool_details: str
    rerun_program: str
    rerun_program_parsed: Program
    run_title: str


class TopDownVisitor:
    def __init__(
        self,
        model_name: str = "text-davinci-002",
        temperature: float = 0.3,
        rerun: bool = True,
        exclude: Sequence[str] = [],
    ):
        self.rerun = rerun
        self.exclude = exclude
        self.execution_details: List[StacktraceItem] = []
        self.program_completer = OpenAIModel(
            model_name, quote="---", temperature=temperature, max_length=500, n=1
        )

    @staticmethod
    def program_ends(program: str) -> bool:
        """Return true if the program contains [EOQ] and a final answer."""
        return "[EOQ]\nAns:" in program

    def batch_visit(
        self, prefixes: List[str], programs: List[str], run_titles: List[str]
    ) -> List[str]:
        answers = []
        for prefix, program, run_title in tqdm(zip(prefixes, programs, run_titles)):
            answers.append(self.visit(prefix, program, run_title))
        return answers

    def shorten_prefix(self, prefix: str, to_remove: int) -> str:
        """Remove few-shot examples from the prefix, starting with the latter ones.

        We start with the latter ones in case they're organized by task relevance.
        """
        # split along demonstrations
        individual_programs = prefix.split("\n----\n")

        shifted_programs = individual_programs[: -(to_remove + 1)] + [individual_programs[-1]]

        return "\n----\n".join(shifted_programs)

    def complete_program(self, prefix: str, program: str, runs: int = 5):
        separator = "\nQ"

        for _ in range(runs):
            if self.program_ends(program):
                break

            # shift window on few-shot demonstrations
            prefix = self.shorten_prefix(prefix, 1)

            # TODO: explicitly check if the program ends in answers or questions and accordingly
            # change the separator

            program += separator + self.program_completer(prefix + program + separator)[0]

        return program

    def rerun_program(
        self, prefix: str, prev_commands: List[Command], current_output: str
    ) -> Tuple[str, str]:
        # recreate the program string up to the current command
        prefix += "\n".join(c.to_program_chunk(i + 1) for i, c in enumerate(prev_commands[:-1]))
        if prefix:
            prefix += "\n"

        # recreate the input text of the current command
        prefix += prev_commands[-1].to_program_chunk(len(prev_commands), input_only=True)
        output_sep = "\n" if "\n" in current_output else " "
        prefix += f"\n#{len(prev_commands)}:{output_sep}{current_output.strip()}\nQ"

        # get corrected output
        continuation = self.program_completer(prefix)[0]

        return prefix, continuation

    def rerun_answer(self, prefix: str) -> str:
        return self.program_completer(prefix)[0]

    def visit(self, prefix: str, program: str, run_title="") -> str:
        # check if the program ends with an answer
        program_ends = self.program_ends(program)

        # The prompt contains the beginning of the program, so we'll prepend it to the program.
        program = trim_demonstrations(prefix) + program

        # print for debugging purposes
        print("visiting program:")
        print(textwrap.indent(program, "  "))
        print("program_ends:", program_ends)

        # If the program does not end completely, run further to end it.
        if not program_ends:
            new_program = self.complete_program(prefix, program)
            if not self.program_ends(new_program):
                # after one attempt at completing program, give up
                print("WARNING: unable to complete program:")
                print(textwrap.indent(program, "  "))

                return program
            program = new_program

        # parse the program
        try:
            p = Program.from_str(program)
        except parsimonious.ParseError as e:
            print("WARNING: error parsing program:")
            print(textwrap.indent(program, "  "))
            print(str(e))

            return program

        previous_input = p.input

        stack_trace = []

        # p gets updated as we run tools, so we have to iterate over the commands manually.
        i = 0
        while i < len(p.commands):
            command = p.commands[i]

            if command.name == "[EOQ]":
                break

            # get the tool function
            tool_func = get_tool(command.name, self.exclude)

            if tool_func is None:
                print(f"tool '{command.name}' not found")
                previous_input = command.output
                i += 1
                continue

            tool_output, tool_details = tool_func(command.input, previous_input, run_title)

            if tool_output:
                # For search, concatenate the GPT-3 output with retrieval output. For everything
                # else, replace.
                if command.name == "[search]":
                    command.output += " " + tool_output
                else:
                    command.output = tool_output

            if self.rerun:
                # run model again with modified command output
                new_prefix, new_output = self.rerun_program(
                    prefix, p.commands[: i + 1], command.output
                )

                # replace later commands (i+1 though N), then have the model re-complete the rest of
                # the program
                program = trim_demonstrations(new_prefix + new_output)
                program = self.complete_program(prefix, program)

                try:
                    p = Program.from_str(program)
                except parsimonious.ParseError as e:
                    print("WARNING: error parsing program:")
                    print(textwrap.indent(program, "  "))
                    print(str(e))

                    # If the rerun fails to generate a legitimate program, its unsafe to assume that
                    # it could replace the rest of the original program. The best course of action
                    # is to exit with the best new program.
                    break

                stack_trace.append(
                    StacktraceItem(
                        command, tool_func, tool_output, tool_details, program, p, run_title
                    )
                )
            else:
                new_program = prefix + "\n".join(
                    c.to_program_chunk(j) for j, c in enumerate(p.commands)
                )

                # The parsed program does not change since the future commands are not updated.
                # However, the final answer could change due to the new tool output.
                continuation = self.rerun_answer(new_program)
                new_program += continuation
                new_program = trim_demonstrations(new_program)
                stack_trace.append(
                    StacktraceItem(
                        command, tool_func, tool_output, tool_details, new_program, p, run_title
                    )
                )

                # TODO: why do we not update `program` here?

            previous_input = command.output
            i += 1

        self.execution_details.append(stack_trace)

        return program
