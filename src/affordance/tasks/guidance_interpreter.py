import parsimonious


def search_engine(query):
    pass


def code_generate(instruction, input):
    pass


def code_execute(code_snippet):
    pass


def arithmetic(equation):
    pass


# Scott's grammars
grammar = parsimonious.grammar.Grammar(
    r"""
template = template_chunk*
template_chunk = command / command_block / content
command = command_start command_content command_end
command_block = command_block_open template command_block_close
command_block_open = command_start "#" block_command_call command_end
command_block_close = command_start "/" command_name command_end
command_start = "{{" "~"?
command_end = "~"? "}}"
command_contents = ~'[^{]*'
block_command_call = command_name command_args
command_content = command_call / variable_ref
command_call = command_name command_args
command_args = command_arg_and_ws+
command_arg_and_ws = ws command_arg
command_arg = named_command_arg / positional_command_arg
positional_command_arg = command_arg_group / variable_ref / literal
named_command_arg = variable_name "=" (variable_ref / literal)
command_arg_group = "(" command_content ")"
ws = ~'\s+'
command_contentasdf = ~"[a-z 0-9]*"i
command_name = ~"[a-z][a-z_0-9\.]*"i
variable_ref = ~"[a-z][a-z_0-9\.]*"i
variable_name = ~"[a-z][a-z_0-9\.]*"i
content  = query_content separator answer_content
answer_content = ~"[^{]*"
query_content = ~"[^{]*"
separator = "[\n]"
literal = ~'"[^\"]*"' / ~"'[^\']*'"
"""
)

test_prompt = """
{{tag}} What are the entities in this sentence?
President George H. W. Bush
Gulf War"""


test_2 = """{{description}} An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
{{input}} President George H. W. Bush called his generals to the Oval Office at the outset of the Gulf War.
{{tag}} What are the entities in this sentence?
President George H. W. Bush
Gulf War
{{search}} When was President George H. W. Bush president?
#2: George H. W. Bush's tenure as the 41st president of the United States began with his inauguration on January 20, 1989, and ended on January 20, 1993.
{{search}} When was the Gulf War fought?
The Gulf War[b] was a 1990–1991 armed campaign waged by a 35-country military coalition in response to the Iraqi invasion of Kuwait.
{{compare timeperiods}} Could these entities have co-existed and interacted based on this information?
Yes. Their time periods intersect.
{{answer}} Is this an anachronism?
No
{{eoc}}
No
----
{{description}} An anachronism is a mistake in chronology, or a person, thing, or event that is out of its proper time. Figure out whether a sentence contains anachronisms or not, and answer 'Yes' or 'No'."
{{input}} Kurt Cobain starred in the 1980 television show "Twin Peaks".
{{tag}} What are the entities in this sentence?
Kurt Cobain
"Twin Peaks"
{{search}} When did television show "Twin Peaks" air?
Twin Peaks is an American mystery serial drama television series created by Mark Frost and David Lynch. It premiered on ABC on April 8, 1990, and originally ran for two seasons until its cancellation in 1991.
{{search}} When did Kurt Cobain live?
Kurt Donald Cobain (February 20, 1967 – c. April 5, 1994) was an American musician, best known as the lead vocalist, guitarist and primary songwriter of the ...
{{compare timeperiods}} Could these entities have co-existed and interacted based on this information?
No. Musician  Kurt Cobain could not have starred in Twin Peaks.
{{answer}} Is this an anachronism?
Yes
{{eoc}}
Yes
----"""

print(grammar.parse(test_prompt))


class Interpreter:
    def __init__():
        pass


class PromptCompletion:
    """Represents the result of an executed prompt."""

    def __init__(self, variables, completed_text, prompt):
        self.variables = variables
        self.completed_text = completed_text
        self.prompt = prompt

    def __getitem__(self, key):
        return self.variables[key]

    def __repr__(self):
        return self.completed_text


class Prompt:
    """A prompt template that can be compiled and executed to generate a PromptCompletion result."""

    def __init__(self, prompt, generator=None):
        """Create a new Prompt object from a prompt string."""
        self.prompt_string = prompt
        self.generator = generator
        self.tree = grammar.parse(self.prompt_string)

    def __call__(self, **kwargs):
        built_ins = {
            "search": search_engine,
            "code generate ": code_generate,
            "code execute": code_execute,
        }
        variables = {}
        variables.update(built_ins)
        variables.update(kwargs)
        vi = TopDownVisitor(variables, self)

        output = vi.visit(self.tree)

        return PromptCompletion(variables, output, self)


class TopDownVisitor(Interpreter):
    def __init__():
        pass

    def syntax_check(program):
        # Look for programs with EOC ending and final answer in syntax.
        pass

    def visit(self, node):
        # Parse a fixed NL Program.
        pass

    def visit_update(self, node):
        # Parse a dynamically changing NL Program.
        pass
