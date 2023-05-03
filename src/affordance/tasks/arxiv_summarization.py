import argparse
import os
from typing import Tuple

import datasets
import evaluate
from prompt_library import llm_similar_tasks, random_tasks, similar_auto_breakdowns, similar_tasks
from sequential_interpreter import TopDownVisitor
import tiktoken
from tools import get_tool
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import OpenAIModel, cache_dir, chunks

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

rouge = evaluate.load("rouge")


def sent_clean(s: str) -> str:
    ret = ""
    nested = 0

    for i in s.strip():
        if i == "[":
            nested += 1
        elif i == "]" and nested > 0:
            nested -= 1
        elif nested == 0:
            ret += i

    return ret


def load_arxiv_sum(
    split: str = "train", sents: int = 15
) -> Tuple[datasets.Dataset, datasets.Dataset]:
    """Load the arXiv summarization dataset ("ccdv/arxiv-summarization") from Huggingface.

    Chunks the articles into 5-sentence-long chunks.
    """
    ds = datasets.load_dataset("ccdv/arxiv-summarization", split=split, cache_dir=cache_dir)

    inputs = ds["article"]
    labels = ds["abstract"]

    article_sentences = [[sent_clean(s) for s in t.split("\n")] for t in inputs]
    inputs = []
    for sentences in article_sentences:
        out = []
        for chunk in chunks(sentences, sents):
            out.append(" ".join(chunk))
        inputs.append(out)

    return inputs, labels


# TODO Only using 10 examples for now
dev_inputs, dev_labels = load_arxiv_sum(split="validation[:10]")

io_pairs = [
    ("""A: <article 1 from arxiv>""", "S: <summary 1 from arxiv>"),
    ("""A: <article 2 from arxiv>""", "S: <summary 2 from arxiv>"),
    ("""A: <article 3 from arxiv>""", "S: <summary 3 from arxiv>"),
]

task_name = "arXiv summarization"
task_description = "Summarize the given articles into an abstract"

prompt_filename = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "arxiv_summarization_prompt.txt"
)
with open(prompt_filename) as f:
    few_shot_cot_prompt = f.read().strip() + " "  # prompt ends with "Q1:", needs space


def format_prompt(prompt: str, i: int, n: int, text: str):
    """Format the prompt with the chunk and supporting information."""
    ins = (
        "Your goal at this stage is to store important information that you can recall later when "
        "you're producing the final summary after the last chunk. Use [read], [write], and "
        "[list-keys] to load/store any information you like as you process this chunk."
    )
    if i == n - 1:
        ins = (
            "Your goal for this final chunk is to leverage the information you stored previously "
            "in order to output a final summary of the entire text using [ans]."
        )

    return prompt.format(instructions=ins, i=i + 1, n=n, text=text)


def get_few_shot_cot_prompt(task_name: str, description: str, strategy: str) -> str:
    if strategy == "fixed":
        return few_shot_cot_prompt
    elif strategy == "random":
        return random_tasks(N=6)
    elif strategy == "similar":
        return similar_tasks(description, io_pairs, N=6)
    elif strategy == "similar_auto_decomp":
        return similar_auto_breakdowns(description, io_pairs, N=6)
    elif strategy == "llm_similar":
        return llm_similar_tasks(task_name, description, io_pairs, N=6)


tool_exclude = [
    "[arithmetic]",
    "[code edit]",
    "[code execute]",
    "[code generate]",
    "[generate]",
    "[search]",
    "[string edit]",
    "[string index]",
    "[string permutation]",
]


def visit(model: OpenAIModel, text: str, run_id: str, q: int = 1, max_runs: int = 5) -> str:
    """Successively evaluates the model to produce new lines until an [EOQ] is reached.

    The last line in text should be "Q{q}:", where q is the question number we're starting from.
    """
    print("NEW PROMPT:")
    print(text.rsplit("----", 1)[1].strip(), end="")

    for _ in range(max_runs):
        print(" AWAITING COMPLETION...", end="", flush=True)
        #print('\ntext length', len(text))
        completion = model(text)[0].strip()
        print(f"\rQ{q}: " + completion + " " * (len("AWAITING COMPLETION...") - len(completion)))
        if completion.startswith("[EOQ]") or completion.startswith("[ans]"):
            return text + completion
        spl = completion.split(" ", 1)
        tool_name = spl[0]
        args = ""
        if len(spl) > 1:
            args = spl[1]

        tool_func = get_tool(tool_name, tool_exclude)
        if tool_func is None:
            print(f"tool '{tool_name}' does not exist")

            continue  # retry

        # run the tool
        print(f"#{q}: RUNNING TOOL...", end="", flush=True)
        output, details = tool_func(args, "", run_id)

        print(f"\r#{q}: {output}" + " " * (len("RUNNING TOOL...") - len(output)))
        if details:
            print(f"tool '{tool_name}' details:", details)

        # update the text
        text += completion + f"\n#{q}: " + output + f"\nQ{q+1}: "
        print(f"Q{q+1}:", end="", flush=True)
        q += 1

    return text


answer_tool = "[ans]"


def get_answer(completion: str) -> str:
    """Extract the answer after [ans]."""
    try:
        idx = completion.rindex(answer_tool) + len(answer_tool)
    except ValueError:
        return None

    return completion[idx:].split("\n", 1)[0].strip()


def nl_program(model_name: str, temperature: float, strategy: str, run_title: str):
    prompt = get_few_shot_cot_prompt(task_name, task_description, strategy)

    n = 1
    runs = 1

    model = OpenAIModel(model_name, quote="\n", temperature=temperature, max_length=1000, n=n)

    # a list of article IDs to pass as run titles
    train_ids = [f"{run_title}_train_{i}" for i in range(len(dev_inputs))]

    perf_array = []
    with tqdm(total=runs * len(dev_inputs)) as pbar:
        for run in range(runs):
            pbar.set_description(f"run {run+1}/{runs}")

            answers = []

            for i in range(len(dev_inputs)):
                article = dev_inputs[i]
                article_id = train_ids[i]

                # run across all the chunks in order
                for j, chunk in enumerate(article):
                    chunk_prompt = format_prompt(prompt, j, len(article), chunk)

                    full_text = visit(model, chunk_prompt, article_id)

                answers.append(get_answer(full_text))

                pbar.update(1)

            perf_array.append(rouge.compute(references=dev_labels, predictions=answers))

    print("performance:", perf_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        choices=[
            "gpt-3.5-turbo",
            "text-davinci-002",
            "text-davinci-003",
            "code-davinci-002",
            "code-cushman-001",
            "davinci-codex-002-msft",
        ],
        default="gpt-3.5-turbo",
    )
    parser.add_argument("--temperature", type=float, default="0.3")
    parser.add_argument("--run-title", type=str, default="default")
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
        default="nl_program",
    )
    parser.add_argument(
        "--num-examples", type=int, default=len(dev_inputs), help="number of examples to run"
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["fixed", "random", "similar", "similar_auto_decomp", "llm_similar"],
        default="fixed",
        help="how to choose the few-shot examples in the prompt",
    )

    args = parser.parse_args()

    dev_inputs = dev_inputs[: args.num_examples]

    print("dataset statistics")
    print(task_description)
    print("dev examples:", len(dev_inputs))

    # for debugging: print the number of tokens in the bare prompt
    enc = tiktoken.encoding_for_model(args.model_name)
    print("number of tokens in fixed prompt:", len(enc.encode(few_shot_cot_prompt)))

    if args.inference_strategy == "nl_program":
        nl_program(
            args.model_name,
            args.temperature,
            strategy=args.selection_strategy,
            run_title=args.run_title,
        )
    else:
        raise NotImplementedError("only nl_program is implemented right now")
