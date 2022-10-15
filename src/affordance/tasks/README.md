## Affordance Cascades

This directory contains exploration for automatic decompositions of tasks which tried to answer the following research questions:

For each tasks, we compute:
* Best human decomposition performance over N runs: Known decomps or ones that we come up with. A further variant of this is (a) Decompositing into individual GPT-3 calls with few-shot prompting (decompositional prompting) and (b) Making and integrating external affordance calls when needed.
* Automatic instruction generation (APE): Reporting on top-K instructions. APE reports average over top-10 for 200 instructions. They also have an efficient score estimation technique whereby promising candidates (evaluated based on a small subset) are given more compute resource. 
* Automatic decomposition generation, followed by zero-shot application to downstream task. Reporting average performance over top-k decompositions
* Automtic decomposition: Instruction refinement and decomposition ensembling
* Potential affordance calls and decompsoitions with those calls.
* Human conversations

Things to keep track of:
* Evaluation metric computation
* Generated sequence length
