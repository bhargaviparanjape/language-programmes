# ART: Automatic multi-step reasoning and tool-use with large black-box language models

## Requirements

Follow these instructions to create a conda environment that will work for this project:

```sh
conda create --name $ENV pytorch==1.12.1 pytorch-cuda==11.7 tk==8.6.12 -c pytorch -c nvidia
conda activate $ENV
pip install "bigbench @ https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"
pip install -r requirements.txt
```

## Instructions

Sore OpenAI Api key in home directory as filename `openai_api_key`

Run Few-shot (Direct) prompting:

```
cd src/affordance/tasks
python ${task_name}.py \
--model_name text-davinci-002 \
--inference_strategy few_shot
```

Run Auto-CoT prompting:

```
cd src/affordance/tasks
python ${task_name}.py \
--model_name text-davinci-002 \
--inference_strategy auto_cot
```


Run ART prompting:

```
cd src/affordance/tasks
python ${task_name}.py \
--model_name text-davinci-002 \
--inference_strategy nl_program
```

Model names can be changed to `code-davinci-002`. Use `--num_dev_examples 100` to quickly evaluate on fewer instances.