# ART: Automatic multi-step reasoning and tool-use with large black-box language models

## Requirements

Python>=3.8
```
pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
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
