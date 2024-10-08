
# GPTQ Quantization Benchmarking of LLaMA 2 and Mistral 7B Models

## Overview

This project benchmarks the memory efficiency, inference speed, and accuracy of LLaMA 2 (7B, 13B) and Mistral 7B models using GPTQ quantization with 2-bit, 3-bit, 4-bit, and 8-bit configurations. The code evaluates these models on downstream tasks for performance assessment, including memory consumption and token generation speed.

## Requirements

To run the benchmarking, you need the following dependencies:

```bash
pip install --upgrade transformers auto-gptq accelerate datasets bitsandbytes
python -m pip install git+https://github.com/huggingface/optimum.git
python -m pip install git+https://github.com/huggingface/optimum-benchmark.git
```

### Additional Dependencies:

- `optimum-benchmark`: Used to track memory consumption and inference speed.
- `lm-evaluation-harness`: For benchmarking on specific tasks like Winogrande, HellaSwag, and Arc Challenge.

To install:

```bash
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
pip install bitsandbytes
```

## Usage

### Benchmarking Memory Efficiency and Inference

To benchmark memory usage and inference speed, modify the YAML configuration for the desired model and bit-width. An example for LLaMA-2 13B is shown below. Replace `Llama-2-13b-hf` with the specific model you want to evaluate.

```python
import os
for w in [2, 3, 4, 8]:
    YAML_DEFAULT = '''
    experiment_name: example/Llama-2-13b-hf-gptq-%sbit
    model: example/Llama-2-13b-hf-gptq-%sbit
    device: cuda
    benchmark:
      memory: true
      warmup_runs: 10
      new_tokens: 1000
      input_shapes:
        sequence_length: 512
        batch_size: 2
    ''' % (str(w), str(w))

    with open("llama2_13b_ob.yaml", 'w') as f:
        f.write(YAML_DEFAULT)
    os.system("optimum-benchmark --config-dir ./ --config-name llama2_13b_ob")
```

### Evaluating Model Performance on Specific Tasks

You can evaluate model performance using the `lm-eval` tool on various tasks. Example:

```bash
lm_eval --model hf --model_args pretrained=example/Mistral-7B-v0.1-gptq-4bit --tasks winogrande,hellaswag,arc_challenge --device cuda:0 --num_fewshot 5 --batch_size 2 --output_path ./eval_harness/Mistral-7B-v0.1-gptq-4bit
```

### Results

Results of the benchmarks will be stored in the `experiments_ob/` directory, including memory usage and speed.

## License

This project is licensed under the MIT License.
