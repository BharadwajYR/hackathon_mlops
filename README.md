---
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: nvidia-open-model-license
license_link: https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
---


# Hymba-1.5B-Base

## Model Overview

Hymba-1.5B-Base is a base text-to-text model that can be adopted for a variety of natural language generation tasks.

The model has hybrid architecture with Mamba and Attention heads running in parallel. Meta tokens, a set of learnable tokens prepended to every prompt, help improve the efficacy of the model. The model shares KV cache between 2 layers and between heads in a single layer. 90% of attention layers are sliding window attention. 

This model is ready for commercial use.

**[Caution] During generation, the batch size needs to be 1. Our current implementation does not fully support padding of Meta tokens + SWA; this is a work in progress. Training and pre-filling support any batch size.**


**Model Developer:** NVIDIA 

**Model Dates:** Hymba-1.5B-Base was trained between September 1, 2024 and November 10th, 2024.

**License:**
This model is released under the [NVIDIA Open Model License Agreement](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf).


## Model Architecture

Hymba-1.5B-Base has a model embedding size of 1600, 25 attention heads, and an MLP intermediate dimension of 5504, with 32 layers in total, 16 SSM states, 3 full attention layers, the rest are sliding window attention. Unlike the standard Transformer, each attention layer in Hymba has a hybrid combination of standard attention heads and Mamba heads in parallel.  Additionally, it uses Grouped-Query Attention (GQA) and Rotary Position Embeddings (RoPE). 

Features of this architecture:

- Fuse attention heads and SSM heads within the same layer, offering parallel and complementary processing of the same inputs.

<div align="center">
<img src="https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/images/module.png" alt="Hymba Module" width="600">
</div>

- Introduce meta tokens that are prepended to the input sequences and interact with all subsequent tokens, thus storing important information and alleviating the burden of "forced-to-attend" in attention.

- Integrate with cross-layer KV sharing and global-local attention to further boost memory and computation efficiency.

<div align="center">
<img src="https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/images/macro_arch.png" alt="Hymba Model" width="600">
</div>


## Performance Highlights
- Hymba-1.5B-Base outperforms all sub-2B public models.

<div align="center">
<img src="https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/images/performance1.png" alt="Compare with SoTA Small LMs" width="800">
</div>

<div align="center">
<img src="https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/images/performance2.png" alt="Compare with SoTA Small LMs" width="800">
</div>


## Model Usage


### Step 1: Environment Setup

Since Hymba-1.5B-Base employs [FlexAttention](https://pytorch.org/blog/flexattention/), which relies on Pytorch2.5 and other related dependencies, we provide two ways to setup the environment:

- **[Local install]** Install the related packages using our provided `setup.sh` (support CUDA 12.1/12.4):

```
wget --header="Authorization: Bearer YOUR_HF_TOKEN" https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/setup.sh
bash setup.sh
```

- **[Docker]** A docker image is provided with all of Hymba's dependencies installed. You can download our docker image and start a container using the following commands:
```
docker pull ghcr.io/tilmto/hymba:v1
docker run --gpus all -v /home/$USER:/home/$USER -it ghcr.io/tilmto/hymba:v1 bash
```


### Step 2: Chat with Hymba-1.5B-Base
After setting up the environment, you can use the following script to chat with our Model

```py
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

# Load the tokenizer and model
repo_name = "nvidia/Hymba-1.5B-Base"

tokenizer = AutoTokenizer.from_pretrained(repo_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo_name, trust_remote_code=True)
model = model.cuda().to(torch.bfloat16)

# Chat with Hymba
prompt = input()
inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
outputs = model.generate(**inputs, max_length=64, do_sample=False, temperature=0.7, use_cache=True)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"Model response: {response}")

```


## Limitations

The model was trained on data that contains toxic language, unsafe content, and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

The testing suggests that this model is susceptible to jailbreak attacks. If using this model in a RAG or agentic setting, we recommend strong output validation controls to ensure security and safety risks from user-controlled model outputs are consistent with the intended use cases.

## Ethical Considerations 
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. 
Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/). 


## Citation
```
@misc{dong2024hymbahybridheadarchitecturesmall,
      title={Hymba: A Hybrid-head Architecture for Small Language Models}, 
      author={Xin Dong and Yonggan Fu and Shizhe Diao and Wonmin Byeon and Zijia Chen and Ameya Sunil Mahabaleshwarkar and Shih-Yang Liu and Matthijs Van Keirsbilck and Min-Hung Chen and Yoshi Suhara and Yingyan Lin and Jan Kautz and Pavlo Molchanov},
      year={2024},
      eprint={2411.13676},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.13676}, 
}