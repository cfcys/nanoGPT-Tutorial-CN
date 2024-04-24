<h1 align="center">
  NanoGPT中文Tutorial
</h1>
<p align="center" width="100%">
  <img src="assets/Logo.png" alt="Nano" style="width: 100%; display: block; margin: auto;"></a>
</p>
<p align="center">
  <font face="楷体" color=Red size="6"> 对新手更友好的中文NanoGPT教程 </font>
</p>

**该仓库在持续更新中.**

<p align="center">
   📺 <a href="https://bilibili.com" target="_blank">BiliBili</a>
   🌐:<a href="https://youtube.com" target="_blank">youtube</a> 
</p> 

</br></br>


## 🗂️ 目录
- [📌 关于本教程](#-llama中文社区)


- [🍄 中文版nanoGPT项目readme](#-如何使用llama模型)

+ [🤖 中文预训练数据集搜集](#-模型预训练)
+ [💡 超级详细的nanoGPT视频教程](#-模型微调)
* [💪 对话能力实现](#-模型评测)
  
* [📖 目前已有的nanoGPT资料收集 ](#-学习中心)
    * [📺 视频](#-致谢)
    * [💻 博客](#-问题反馈)


- [📌 其它](#-其它)
  * [🎉 致谢](#-致谢)
  * [🤔 问题反馈](#-问题反馈)

## 📌 关于本教程

### 🔥 为什么还要去做nanoGPT的中文版教程

作为karpathy大神开源的目前已经获得三万多⭐的项目，nanoGPT是每个NLP爱好者入门必修项目。我必须承认karpathy在其Youtube视频课程中已经讲述的十分清楚，但是国内目前公开的视频网站或者博客，大多是将原版教学视频进行机翻，亦或者将直接对成型项目的代码进行逐行解读；鲜有对于下游任务的扩展以及整体项目的讲解。

### 🎯 我会去做什么

我会对目前已有的资料进行整理，同时发布一个相对全面的教学视频


# 🔵 中文版nanoGPT项目readme

<p align="left">
   我对nanoGPT项目进行了 <a href="README.md">中文</a>
</p>



## 📌 如何使用Llama模型?


你可以选择下面的快速上手的任一种方式，开始使用 Llama 系列模型。推荐使用[中文预训练对话模型](#llama2中文预训练模型atom-7b)进行使用，对中文的效果支持更好。


### 快速上手-使用Anaconda

第 0 步：前提条件
- 确保安装了 Python 3.10 以上版本。

第 1 步：准备环境

如需设置环境，安装所需要的软件包，运行下面的命令。
```bash
git clone https://github.com/LlamaFamily/Llama-Chinese.git
cd Llama-Chinese
pip install -r requirements.txt
```

第 2 步：下载模型

你可以从以下来源下载Atom-7B-Chat模型。
- [HuggingFace](https://huggingface.co/FlagAlpha)
- [ModelScope](https://modelscope.cn/organization/FlagAlpha)
- [WideModel](https://wisemodel.cn/models/FlagAlpha/Atom-7B-Chat)

第 3 步：进行推理

使用Atom-7B-Chat模型进行推理
创建一个名为 quick_start.py 的文件，并将以下内容复制到该文件中。
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B-Chat',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Atom-7B-Chat',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

运行 quick_start.py 代码。
```bash
python quick_start.py
```

### 快速上手-使用Docker

详情参见：[Docker部署](https://github.com/LlamaFamily/Llama-Chinese/blob/main/docs/chat_gradio_guide.md)

第 1 步：准备docker镜像，通过docker容器启动[chat_gradio.py](../examples/chat_gradio.py)
```bash
git clone https://github.com/LlamaFamily/Llama-Chinese.git

cd Llama-Chinese

docker build -f docker/Dockerfile -t flagalpha/llama2-chinese:gradio .
```

第 2 步：通过docker-compose启动chat_gradio
```bash
cd Llama-Chinese/docker
doker-compose up -d --build
```

### 快速上手-使用llama.cpp
详情参见：[使用llama.cpp](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/CPU/ggml/README.md)

### 快速上手-使用gradio
基于gradio搭建的问答界面，实现了流式的输出，将下面代码复制到控制台运行，以下代码以Atom-7B-Chat模型为例，不同模型只需修改一下面的model_name_or_path对应的模型名称就好了😊
```
python examples/chat_gradio.py --model_name_or_path FlagAlpha/Atom-7B-Chat
```

### 快速上手-构建API服务
使用FastChat构建和OpenAI一致的推理服务接口。

<details>
第 0 步：前提条件

安装fastchat
```bash
pip3 install "fschat[model_worker,webui]"
```
第 1 步：启动Restful API

开启三个控制台分别执行下面的三个命令
- 首先启动controler
```bash
python3 -m fastchat.serve.controller \
--host localhost \
--port 21001
```

- 启动模型
```bash
CUDA_VISIBLE_DEVICES="0" python3 -m fastchat.serve.model_worker --model-path /path/Atom-7B-Chat \
--host localhost \
--port 21002 \
--worker-address "http://localhost:21002" \
--limit-worker-concurrency 5 \
--stream-interval 2 \
--gpus "1" \
--load-8bit
```

- 启动RESTful API 服务
```bash
python3 -m fastchat.serve.openai_api_server \
--host localhost \
--port 21003 \
--controller-address http://localhost:21001
```

第 2 步：测试api服务

执行下面的python代码测试上面部署的api服务
```python
# coding=utf-8
import json
import time
import urllib.request
import sys
import requests

def test_api_server(input_text):
    header = {'Content-Type': 'application/json'}

    data = {
          "messages": [{"role": "system", "content": ""}, {"role": "user", "content": input_text}],
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "max_tokens": 512, 
          "model": "LLama2-Chinese-13B",
          "stream" : False,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 1.2, 
          "frequency_penalty": 0.2,           
          "top_k": 50, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False,
          "logprobs": None
    }
    response = requests.post(
        url='http://127.0.0.1:21003/v1/chat/completions',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        result = json.loads(response.content)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result

if __name__ == "__main__":
    test_api_server("如何去北京?")
```

</details>



## 🤖 模型预训练
虽然Llama2的预训练数据相对于第一代LLaMA扩大了一倍，但是中文预训练数据的比例依然非常少，仅占0.13%，这也导致了原始Llama2的中文能力较弱。为了能够提升模型的中文能力，可以采用微调和预训练两种路径，其中：
- 微调需要的算力资源少，能够快速实现一个中文Llama的雏形。但缺点也显而易见，只能激发基座模型已有的中文能力，由于Llama2的中文训练数据本身较少，所以能够激发的能力也有限，治标不治本。

- 基于大规模中文语料进行预训练，成本高，不仅需要大规模高质量的中文数据，也需要大规模的算力资源。但是优点也显而易见，就是能从模型底层优化中文能力，真正达到治本的效果，从内核为大模型注入强大的中文能力。

我们为社区提供了Llama模型的预训练代码，以及[中文测试语料](https://github.com/LlamaFamily/Llama-Chinese/tree/main/data)，更多数据可以参考[中文语料](#-中文数据)。具体代码和配置如下：
- 模型预训练脚本：[train/pretrain/pretrain.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/pretrain.sh)
- 预训练实现代码：[train/pretrain/pretrain_clm.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/pretrain_clm.py)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)加速：
  - 对于单卡训练，可以采用ZeRO-2的方式，参数配置见 [train/pretrain/ds_config_zero2.json](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/ds_config_zero2.json)
  - 对于多卡训练，可以采用ZeRO-3的方式，参数配置见 [train/pretrain/ds_config_zero3.json](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/ds_config_zero3.json)
- 训练效果度量指标：[train/pretrain/accuracy.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/accuracy.py)

## 💡 模型微调

本仓库中同时提供了LoRA微调和全量参数微调代码，关于LoRA的详细介绍可以参考论文“[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)”以及微软Github仓库[LoRA](https://github.com/microsoft/LoRA)。

### Step1: 环境准备

根据[requirements.txt](https://github.com/LlamaFamily/Llama-Chinese/blob/main/requirements.txt)安装对应的环境依赖。

### Step2: 数据准备
在data目录下提供了一份用于模型sft的数据样例：
- 训练数据：[data/train_sft.csv](https://github.com/LlamaFamily/Llama-Chinese/blob/main/data/train_sft.csv)
- 验证数据：[data/dev_sft.csv](https://github.com/LlamaFamily/Llama-Chinese/blob/main/data/dev_sft.csv)

每个csv文件中包含一列“text”，每一行为一个训练样例，每个训练样例按照以下格式将问题和答案组织为模型输入，您可以按照以下格式自定义训练和验证数据集：
```
"<s>Human: "+问题+"\n</s><s>Assistant: "+答案
```
例如，
```
<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>
```

### Step3: 微调脚本

#### LoRA微调
LoRA微调脚本见：[train/sft/finetune_lora.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_lora.sh)，关于LoRA微调的具体实现代码见[train/sft/finetune_clm_lora.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_clm_lora.py)，单机多卡的微调可以通过修改脚本中的`--include localhost:0`来实现。

#### 全量参数微调
全量参数微调脚本见：[train/sft/finetune.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune.sh)，关于全量参数微调的具体实现代码见[train/sft/finetune_clm.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_clm.py)。


### Step4: 加载微调模型

#### LoRA微调
基于LoRA微调的模型参数见：[基于Llama2的中文微调模型](#基于llama2的中文微调模型)，LoRA参数需要和基础模型参数结合使用。

通过[PEFT](https://github.com/huggingface/peft)加载预训练模型参数和微调模型参数，以下示例代码中，base_model_name_or_path为预训练模型参数保存路径，finetune_model_path为微调模型参数保存路径。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel,PeftConfig
# 例如: finetune_model_path='FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
finetune_model_path=''  
config = PeftConfig.from_pretrained(finetune_model_path)
# 例如: base_model_name_or_path='meta-llama/Llama-2-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model =model.eval()
input_ids = tokenizer(['<s>Human: 介绍一下北京\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

#### 全量参数微调
对于全量参数微调的模型，调用方式同[模型调用代码示例](#模型调用代码示例)，只需要修改其中的模型名称或者保存路径即可。

## 🍄 模型量化
我们对中文微调的模型参数进行了量化，方便以更少的计算资源运行。目前已经在[Hugging Face](https://huggingface.co/FlagAlpha)上传了13B中文微调模型[FlagAlpha/Llama2-Chinese-13b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat)的4bit压缩版本[FlagAlpha/Llama2-Chinese-13b-Chat-4bit](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-4bit)，具体调用方式如下：

环境准备：
```
pip install git+https://github.com/PanQiWei/AutoGPTQ.git
```

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized('FlagAlpha/Llama2-Chinese-13b-Chat-4bit', device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Llama2-Chinese-13b-Chat-4bit',use_fast=False)
input_ids = tokenizer(['<s>Human: 怎么登上火星\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

## 🚀 部署加速
随着大模型参数规模的不断增长，在有限的算力资源下，提升模型的推理速度逐渐变为一个重要的研究方向。常用的推理加速框架包含 lmdeploy、TensorRT-LLM、vLLM和JittorLLMs 等。

### TensorRT-LLM
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main)由NVIDIA开发，高性能推理框架

详细的推理文档见：[inference-speed/GPU/TensorRT-LLM_example](https://github.com/LlamaFamily/Llama-Chinese/tree/main/inference-speed/GPU/TensorRT-LLM_example)

### vLLM
[vLLM](https://github.com/vllm-project/vllm)由加州大学伯克利分校开发，核心技术是PageAttention，吞吐量比HuggingFace Transformers高出24倍。相较与FasterTrainsformer，vLLM更加的简单易用，不需要额外进行模型的转换，支持fp16推理。

详细的推理文档见：[inference-speed/GPU/vllm_example](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/GPU/vllm_example/README.md)

### JittorLLMs
[JittorLLMs](https://github.com/Jittor/JittorLLMs)由非十科技领衔，与清华大学可视媒体研究中心合作研发，通过动态swap机制大幅降低硬件配置要求（减少80%）,并且Jittor框架通过零拷贝技术，大模型加载相比Pytorch开销降低40%，同时，通过元算子自动编译优化，计算性能提升20%以上。

详细的推理文档见：[inference-speed/GPU/JittorLLMs](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/GPU/JittorLLMs_example/README.md)

### lmdeploy
[lmdeploy](https://github.com/InternLM/lmdeploy/) 由上海人工智能实验室开发，推理使用 C++/CUDA，对外提供 python/gRPC/http 接口和 WebUI 界面，支持 tensor parallel 分布式推理、支持 fp16/weight int4/kv cache int8 量化。

详细的推理文档见：[inference-speed/GPU/lmdeploy_example](https://github.com/LlamaFamily/Llama-Chinese/tree/main/inference-speed/GPU/lmdeploy_example)

## 💪 外延能力

除了持续增强大模型内在的知识储备、通用理解、逻辑推理和想象能力等，未来，我们也会不断丰富大模型的外延能力，例如知识库检索、计算工具、WolframAlpha、操作软件等。
我们首先集成了LangChain框架，可以更方便地基于Llama2开发文档检索、问答机器人和智能体应用等，关于LangChain的更多介绍参见[LangChain](https://github.com/langchain-ai/langchain)。

### LangChain
针对LangChain框架封装的Llama2 LLM类见[examples/llama2_for_langchain.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/examples/llama2_for_langchain.py)，简单的调用代码示例如下：
```python
from llama2_for_langchain import Llama2

# 这里以调用FlagAlpha/Atom-7B-Chat为例
llm = Llama2(model_name_or_path='FlagAlpha/Atom-7B-Chat')

while True:
    human_input = input("Human: ")
    response = llm(human_input)
    print(f"Llama2: {response}")
```

## 🥇 模型评测

### Llama2和Llama3对比评测
基础模型对比
<p align="center" width="100%">
<img src="./assets/base_eval.png" style="width: 100%; display: block; margin: auto;">
</p>
微调模型对比
<p align="center" width="100%">
<img src="./assets/tuned_eval.png" style="width: 100%; display: block; margin: auto;">
</p>

### Llama3模型评测
<p align="center" width="100%">
<img src="./assets/llama3_eval.png" style="width: 100%; display: block; margin: auto;">
</p>

### Llama2模型评测
<p align="center" width="100%">
<img src="./assets/llama_eval.jpeg" style="width: 100%; display: block; margin: auto;">
</p>

为了能够更加清晰地了解Llama2模型的中文问答能力，我们筛选了一些具有代表性的中文问题，对Llama2模型进行提问。我们测试的模型包含Meta公开的Llama2-7B-Chat和Llama2-13B-Chat两个版本，没有做任何微调和训练。测试问题筛选自[AtomBulb](https://github.com/AtomEcho/AtomBulb)，共95个测试问题，包含：通用知识、语言理解、创作能力、逻辑推理、代码编程、工作技能、使用工具、人格特征八个大的类别。

测试中使用的Prompt如下，例如对于问题“列出5种可以改善睡眠质量的方法”：
```
[INST] 
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. The answer always been translate into Chinese language.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

The answer always been translate into Chinese language.
<</SYS>>

列出5种可以改善睡眠质量的方法
[/INST]
```
Llama2-7B-Chat的测试结果见[meta_eval_7B.md](assets/meta_eval_7B.md)，Llama2-13B-Chat的测试结果见[meta_eval_13B.md](assets/meta_eval_13B.md)。

通过测试我们发现，Meta原始的Llama2 Chat模型对于中文问答的对齐效果一般，大部分情况下都不能给出中文回答，或者是中英文混杂的形式。因此，基于中文数据对Llama2模型进行训练和微调十分必要。


## 📖 学习中心

### 官方文档
Meta Llama全系列模型官方文档：https://llama.meta.com/docs/get-started

### Llama3
Llama 3官方链接：https://llama.meta.com/llama3

### Llama2

#### Meta官方对于[Llama2](https://ai.meta.com/llama)的介绍
自从Meta公司发布第一代LLaMA模型以来，羊驼模型家族繁荣发展。近期Meta发布了Llama2版本，开源可商用，在模型和效果上有了重大更新。Llama2总共公布了7B、13B和70B三种参数大小的模型。相比于LLaMA，Llama2的训练数据达到了2万亿token，上下文长度也由之前的2048升级到4096，可以理解和生成更长的文本。Llama2 Chat模型基于100万人类标记数据微调得到，在英文对话上达到了接近ChatGPT的效果。

### Llama相关论文
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)


## 📌 其它

### 🎉 致谢

感谢原子回声[AtomEcho](https://github.com/AtomEcho)团队的技术和资源支持！

感谢芯格[Coremesh](https://coremesh.net)团队的技术和资源支持！

感谢 @xzsGenius 对Llama2中文社区的贡献！

感谢 @Z Potentials社区对Llama2中文社区的支持！

### 🤔 问题反馈

如有问题，请在GitHub Issue中提交，在提交问题之前，请先查阅以往的issue是否能解决你的问题。

礼貌地提出问题，构建和谐的讨论社区。

加入[飞书知识库](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink)，一起共建社区文档。

加入微信群讨论😍😍

<p align="center" width="100%">
<img src="./assets/wechat.jpeg" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>

<p align="center" width="100%">
<img src="https://api.star-history.com/svg?repos=LlamaFamily/Llama-Chinese&type=Date" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>
