
# nanoGPT

![nanoGPT](./assets/nanogpt.jpg)

这是目前最简单、最快的一个训练/微调中等大小的GPT仓库。该项目是对[minGPT](https://github.com/karpathy/minGPT) 项目的重构。项目虽然仍处于积极的开发阶段；但是目前其中的`train.py`文件在OpenWebText上已经可以复现GPT-2（124M）在8个A100（40GB）上训练四天的效果。此外，代码写的十分简单易读：`train.py`是一个300行的训练的模版，而 `model.py` 是一个300行的GPT模型定义的模版，该模板支持选择加载OpenAI的GPT-2权重。

![repro124m](assets/gpt2_124M_loss.png)

因为该仓库的代码实在非常简单，因此很容易根据您的需求进行应用、或者从零开始训练新模型以及微调预训练的检查点（例如，目前可用的最大的预训练模型是OpenAI开源的GPT-2 1.3B模型）。

## 安装条件

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依赖的库：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (用于加载 GPT-2 模型的检查点)
-  `datasets` for huggingface datasets <3 (如果您想下载并且处理OpenWebText数据集)
-  `tiktoken` for OpenAI's 最快的BEP编码算法 <3
-  `wandb` for optional logging <3
-  `tqdm` for 加载进度条 <3

## 简易上手

如果您不是专业从事于深度学习行业的人士，只是想感受GPT的魅力以及自己训练GPT的快感，那么最快的入门方法就是训练一个可以创造莎士比亚作品的GPT。首先，我们先下载一个1MB大小的文件，并将其从原始文本转换为一个大的整数流（将原始数据转化为一个embedding之后的整数流数据）：

```
$ python data/shakespeare_char/prepare.py
```

这会在数据目录中创建`train.bin` and `val.bin` 文件，现在就是时候来训练你的GPT了。而你训练GPT的规模在很大程度上取决于你能提供的算力情况(是否有显卡以及显存的大小是多少)：

如果您的设备**配置了显卡**，那么很棒，我们可以使用在[config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file提供的设置很快地训练一个小型GPT

```
$ python train.py config/train_shakespeare_char.py
```

如果您恰好有兴趣观看了代码的细节部分，您将会发现我们正在训练一个上下文大小高达256个字符、384个特征通道，的一个6层的且每层的heads数量有6个的Transformer。在一个A100型号的显卡中，此训练运行大约需要3分钟，最佳验证损失可以减小到1.4697。根据代码中的设置，模型检查点被写入`--out_dir`目录`out-shakespeare-char`中。因此在训练完成后，我们可以通过将验证脚本`sample.py`在最佳的模型进行文本的效果生成。

```
$ python sample.py --out_dir=out-shakespeare-char
```

这会产生一些新的样例，例如:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

哇，`¯\_(ツ)_/¯`. 对于一个只在GPU上训练了三分钟的用于角色扮演的GPT来说还真不错。 而如果是在此数据集上微调预训练好的GPT-2模型，有很大的可能会获得更好的结果的（详情见后面的微调部分）

如果**您只有一台MacBook或者其他更为便宜的电脑(没有显卡)**怎么办？不用慌 ,我们仍然可以训练一个GPT,只是我们要让事情低调一点，在这里我推荐使用最新版的每晚更新的pytorch，因为这有可能让您的代码变得更加高效。当没有显卡的时候，可以使用下面这个简单的训练脚本。

```
$ python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

此时，由于我们是在CPU而非GPU上运行，我们必须设置`--device=cpu`并且使用`--compile=False`关闭掉PyTorch 2.0版本的compile功能。 然后，当我们在测试时候时，得到的估计结果可以更不精确但是更快（`eval_iters=20`，将迭代的次数从200下降到20），此时我们使用的上下文大小范围只有64个字符，而不是256个字符，每次迭代的batchsize只有12个样本，而不是64个样本。我们还将使用一个更小的Transformer结构（4layer，4个head，128个嵌入大小），并将迭代次数响应地减少到2000（相应地，通常使用`--lr_decay_iters`将学习率衰减到max_iters左右）。

```
$ python sample.py --out_dir=out-shakespeare-char --device=cpu
```
这样会生成下面这样的样本：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

因为是只在cpu上训练了3分钟，因此能输出这样符合语法，和要求格式的样子算很不错了。如果您乐意等待更长的时间，请尽情去调整超参数，并且去增加网络的大小，使用`--block_size`去调整上下文的长度

最终，如果是在苹果的M系列处理器上进行实验(最新的pytorch版本)，请务必将添加`--device=mps`的设置，PyTorch会使用在芯片上的图形处理单元(GPU)来显著加快训练速度(2-3倍)，并且允许您使用更大的网络规模。您可以在 [Issue 28](https://github.com/karpathy/nanoGPT/issues/28)找到更多相关的内容

## 复现GPT-2

一些更专业的深度学习从业者也许会更关注如何去复现GPT-2的效果。因此，让我们开始，我们首先将数据集进行tokenize，以[OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)数据集(OpenAI的WebText的开放版本)为例：

```
$ python data/openwebtext/prepare.py
```

这会下载并将 [OpenWebText](https://huggingface.co/datasets/openwebtext) 数据集进行分词化.这将会创造一个保存了`train.bin`和`val.bin`

```
$ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

如果您有集群环境可以使用，并且您拥有多个GPU节点，您可以使GPU在2个节点上进行，例如:

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

在您的计算机系统中对内部连接（例如使用iperf3工具）进行性能测试是一个很好的主意。如果您没有安装Infiniband技术，那么在上述启动命令前还应该添加`NCCL_IB_DISABLE=1`。这样您的多节点训练可以正常进行，但很可能速度会非常慢。默认情况下，检查点会定期写入到`--out_dir`指定的目录中。我们可以通过简单地执行`$ python sample.py`命令从模型中进行抽样。

最终，如果想在单个GPU上进行训练，您只需要简单的运行`$ python train.py`脚本即可。看看args中的这些参数，着脚本是如此的易读，易用以及是可迁移的。您可以根据您的需要在这些变量中进行随意地调整。

## baselines

OpenAI GPT-2 提供的一些模型转台允许我们checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

OpenAI GPT-2的提供的模型保存点允许我们为openwebtext数据集建立一些基准测试。我们可以按照以下方式获得这些数值：

```
$ python train.py eval_gpt2
$ python train.py eval_gpt2_medium
$ python train.py eval_gpt2_large
$ python train.py eval_gpt2_xl
```

并且可以观察到在训练集和测试集上的损失:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

然而，我们必须要注意到一点，GPT-2实际上是在未开源的，甚至是从未发行过的WebText数据集上进行训练的，而OpenWebText只是对这个数据集的最大努力地复制。这意味着数据集之间的差异是显著存在的。事实上，我们使用GPT-2 (124M) 的检查点，利用OpenWebText数据集进行微调，不一会就能观察到损失会降低到2.85左右，对与复现来说，这是实际上是更为合适的基线。

## 模型微调

微调和训练的差别并不大，我们只需要保证要从一个预训练的模型进行初始化并且用一个更小的学习率进行训练即可。如果想了解如何在新的数据集上微调GPT模型，您可以参考shakespeare的例子：转到`data/shakespeare`目录并运行`prepare.py`，以下载tiny shakespeare数据集并使用GPT-2的OpenAI BPE分词器，将其处理成`train.bin`和`val.bin`文件。不像使用OpenWebText数据集从零开始训练，这会在几秒钟内完成。微调可能需要很少的时间，甚至是在单个GPU上仅需几分钟。我们可以像这样运行一个微调的例子：

```
$ python train.py config/finetune_shakespeare.py
```

这将加载写在`config/finetune_shakespeare.py`中的配置参数（虽然我没有太多调整它们）。基本上，我们使用`init_from`从一个GPT2的模型保存点初始化，并像通常一样进行训练，只是训练时间更短，学习率更小。如果您在训练过程中发现显存溢出的情况，您可以尝试减小模型大小（可选的有`'gpt2'`, `'gpt2-medium'`, `'gpt2-large'`, `'gpt2-xl'`），或者可能减小`block_size`（上下文长度）。最佳的模型保存点（验证损失最低）会在`out_dir`目录中，根据配置文件例如默认情况下会保存在`out-shakespeare`中。然后，你可以运行在`sample.py --out_dir=out-shakespeare`中的代码：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Emmm，此时GPT似乎产生了一些不能解释（黑洞）的东西。我并没有真正对配置中的超参数进行太多调整，请随时尝试！

## sampling / inference

使用`sample.py`脚本可以从OpenAI发布的预训练GPT-2模型或者从您自己训练好的模型中进行内容生成。例如这里您可以通过这样的方式从目前最大的`gpt2-xl` 模型中进行推理：

```
$ python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. `$ python sample.py --start=FILE:prompt.txt`.

## efficiency notes

对于简单的模型基准测试和性能分析，`bench.py`可能会很有用。它与`train.py`中训练循环的核心部分所发生的事情相同，但省略了许多其他一些复杂的东西。

请注意，默认情况下代码使用的是[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)。在撰写本文时（2022年12月29日），这使得`torch.compile()`在每日更新版本中可用。`torch.compile()`所带来的提升是显著的，例如，将迭代时间从约250毫秒/次迭代减少到135毫秒/次迭代。PyTorch团队牛逼！

## 可以做的事情

- 进行调研并使用FSDP代替DDP
- 在标准评估上评估了零样本的困惑度（例如LAMBADA? HELM? 等）
- 可以微调 finetuning script，我认为超参数设置得不太好
- 在训练过程中安排线性增加批量大小
- 合并其他嵌入（如旋转嵌入，alibi嵌入）
- 我认为应该在检查点中将优化缓冲区与模型参数分开
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## 警告

请注意，默认情况下这个仓库使用PyTorch 2.0（即`torch.compile`）。这是相对较新和实验性的功能，且尚未在所有平台（例如Windows）上提供。如果你遇到相关的错误信息，尝试通过添加`--compile=False`标志来禁用此功能。这将减慢代码运行速度，但至少可以运行。对于这个仓库中的一些内容，如GPT，和语言模型等等，也许去观看我的 [Zero To Hero series](https://karpathy.ai/zero-to-hero.html)是有帮助的。此外，[GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) 目前是很受欢迎，如果您之前有语言建模方面的知识先验的话。

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

对于更多问题/讨论，请在Discord的 **#nanoGPT**话题进行讨论

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 感谢

所有nanoGPT的实验算力由 [Lambda labs](https://lambdalabs.com)提供, 我最爱的云服务器算力提供者.感谢你们对于nanoGPT项目的资助。
