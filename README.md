# ai-tools
大语言模型相关的工具集合，包括学习资料，高效训练工具，开源数据集等相关内容

模型训练
改进版的lora训练代码，融合和参考了以下知名训练代码：
斯坦福的lora训练代码（Alpaca模型）https://github.com/tatsu-lab/stanford_alpaca
FastChat的训练代码（Vircuna）https://github.com/lm-sys/FastChat
Qlora原版的训练代码 https://github.com/artidoro/qlora

参考借鉴了以下知名开源项目
Chinese-Vicuna  https://github.com/Facico/Chinese-Vicuna/
Chinese-LLaMA-Alpaca https://github.com/ymcui/Chinese-LLaMA-Alpaca
Lightning-AI https://github.com/Lightning-AI

改进的重要地方
混乱的BOS，EOS，PAD token的纠正，这个很关键，这些TOken错误，导致训练结果出现严重问题
Tokenlized的过程改进，包括计算效率的改进，正确拼接模板，正确加BOS，EOS以及Pad　Token的逻辑，
其中，ｉｎｐｕｔ部分不做Loss计算，也是关键的细节点，这些都是在大量研究的基础上纠正解决
Lora训练哪些模块的问题也彻底纠正澄清

给出模型的结构构造

训练过程

data/test-instruct.json是模拟的标准指令集数据，按照自己的要求增加训练数据即可
tran.sh是训练脚本，相关重要参数都给出了，其中
template是Prompt模板，默认是ALPACA_PROMP，可以在代码prompt.py里增加自己的模板提示语
gradient_accumulation_steps与micro_batch_size是影响内存的最重要两个参数，如果显卡内存大，训练数据比较短，可以增加这两个参数，
特别是micro_batch_size，它表示一次训练多少条数据，gradient_accumulation_steps表示累计多少个micro_batch_size计算一次梯度，也是省内存

1 标准Lora训练命令
cd %panyun%

PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:24 \
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0  \
python src/panyun/lora.py \
--base_model /models/llama-2-13b-hf \
 --output_dir /models/llama-2-13b-hf-delta-kem  \
 --data_path data/test-instruct.json \
 --learning_rate: 0.0001 \
 --fp16 True \
 --source_max_len 512 \
 --target_max_len 1024 \
 --gradient_accumulation_steps 8 \
 --micro_batch_size 32\
 --lora_r 8   \
 --save_steps 50   \
 --lora_target_modules '[q_proj,v_proj,k_proj]'

2 如果要用8bit量化来节省内存，则增加参数
--load_in_8bit True

3 如果要用谷歌的Lion优化器，可增加下面的参数 
--optim=paged_lion_32bit 或者 paged_lion_8bit

标准qlora训练命令
增加参数 
--load_in_4bit True





训练数据的Token长度
了解训练数据Token以后的长度，包括中位数，最大长度，最小长度信息很重要，用于设置Max source Length与Max target Length参数
下面的脚本可实现这个功能
CUDA_VISIBLE_DEVICES=1  python src/panyun/analyse-data.py --base_model /models/vicuna-7b-v1.3  --data_path data/test-instruct.json  --template ALPACA_PROMP

程序方式生成instruct指令的训练数据
instruct指令的数据如果有有以下特点，可以借助这里提供的工具来实现
1 有大量可变参数

比如 请生成一封邮件：有三个参数：主题，收件人，邮件内容，
则可以建模如下：
主题：[代办任务，日报】
收件人：【王总，李总，王总】
邮件内容：【aaaaaaaa,bbbbb】

这种情况就可自动生成2*3*2=12条指令，比手写要来的快多了，参数如果比较多，或者参数选项比较多，则很容易自动生成上万条训练数据

 template2dataset.py 代码在这里，用了Python的模板技术，实现方式如下：
 一个模板文件，记录了生成怎样格式的训练数据，包括参数引用
 一个对应的模板参数文件，JSON格式的，记录了模板中所用的所有参数
 

