"""
文件用途: MiniMind 模型推理/对话脚本
主要功能:
1. 初始化并加载 MiniMind 模型（支持 LoRA、MoE 等配置）
2. 提供预设测试问题或用户手动输入
3. 使用 HuggingFace 的 TextStreamer 实时打印生成结果
4. 支持携带历史上下文对话（history_cnt）
核心作用：
  - 作为模型训练后的"使用接口"，验证预训练/SFT/LoRA微调等不同阶段模型的效果。
  - 支持不同配置（基础模型、LoRA适配器、MoE结构）的模型加载与测试。
  - 模拟真实对话场景，通过上下文管理实现多轮交互。
"""
import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    """
    初始化并加载 MiniMind 模型
    参数:
        args: argparse 解析后的命令行参数
    核心逻辑：
      - 支持两种加载模式：
        1) 原生torch权重（args.load=0）：加载预训练/SFT/RLHF等阶段保存的pth文件，
           可叠加LoRA适配器（通过apply_lora和load_lora实现）。
        2) transformers接口（args.load=1）：加载转换为HuggingFace格式的完整模型，
           兼容transformers生态的生成函数。
      - 自动适配MoE结构（根据args.use_moe）和不同模型规模（hidden_size/num_hidden_layers）。
    返回:
        - model: 已加载权重并切换到eval()的模型（禁用dropout等训练特有操作）
        - tokenizer: 对应的分词器（负责文本编码/解码）
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 初始化 MiniMind 模型配置与结构
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        # 模型模式与权重文件路径映射
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        # 加载保存好的权重
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        # 如果使用 LoRA，加载 LoRA 权重
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # 通过 transformers 接口加载完整模型（默认路径 ./MiniMind2）
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    # 打印模型参数量 (以百万为单位)
    # print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.half().eval().to(args.device), tokenizer

def main():
    """
    主函数入口：
    流程链条：
      1) 参数解析 → 2) 模型/分词器加载 → 3) 测试prompt生成 → 
      4) 交互模式选择（自动测试/手动输入） → 5) 对话上下文管理 → 
      6) 输入编码 → 7) 流式生成 → 8) 输出解码与上下文更新
    核心特性：
      - 流式输出（TextStreamer）：模拟人类打字效果，提升交互体验。
      - 上下文管理（history_cnt）：支持多轮对话，通过截断控制上下文长度。
      - 生成参数可调（temperature/top_p）：控制输出的随机性和多样性。
    """
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, rlhf, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    # 模型结构相关参数
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度（注意：并非模型实际长文本能力）")
    parser.add_argument('--temperature', default=0.85, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    # 对话历史参数
    # history_cnt 必须为偶数 (用户+助手为一组)
    parser.add_argument('--historys', default=0, type=int, help="携带历史对话轮数（需为偶数，0表示不携带历史）")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    # 初始化模型与分词器   
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    # HuggingFace 自带的实时输出器
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
		# 设置随机种子，保证每轮输出的随机性（或固定性）
		setup_seed(random.randint(0, 31415926))
        # setup_seed(2025)  # 如果需要固定结果，可写死种子
        if input_mode == 0: print(f'💬: {prompt}')
        # 维护历史上下文（如果 history_cnt > 0）
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            # 预训练模型只做简单的续写
            inputs = tokenizer.bos_token + prompt
        else:
            # 构建对话模板：
            # tokenizer.apply_chat_template 将历史消息转换为模型预期的格式（如添加角色标记<user>/<assistant>），
            # 不同模型可能需要不同模板，确保模型能正确区分对话角色和轮次。
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        # 编码输入
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)
        # 生成回答
        print('🧠: ', end='')
        st = time.time()
        # 生成回答后，可从以下维度评估：
        # 1) 相关性：回答是否紧扣问题，无冗余信息。
        # 2) 流畅性：语句是否通顺，无语法错误。
        # 3) 准确性：事实性内容是否正确（如医疗建议、知识问答）。
        # 4) 安全性：是否包含不当内容（针对对话模型）。
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        # 解码生成内容（去掉输入部分，只保留新生成的）
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()