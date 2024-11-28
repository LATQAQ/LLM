import pandas as pd
import json
import os
import time
import re
from typing import List, Dict
from openai import OpenAI
import logging
import backoff
from concurrent.futures import ThreadPoolExecutor, as_completed


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从环境变量中获取 API 密钥
api_key = "sk-0b2d1ed3c2ee4168ad9189bcde52b1b7"

# 初始化 OpenAI 客户端
client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key)

def read_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def generate_single_entry(text: str) -> Dict:
    prompt = f"""
    基于以下 DND 文本内容，生成 1 个高质量的指令数据集条目。条目应该直接关联到给定的 DND 内容，提出与角色、情节、世界观或游戏规则相关的问题或任务。
    确保生成的指令类型丰富且多样化，包括但不限于以下类型：
    - **故事生成**: "根据内容，编写一个...的故事"
    - **情景设计**: "设计一个冒险场景，其中包含..."
    - **角色分析**: "分析...的性格动机或能力特性"
    - **规则解释**: "解释 DND 中...的具体规则和应用"
    - **问答类**: "在...场景中，为什么..."
    - **策略建议**: "为玩家提供在...情境下的最佳策略建议"

    文本内容：
    {text}

    按以下格式生成条目，确保所有字段内容准确、详细，且以 JSON 格式输出：
    {{
        "instruction": "清晰描述任务或问题",
        "input": "如果需要额外的上下文信息，请在这里提供，否则留空",
        "output": "对指令的详细回答或完成结果"
    }}

    请确保：
    1. 生成的条目与 DND 的世界设定相关，具体到角色、地点、任务或规则。
    2. 条目具备创意且具有可玩性，适合用作训练数据。
    """


    try:
        response = client.chat.completions.create(
            model="qwen-max-latest",  # 尝试使用自己选择的 模型
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # 增加温度以提高多样性
            max_tokens=4098
        )
        logger.info(f"API 响应: {response.choices[0].message.content}")

        response.choices[0].message.content = response.choices[0].message.content.replace("\n", " ")

        json_match = re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL)
        if json_match:
            entry = json.loads(json_match.group())
            required_keys = ['instruction', 'input', 'output']
            if isinstance(entry, dict) and all(key in entry for key in required_keys):
                # 根据 input 是否为空来设置 text 字段
                if entry['input'].strip():
                    entry[
                        'text'] = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"
                else:
                    entry[
                        'text'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"

                logger.info("成功生成完整条目")
                return entry
            else:
                logger.warning("JSON 解析成功，但缺少必要字段")
                return {}
        else:
            logger.error("无法从API响应中提取有效的JSON")
            return {}

    except Exception as e:
        logger.error(f"生成条目时发生错误: {str(e)}")
        return {}

def process_file(file_path: str, entries_per_file: int) -> List[Dict]:
    dataset = []
    try:
        text = read_file(file_path)
        for j in range(entries_per_file):
            logger.info(f"  生成第 {j + 1}/{entries_per_file} 个条目")
            entry = generate_single_entry(text)
            if entry and all(key in entry for key in ['instruction', 'input', 'output', 'text']):
                dataset.append(entry)
                logger.info(f"  成功生成 1 个完整条目")
            else:
                logger.warning(f"  跳过不完整的条目")
            time.sleep(2)  # 在请求之间增加延迟到2秒
            
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时发生未知异常: {str(e)}")
    return dataset

def generate_dataset(folder_path: str, entries_per_file: int = 2) -> List[Dict]:
    dataset = []
    files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(".txt")]
    with ThreadPoolExecutor(max_workers=8) as executor:  # 调整 max_workers 数量以适应你的硬件资源
        futures = [executor.submit(process_file, file_path, entries_per_file) for file_path in files]
        for future in as_completed(futures):
            try:
                dataset.extend(future.result())
            except Exception as e:
                logger.error(f"处理未来任务时发生未知异常: {str(e)}")

    return dataset

def save_dataset(dataset: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=2)
    logger.info(f"数据集已保存到 {output_file}")




def json_to_parquet(json_file, parquet_file):
    """
    将 JSON 文件转换为 Parquet 格式，嵌套字段直接作为字符串存储。

    :param json_file: 输入的 JSON 文件路径
    :param parquet_file: 输出的 Parquet 文件路径
    """
    try:
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保 JSON 内容是列表
        if not isinstance(data, list):
            raise ValueError("JSON 文件内容需要是一个数组（列表）")

        # 将嵌套字段转换为字符串
        for item in data:
            for key, value in item.items():
                if isinstance(value, (dict, list)):
                    item[key] = json.dumps(value, ensure_ascii=False)

        print(data)

        # 转换为 DataFrame
        df = pd.DataFrame(data)

        # 写入 Parquet 文件
        df.to_parquet(parquet_file, engine='pyarrow', index=False)

        print(f"转换成功：{parquet_file}")

    except Exception as e:
        print(f"转换失败：{e}")


if __name__ == "__main__":
    # input_folder = "./Data/chunks"  # 指定输入文件夹路径
    # output_file = "instruction_dataset.json"

    # logger.info("开始生成数据集")
    # dataset = generate_dataset(input_folder, entries_per_file=2)
    # save_dataset(dataset, output_file)
    # logger.info(f"数据集已生成并保存到 {output_file}")
    # logger.info(f"共生成 {len(dataset)} 个有效条目")

    # 转换为 Parquet 格式

    input_file = "instruction_dataset.json"
    output_file = "instruction_dataset.parquet"

    with open(input_file, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
        json_to_parquet(input_file, output_file)

