import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    """
    处理单个数据项，内置了重试和错误处理逻辑。
    """
    max_retries = 3  # 最多重试3次
    
    for attempt in range(max_retries):
        try:
            # 正常调用API
            response: Structure = chain.invoke({
                "language": language,
                "content": item['summary']
            })
            
            # 检查API是否返回了空值 (处理 'NoneType' 错误)
            if response is None:
                print(f"Item {item['id']} received a None response. Marking as error.", file=sys.stderr)
                # 这是一个无法通过重试解决的问题，所以直接跳出循环
                break

            # 如果成功，赋值并直接返回结果
            item['AI'] = response.model_dump()
            return item

        except Exception as e:
            # 检查这个异常是不是速率限制错误
            if "Error code: 429" in str(e):
                wait_time = (attempt + 1) * 5  # 第一次等5秒, 第二次等10秒, ...
                print(f"Rate limit hit for item {item['id']}. Attempt {attempt + 1}/{max_retries}. Retrying in {wait_time}s...", file=sys.stderr)
                time.sleep(wait_time)
                # 等待后，继续下一次循环（重试）
                continue
            
            # 如果是其他任何类型的错误 (包括 OutputParserException)
            else:
                print(f"An unrecoverable error occurred for item {item['id']}: {e}", file=sys.stderr)
                # 这是一个无法通过重试解决的问题，所以直接跳出循环
                break

    # 如果循环结束（无论是重试用尽还是中途跳出），都说明处理失败
    # 给item赋一个默认的错误值，防止后续程序出错
    item['AI'] = {
        "tldr": "Error",
        "motivation": "Error",
        "method": "Error",
        "result": "Error",
        "conclusion": "Error"
    }
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # 保持原始数据
                processed_data[idx] = data[idx]
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
