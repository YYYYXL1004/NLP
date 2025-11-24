import torch
import json
import os
import numpy as np

def explore_data():
    # 路径设置
    word2vec_path = './data/word2vec.pt'
    dataset_path = './data/open_qa.jsonl'
    
    print(f"正在加载词向量: {word2vec_path} ...")
    if not os.path.exists(word2vec_path):
        print(f"Error: 文件不存在 {word2vec_path}")
        return

    # 加载 word2vec
    try:
        idx_to_word, word_to_idx, word2vec, idx_to_freq = torch.load(word2vec_path)
        
        print("\n=== Word2Vec 数据结构分析 ===")
        print(f"word2vec tensor shape: {word2vec.shape}")
        
        # 推理变量含义
        print(f"\nidx_to_word 类型: {type(idx_to_word)}")
        if isinstance(idx_to_word, list):
            print(f"idx_to_word 前5个元素: {idx_to_word[:5]}")
        elif isinstance(idx_to_word, dict):
             print(f"idx_to_word 前5个元素: {list(idx_to_word.items())[:5]}")
             
        print(f"\nword_to_idx 类型: {type(word_to_idx)}")
        if isinstance(word_to_idx, dict):
            print(f"word_to_idx 前5个元素: {list(word_to_idx.items())[:5]}")
            
        print(f"\nidx_to_freq 类型: {type(idx_to_freq)}")
        if isinstance(idx_to_freq, list):
             print(f"idx_to_freq 前5个元素: {idx_to_freq[:5]}")
    except Exception as e:
        print(f"加载 word2vec 出错: {e}")
        return

    # 读取 open_qa 数据集样例
    print(f"\n正在读取数据集: {dataset_path} ...")
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 2:
                    data = json.loads(line)
                    print(f"\n数据样例 {i+1}:")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                else:
                    break
    else:
        print(f"Error: 文件不存在 {dataset_path}")

if __name__ == "__main__":
    explore_data()
