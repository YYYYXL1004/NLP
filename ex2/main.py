import torch
import json
import os
import re
import numpy as np
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORD2VEC_PATH = os.path.join(BASE_DIR, 'data/word2vec.pt')
DATASET_PATH = os.path.join(BASE_DIR, 'data/open_qa.jsonl')

def load_data():
    print(f"正在加载 {WORD2VEC_PATH} ...")
    idx_to_word, word_to_idx, word2vec, idx_to_freq = torch.load(WORD2VEC_PATH)
    print(f"词汇表大小: {len(idx_to_word)}")
    print("\n=== Word2Vec 变量分析 ===")
    # idx_to_word
    print(f"idx_to_word 类型: {type(idx_to_word)}")
    if isinstance(idx_to_word, list):
        print(f"样例 (前5个): {idx_to_word[:5]}")
    # word_to_idx
    print(f"word_to_idx 类型: {type(word_to_idx)}")
    if isinstance(word_to_idx, dict):
        print(f"样例 (前5个): {list(word_to_idx.items())[:5]}")
    # word2vec
    print(f"word2vec 类型: {type(word2vec)}")
    print(f"word2vec 尺寸: {word2vec.size()}")
    # idx_to_freq
    print(f"idx_to_freq 类型: {type(idx_to_freq)}")
    if isinstance(idx_to_freq, list):
        print(f"样例 (前5个): {idx_to_freq[:5]}")
    print("==================================\n")
    return word_to_idx, word2vec, idx_to_freq

def text_to_embedding(text, word_to_idx, word2vec, idx_to_freq, use_weighting=True):
    words = re.findall(r'\w+', text.lower())
    valid_vectors = []
    weights = []
    unknown_count = 0
    
    for w in words:
        if w in word_to_idx:
            idx = word_to_idx[w]
            vec = word2vec[idx]
            valid_vectors.append(vec)
            
            if use_weighting:
                # 频率越高的词(如 'the')，权重应该越低。
                # 这里用一个简单的平滑倒数公式: 1 / (freq + 10)
                # +10 是为了防止低频词权重过大导致方差失控
                freq = idx_to_freq[idx]
                weight = 1.0 / (freq + 10.0)
                weights.append(weight)
            else:
                weights.append(1.0)
        else:
            unknown_count += 1
            
    if not valid_vectors:
        return torch.zeros(word2vec.size(1)), 1.0 
    # Stack成矩阵: [seq_len, hidden_dim]
    valid_vectors = torch.stack(valid_vectors)
    # 计算加权平均
    weights_tensor = torch.tensor(weights).view(-1, 1) # [seq_len, 1]
    # 归一化权重，使其和为1
    weights_tensor = weights_tensor / weights_tensor.sum()
    # 加权和: (weights * vectors).sum(dim=0)
    sentence_vec = (valid_vectors * weights_tensor).sum(dim=0)
    # 计算 OOV Rate (未登录词占比)
    oov_rate = unknown_count / len(words) if words else 0
    return sentence_vec, oov_rate

def main():
    word_to_idx, word2vec, idx_to_freq = load_data()
    results = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            data = json.loads(line)
            q = data.get('question', '')
            h_ans = data.get('human_answers', [])
            h_text = " ".join(h_ans) if isinstance(h_ans, list) else str(h_ans)
            g_ans = data.get('chatgpt_answers', [])
            g_text = " ".join(g_ans) if isinstance(g_ans, list) else str(g_ans)
            if not h_text or not g_text: continue
            # 计算向量
            vec_h, oov_h = text_to_embedding(h_text, word_to_idx, word2vec, idx_to_freq)
            vec_g, oov_g = text_to_embedding(g_text, word_to_idx, word2vec, idx_to_freq)
            sim = F.cosine_similarity(vec_h.unsqueeze(0), vec_g.unsqueeze(0)).item()
            results.append({
                'id': line_num,
                'question': q,
                'sim': sim,
                'oov_avg': (oov_h + oov_g) / 2, # 平均 OOV 率
                'h_text_preview': h_text[:50],
                'g_text_preview': g_text[:50]
            })
    results.sort(key=lambda x: x['sim'])
    sims = [r['sim'] for r in results]
    oovs = [r['oov_avg'] for r in results]
    
    print(f"分析报告 (Analysis Report)")
    print("="*40)
    print(f"样本总数: {len(results)}")
    print(f"平均相似度 (Mean Cosine Similarity): {np.mean(sims):.4f}")
    print(f"平均 OOV 率 (Out-of-Vocabulary Rate): {np.mean(oovs)*100:.2f}%")
    print("-" * 40)
    
    # 打印极值 Case，用于定性分析
    print("\n[案例分析: 差异最大 (Least Similar)]")
    worst = results[0]
    print(f"问题: {worst['question']}")
    print(f"人类回答: {worst['h_text_preview']}...")
    print(f"AI 回答:    {worst['g_text_preview']}...")
    print(f"相似度得分: {worst['sim']:.4f}")
    
    print("\n[案例分析: 最相似 (Most Similar)]")
    best = results[-1]
    print(f"问题: {best['question']}")
    print(f"人类回答: {best['h_text_preview']}...")
    print(f"AI 回答:    {best['g_text_preview']}...")
    print(f"相似度得分: {best['sim']:.4f}")

if __name__ == "__main__":
    main()