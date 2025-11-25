import spacy
from spacy.tokens import Doc
import matplotlib.pyplot as plt
import os
from sklearn.metrics import ConfusionMatrixDisplay
import collections

# 解决绘图报错
plt.switch_backend('Agg')

CWS_FILE = "./data/中文分词.txt"
POS_FILE = "./data/词性标注.tsv"
NER_FILE = "./data/命名实体识别.tsv"
IMG_DIR = "images"
BAD_CASE_FILE = "bad_cases.txt"

if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

PKU_TO_CTB_MAP = {
    'n': 'NN', 'nr': 'NR', 'ns': 'NR', 'nt': 'NR', 'nz': 'NR',
    'v': 'VV', 'vn': 'VV', 'a': 'VA', 'd': 'AD', 'p': 'P',
    'c': 'CC', 'u': 'DEG', 'm': 'CD', 'q': 'M', 'r': 'PN',
    't': 'NT', 'f': 'LC', 'w': 'PU'
}

SPACY_TO_NER_MAP = {
    'PERSON': 'nr', 'ORG': 'nt', 'GPE': 'ns', 'LOC': 'ns'
}

def save_bad_case(task, text, gold, pred):
    with open(BAD_CASE_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{task}] 错误样本:\n")
        f.write(f"Text: {text}\n")
        f.write(f"Gold: {gold}\n")
        f.write(f"Pred: {pred}\n")
        f.write("-" * 30 + "\n")

def load_cws_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            words = line.split()
            data.append(("".join(words), words))
    return data

def load_pos_data(file_path):
    sentences = []
    words = []
    tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words = []
                    tags = []
                continue
            
            parts = line.split('\t')
            if len(parts) == 2:
                words.append(parts[0])
                tags.append(parts[1])
                
    if words:
        sentences.append((words, tags))
    return sentences

load_ner_data = load_pos_data

def plot_metrics(names, values, title, filename):
    plt.figure(figsize=(8, 6))
    colors = ['#8ecfc9', '#ffbe7a', '#fa7f6f', '#82b0d2']
    
    bar_colors = colors[:len(names)] if len(names) <= len(colors) else colors * (len(names) // len(colors) + 1)
    
    bars = plt.bar(names, values, color=bar_colors, width=0.5)
    plt.title(title)
    plt.ylim(0, 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=10)
                 
    plt.savefig(f"{IMG_DIR}/{filename}")
    print(f"图保存到了 {filename}")
    plt.close()

def get_intervals(words):
    intervals = set()
    idx = 0
    for word in words:
        intervals.add((idx, idx + len(word)))
        idx += len(word)
    return intervals

def evaluate_cws(nlp, data):
    print("正在评估分词...")
    
    if os.path.exists(BAD_CASE_FILE):
        os.remove(BAD_CASE_FILE)
        
    tp = 0
    fp = 0
    fn = 0
    
    for i, (raw_text, gold_words) in enumerate(data):
        doc = nlp(raw_text)
        pred_words = [t.text for t in doc]
        
        gold_inv = get_intervals(gold_words)
        pred_inv = get_intervals(pred_words)
        
        correct = len(gold_inv & pred_inv)
        tp += correct
        fp += len(pred_inv) - correct
        fn += len(gold_inv) - correct
        
        if gold_inv != pred_inv and i < 5:
            save_bad_case("分词", raw_text, gold_words, pred_words)
        
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"分词结果: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    plot_metrics(['Precision', 'Recall', 'F1'], [p, r, f1], 'CWS Metrics', 'cws.png')

def evaluate_pos(nlp, data):
    print("正在评估词性标注...")
    correct = 0
    total = 0
    
    all_gold = []
    all_pred = []
    
    for i, (words, gold_tags) in enumerate(data):
        doc = Doc(nlp.vocab, words=words)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
            
        pred_tags = []
        has_error = False
        
        for j, token in enumerate(doc):
            if j >= len(gold_tags): break
            gold = gold_tags[j]
            gold = PKU_TO_CTB_MAP.get(gold, gold)
            pred = token.tag_
            
            all_gold.append(gold)
            all_pred.append(pred)
            pred_tags.append(pred)
            
            if pred == gold:
                correct += 1
            else:
                has_error = True
            total += 1
            
        if has_error and i < 5:
            save_bad_case("词性", "".join(words), gold_tags, pred_tags)
            
    acc = correct / total if total > 0 else 0
    print(f"POS 准确率: {acc:.4f}")
    
    # 绘制混淆矩阵
    print("正在绘制 POS 混淆矩阵...")
    counter = collections.Counter(all_gold)
    top_labels = [k for k, v in counter.most_common(10)] # 改成10个防拥挤
    
    f_gold = []
    f_pred = []
    for g, p in zip(all_gold, all_pred):
        if g in top_labels and p in top_labels:
            f_gold.append(g)
            f_pred.append(p)
            
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        f_gold, f_pred, 
        normalize='true', 
        cmap='Blues', 
        values_format='.2f',
        ax=ax,
        xticks_rotation=45
    )
    plt.title(f'POS Confusion Matrix (Top {len(top_labels)} Tags)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/pos_confusion.png", dpi=300)
    print(f"图保存到了 pos_confusion.png")
    plt.close()

def evaluate_ner(nlp, data):
    print("正在评估命名实体识别...")
    
    stats = {
        'ALL': {'tp': 0, 'fp': 0, 'fn': 0},
        'nr': {'tp': 0, 'fp': 0, 'fn': 0},
        'ns': {'tp': 0, 'fp': 0, 'fn': 0},
        'nt': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    for words, gold_tags in data:
        doc = Doc(nlp.vocab, words=words)
        if 'ner' in nlp.pipe_names:
            doc = nlp.get_pipe('ner')(doc)
            
        gold_ents = set()
        for i, t in enumerate(gold_tags):
            if t in ['nr', 'ns', 'nt']:
                gold_ents.add((i, t))
                
        pred_ents = set()
        for ent in doc.ents:
            mapped = SPACY_TO_NER_MAP.get(ent.label_)
            if mapped:
                for i in range(ent.start, ent.end):
                    pred_ents.add((i, mapped))
                    
        for label in ['nr', 'ns', 'nt']:
            g = {x for x in gold_ents if x[1] == label}
            p = {x for x in pred_ents if x[1] == label}
            c = len(g & p)
            stats[label]['tp'] += c
            stats[label]['fp'] += len(p) - c
            stats[label]['fn'] += len(g) - c
            
            stats['ALL']['tp'] += c
            stats['ALL']['fp'] += len(p) - c
            stats['ALL']['fn'] += len(g) - c
            
    print("\n" + "="*45)
    print(f"{'Type':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 45)
    
    final_p, final_r, final_f1 = 0, 0, 0
    
    for k in ['ALL', 'nr', 'ns', 'nt']:
        s = stats[k]
        tp, fp, fn = s['tp'], s['fp'], s['fn']
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        if k == 'ALL':
            final_p, final_r, final_f1 = p, r, f1
            
        print(f"{k:<10} {p:<10.4f} {r:<10.4f} {f1:<10.4f}")
    print("="*45 + "\n")
    
    plot_metrics(['Precision', 'Recall', 'F1'], [final_p, final_r, final_f1], 'NER Metrics', 'ner.png')

def interactive_demo(nlp):
    print("\n" + "="*30)
    print("交互演示系统 (输入 'q' 退出)")
    print("="*30)
    while True:
        text = input("\n请输入文本: ")
        if text == 'q': break
        
        doc = nlp(text)
        print(f"分词: {[t.text for t in doc]}")
        print(f"词性: {[t.tag_ for t in doc]}")
        
        ents = [(e.text, e.label_) for e in doc.ents]
        print(f"实体: {ents}")

def main():
    nlp = spacy.load("zh_core_web_sm")
    evaluate_cws(nlp, load_cws_data(CWS_FILE))
    evaluate_pos(nlp, load_pos_data(POS_FILE))
    evaluate_ner(nlp, load_ner_data(NER_FILE))
        
    print(f"\n错误报告已保存至 {BAD_CASE_FILE}")
    a = input("\n是否进入演示模式? (y/n): ")
    if a == 'y':
        interactive_demo(nlp)

if __name__ == "__main__":
    main()
