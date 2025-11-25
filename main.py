import spacy
from spacy.tokens import Doc
import matplotlib.pyplot as plt
import os

# 解决绘图报错
plt.switch_backend('Agg')

CWS_FILE = "./data/中文分词.txt"
POS_FILE = "./data/词性标注.tsv"
NER_FILE = "./data/命名实体识别.tsv"
IMG_DIR = "images"

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

def plot_metrics(p, r, f1, title, filename):
    plt.figure(figsize=(8, 6))
    plt.bar(['Precision', 'Recall', 'F1'], [p, r, f1])
    plt.title(title)
    plt.ylim(0, 1.0)
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
    tp = 0
    fp = 0
    fn = 0
    
    for raw_text, gold_words in data:
        doc = nlp(raw_text)
        pred_words = [t.text for t in doc]
        
        gold_inv = get_intervals(gold_words)
        pred_inv = get_intervals(pred_words)
        
        correct = len(gold_inv & pred_inv)
        tp += correct
        fp += len(pred_inv) - correct
        fn += len(gold_inv) - correct
        
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"分词结果: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    plot_metrics(p, r, f1, 'CWS Metrics', 'cws.png')

def evaluate_pos(nlp, data):
    print("正在评估词性标注...")
    correct = 0
    total = 0
    
    for words, gold_tags in data:
        doc = Doc(nlp.vocab, words=words)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
            
        for i, token in enumerate(doc):
            if i >= len(gold_tags): break
            gold = gold_tags[i]
            gold = PKU_TO_CTB_MAP.get(gold, gold)
            if token.tag_ == gold:
                correct += 1
            total += 1
            
    acc = correct / total if total > 0 else 0
    print(f"POS 准确率: {acc:.4f}")

def evaluate_ner(nlp, data):
    print("正在评估命名实体识别...")
    tp = 0
    fp = 0
    fn = 0
    
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
                    
        correct = len(gold_ents & pred_ents)
        tp += correct
        fp += len(pred_ents) - correct
        fn += len(gold_ents) - correct
        
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    print(f"NER结果: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
    plot_metrics(p, r, f1, 'NER Metrics', 'ner.png')

def main():
    try:
        nlp = spacy.load("zh_core_web_sm")
    except:
        print("没找到模型，请安装 zh_core_web_sm")
        return

    if os.path.exists(CWS_FILE):
        evaluate_cws(nlp, load_cws_data(CWS_FILE))
        
    if os.path.exists(POS_FILE):
        evaluate_pos(nlp, load_pos_data(POS_FILE))
        
    if os.path.exists(NER_FILE):
        evaluate_ner(nlp, load_ner_data(NER_FILE))

if __name__ == "__main__":
    main()
