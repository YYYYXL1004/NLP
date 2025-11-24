import spacy
from spacy.tokens import Doc
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import collections

# 设置 matplotlib 后端，避免无显示环境报错
plt.switch_backend('Agg')

# ==========================================
# 配置区域
# ==========================================
DATA_DIR = "./data"
CWS_FILE = os.path.join(DATA_DIR, "中文分词.txt")
POS_FILE = os.path.join(DATA_DIR, "词性标注.tsv")
NER_FILE = os.path.join(DATA_DIR, "命名实体识别.tsv")
MODEL_NAME = "zh_core_web_sm"
REPORT_FILE = "evaluation_report.txt"
IMG_DIR = "images"

# 创建图片目录
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

# 映射表: 将数据集的标签 (PKU风格) 映射到 spaCy/CTB 标签 (OntoNotes风格)
PKU_TO_CTB_MAP = {
    'n': 'NN', 'nr': 'NR', 'ns': 'NR', 'nt': 'NR', 'nz': 'NR',
    'v': 'VV', 'vn': 'VV', 
    'a': 'VA', 
    'd': 'AD',
    'p': 'P',
    'c': 'CC', 
    'u': 'DEG', 
    'm': 'CD',
    'q': 'M',
    'r': 'PN',
    't': 'NT',
    'f': 'LC', 
    'w': 'PU',
    # 注意：这只是基础映射，spaCy 实际上有很多细分标签，如 DEC, DEG, AS, SP 等
    # 如果想提高准确率，需要更细致的分析
}

# NER 映射表: SpaCy (OntoNotes) -> 数据集标签
SPACY_TO_NER_MAP = {
    'PERSON': 'nr',
    'ORG': 'nt',
    'GPE': 'ns',
    'LOC': 'ns'
}

def load_cws_data(file_path):
    """加载分词数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            words = line.split()
            raw_text = "".join(words)
            data.append((raw_text, words))
    return data

def load_pos_data(file_path):
    """加载词性标注数据"""
    sentences = []
    current_words = []
    current_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append((current_words, current_tags))
                    current_words = []
                    current_tags = []
                continue
            
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            word, tag = parts
            current_words.append(word)
            current_tags.append(tag)
            
    if current_words:
        sentences.append((current_words, current_tags))
    return sentences

load_ner_data = load_pos_data

def log_to_file(message):
    """记录日志到文件"""
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        f.write(message + "\n")
    print(message)

def plot_metrics(precision, recall, f1, title, filename):
    """通用指标绘制函数"""
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.1)
    plt.title(title)
    plt.ylabel('Score')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
                 
    plt.savefig(os.path.join(IMG_DIR, filename))
    plt.close()
    log_to_file(f"[图片生成] {title}已保存至 {IMG_DIR}/{filename}")

def evaluate_cws(nlp, data):
    """评估分词效果 (P, R, F1)"""
    log_to_file(f"\n{'='*20} 分词任务评估 {'='*20}")
    log_to_file(f"总样本数: {len(data)}")
    
    tp = 0
    fp = 0
    fn = 0
    
    error_cases = [] 

    for raw_text, gold_words in data:
        doc = nlp(raw_text)
        pred_words = [token.text for token in doc]
        
        gold_intervals = get_intervals(gold_words)
        pred_intervals = get_intervals(pred_words)
        
        correct = len(gold_intervals & pred_intervals)
        tp += correct
        fp += len(pred_intervals) - correct
        fn += len(gold_intervals) - correct
        
        if len(gold_intervals) > 0:
            p = correct / len(pred_intervals) if pred_intervals else 0
            r = correct / len(gold_intervals)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 < 0.6: # 稍微放宽条件，记录更多 bad case
                error_cases.append({
                    "text": raw_text,
                    "gold": gold_words,
                    "pred": pred_words,
                    "f1": f1
                })

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    log_to_file(f"[CWS 结果] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    
    # 绘图
    plot_metrics(precision, recall, f1_score, 'Chinese Word Segmentation Performance', 'cws_metrics.png')
    
    return error_cases

def get_intervals(words):
    intervals = set()
    idx = 0
    for word in words:
        intervals.add((idx, idx + len(word)))
        idx += len(word)
    return intervals

def plot_confusion_matrix_custom(y_true, y_pred, top_n=15):
    """绘制混淆矩阵（仅展示最常出现的 top_n 个标签）"""
    # 统计最常见的标签
    counter = collections.Counter(y_true)
    top_tags = [tag for tag, count in counter.most_common(top_n)]
    
    # 过滤数据，只保留 top_tags 涉及的数据（或者全部计算但只画这部分）
    # 为了简单，我们计算完整的 CM，然后只切片 top_tags
    labels = sorted(list(set(y_true + y_pred)))
    # cm = confusion_matrix(y_true, y_pred, labels=labels) # 不需要全量计算，下面用 mask 更好
    
    # 策略：筛选出 y_true 属于 top_tags 的样本
    filtered_true = []
    filtered_pred = []
    for t, p in zip(y_true, y_pred):
        if t in top_tags and p in top_tags:
            filtered_true.append(t)
            filtered_pred.append(p)
            
    if not filtered_true:
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay.from_predictions(
        filtered_true, 
        filtered_pred, 
        cmap='Blues', 
        normalize='true',
        values_format='.2f',
        ax=ax
    )
    plt.title(f'POS Confusion Matrix (Top {top_n} Tags, Normalized)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'pos_confusion_matrix.png'))
    plt.close()
    log_to_file(f"[图片生成] 词性标注混淆矩阵已保存至 {IMG_DIR}/pos_confusion_matrix.png")

def evaluate_pos(nlp, data):
    """评估词性标注效果 (Accuracy)"""
    log_to_file(f"\n{'='*20} 词性标注评估 {'='*20}")
    log_to_file(f"总句子数: {len(data)}")
    
    correct_tags = 0
    total_tags = 0
    
    y_true_all = []
    y_pred_all = []
    
    for words, gold_tags in data:
        doc = Doc(nlp.vocab, words=words)
        # spaCy v3 fix: 手动运行 pipeline 组件
        for name, proc in nlp.pipeline:
            doc = proc(doc)
        
        pred_tags = [token.tag_ for token in doc]
        
        for gold, pred in zip(gold_tags, pred_tags):
            mapped_gold = PKU_TO_CTB_MAP.get(gold, gold) 
            
            if mapped_gold == pred:
                correct_tags += 1
            
            y_true_all.append(mapped_gold)
            y_pred_all.append(pred)
            total_tags += 1
            
    accuracy = correct_tags / total_tags if total_tags > 0 else 0
    log_to_file(f"[POS 结果] Accuracy: {accuracy:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix_custom(y_true_all, y_pred_all, top_n=15)

def evaluate_ner(nlp, data):
    """评估命名实体识别效果 (P, R, F1)"""
    log_to_file(f"\n{'='*20} 命名实体识别评估 {'='*20}")
    log_to_file(f"总句子数: {len(data)}")
    
    tp = 0
    fp = 0
    fn = 0
    
    # 这里的 "interval" 定义为 (token_index, label)
    # 这种粒度介于 Token-level 和 Entity-level 之间
    # 考虑到数据集分词粒度较粗，单个词往往就是一个实体，这种对齐是合理的。
    
    for words, gold_tags in data:
        doc = Doc(nlp.vocab, words=words)
        # 运行 NER
        if 'ner' in nlp.pipe_names:
            doc = nlp.get_pipe('ner')(doc)
        
        # 构建 Gold 集合: {(index, label)}
        gold_entities = set()
        for i, tag in enumerate(gold_tags):
            if tag in ['nr', 'ns', 'nt']:
                gold_entities.add((i, tag))
        
        # 构建 Pred 集合
        pred_entities = set()
        for ent in doc.ents:
            mapped_label = SPACY_TO_NER_MAP.get(ent.label_)
            if mapped_label:
                # 将 span 映射回 token index
                for i in range(ent.start, ent.end):
                    pred_entities.add((i, mapped_label))
        
        # 计算指标
        correct = len(gold_entities & pred_entities)
        tp += correct
        fp += len(pred_entities) - correct
        fn += len(gold_entities) - correct
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    log_to_file(f"[NER 结果] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    
    # 绘图
    plot_metrics(precision, recall, f1_score, 'NER Performance', 'ner_metrics.png')

def main():
    # 清空旧日志
    if os.path.exists(REPORT_FILE):
        os.remove(REPORT_FILE)
        
    try:
        nlp = spacy.load(MODEL_NAME)
    except OSError:
        print(f"错误: 未找到模型 '{MODEL_NAME}'。")
        print(f"请运行: python -m spacy download {MODEL_NAME}")
        return

    # 1. 中文分词
    if os.path.exists(CWS_FILE):
        cws_data = load_cws_data(CWS_FILE)
        errors = evaluate_cws(nlp, cws_data)
        
        log_to_file(f"\n[扩展分析] 记录分词 F1 < 0.6 的样本 (Top 5):")
        for i, err in enumerate(errors[:5]):
            log_to_file(f"样本 {i+1}: {err['text']}")
            log_to_file(f"  Gold: {err['gold']}")
            log_to_file(f"  Pred: {err['pred']}")
            log_to_file(f"  F1:   {err['f1']:.4f}")
    else:
        log_to_file(f"未找到分词数据: {CWS_FILE}")

    # 2. 词性标注
    if os.path.exists(POS_FILE):
        pos_data = load_pos_data(POS_FILE)
        evaluate_pos(nlp, pos_data)
    else:
        log_to_file(f"未找到词性标注数据: {POS_FILE}")
        
    # 3. 命名实体识别
    if os.path.exists(NER_FILE):
        ner_data = load_ner_data(NER_FILE)
        evaluate_ner(nlp, ner_data)
    else:
        log_to_file(f"未找到命名实体识别数据: {NER_FILE}")
        
    print(f"\n所有结果已保存至 {REPORT_FILE}")
    print(f"图片已保存至 {IMG_DIR}/ 目录")

if __name__ == "__main__":
    main()
