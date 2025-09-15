# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import json
from tqdm import tqdm
from loguru import logger
import pandas as pd
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import argparse
client = OpenAI(api_key="...", base_url="https://api.deepseek.com")

'''
    输入数据格式：{
                'chq": "Consumer Health Question",
                'summary': "Summary of model output"
                    }



'''
# 计算两个短语之间的余弦相似度
def cosine_sim_score(phrase1, phrase2, vectorizer):
    tfidf_matrix = vectorizer.fit_transform([phrase1, phrase2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


# 生成短语候选并计算TextRank得分
def textrank_phrases(text, top_n=5):

    # 加载spaCy的英文模型
    nlp = spacy.load('en_core_web_sm')

    # 使用spaCy进行文本处理
    doc = nlp(text)
    
    # 提取名词短语，作为候选短语
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    print(noun_phrases)
    # 过滤停用词
    noun_phrases = [phrase for phrase in noun_phrases if phrase.lower() not in nlp.Defaults.stop_words]

    # TF-IDF向量化
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # 计算短语之间的相似度矩阵
    similarity_matrix = np.zeros((len(noun_phrases), len(noun_phrases)))
    for i in range(len(noun_phrases)):
        for j in range(len(noun_phrases)):
            if i != j:
                similarity_matrix[i][j] = cosine_sim_score(noun_phrases[i], noun_phrases[j], vectorizer)
    
    # PageRank计算短语得分
    scores = np.ones(len(noun_phrases))  # 初始化得分
    d = 0.85  # 阻尼因子
    max_iter = 100  # 最大迭代次数
    tol = 1e-6  # 收敛阈值
    
    for _ in range(max_iter):
        prev_scores = scores.copy()
        for i in range(len(noun_phrases)):
            scores[i] = (1 - d) + d * sum(similarity_matrix[j][i] * prev_scores[j] for j in range(len(noun_phrases)))
        
        if np.linalg.norm(scores - prev_scores, ord=1) < tol:
            break
    
    # 获取得分最高的短语作为关键词
    phrase_scores = {noun_phrases[i]: scores[i] for i in range(len(noun_phrases))}
    top_phrases = [phrase for phrase, score in Counter(phrase_scores).most_common(top_n)]
    
    return top_phrases

def get_atomic_facts(text):
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a linguistics expert, please complete the following questions according to my requirements"},
        {"role": "user", "content": f"Text:{text}.Please break down the above text into atomic facts (the most basic facts that cannot be decomposed any further, which represent the smallest information unit in the data). The generated format is as follows: Atomic facts: [atomic fact 1, atomic fact 2, atomic fact 3, ...]"},
    ],
    stream=False,
)
    return response.choices[0].message.content
    
def Is_imply(source,atomic_facts):
    response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a linguistics expert, please complete the following questions according to my requirements"},
        {"role": "user", "content": f"""Source:{source}\nAtomic_facts:{atomic_facts}\nPlease determine which of these atomic facts are contained in the original text and provide the number of atomic facts contained.Finally, provide the number of atoms included and not included, as well as the total number of atomic facts.Please output strictly in the following format:[The number of contained, The number of not contained, The total number of atomic facts](Only provide answers according to my requirements, do not give any other content)"""
            },
    ],
    stream=False)


def calculaet_score(data):
    chq = data['chq']
    summary = data['summary']
    atomic_facts_from_chq = get_atomic_facts(chq)
    atomic_facts_from_summary = get_atomic_facts(summary)
    
    result_summary = Is_imply(source=chq,atomic_facts=atomic_facts_from_summary)
    assert result_summary[2] == result_summary[0]+result_summary[1]
    Fuithness_score = result_summary[0]/(result_summary[0]+result_summary[1])

    result_chq = Is_imply(source=summary,atomic_facts=atomic_facts_from_chq)
    assert result_chq[2] == result_chq[0]+result_chq[1]
    Consistency_score = result_chq[0]/(result_chq[0]+result_chq[1])

    keyphrase = textrank_phrases(summary,top_n=3)
    key_len = 0
    for key in keyphrase:
        key_len += len(key)
    concision_score = key_len/len(summary)

    return Fuithness_score,Consistency_score,concision_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="...", help="data path")
    args = parser.parse_args()
    data_path = args.data_path
    with open(data_path,'r') as f:
        data = json.load(f)
    for item in data:
        Fuithness_score,Consistency_score,concision_score = calculaet_score(item)
        item['Fuithness_score'] = Fuithness_score
        item['Consistency_score'] = Consistency_score
        item['concision_score'] = concision_score

    with open(data_path,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
