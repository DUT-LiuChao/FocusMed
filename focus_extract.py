import json
from transformers import AutoTokenizer,AutoModelForCausalLM
import argparse
from tqdm import tqdm
import pandas as pd
import torch
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
def dataset_process(args):
    save_path = args.save_path
    datset_path = args.datset_path
    model_name_and_path = args.model_name_and_path
    tokenizer = AutoTokenizer.from_pretrained(model_name_and_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_and_path).to("cuda")

    type = args.type
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    top_p = 0.9
    temperature = 0.35
    repetition_penalty = 1.0
    max_new_tokens = 2048

    Result = []
    chq = None

    instcution = '''
            Consumer Health Question: {chq}.
            Find out the focus of the health questions asked by the above consumers, 
            and give the most important symptom entities and drug entities involved in the focus 
            (up to two for each entity), and try to keep the expression in the original text as concise as possible.

            Expected Output Format:
            Provide your response in dict format, following this example:
            {{
                "Focus": …,
                "Drug Entities": [entity1, entity2],
                "Sysptom entities": [entity1, entity2],
                "Explanation": Reasons for choosing these entities
            }}
            '''
    
    prompt = "Summarize the consumer health question into one question of 10 words or less.[CHQ]:{chq}\nThe focus of the question:{focus}"

    df = pd.read_excel(datset_path)
    Data = []

    for index,row in df.iterrows():
        if type == "train":
            chq = row['CHQ']
            summary = row['Summary']
            messages = [{"role": "user", "content": instcution.format(chq=chq)}]   
            text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device) 
            eos_token_id = tokenizer.eos_token_id
            for i in range(3):
                generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p, 
                temperature=temperature, 
                repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id,
                pad_token_id=eos_token_id
            )      
                generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                print(response)
                response = json.loads(response)

                Keyphrase = textrank_phrases(response['Focus'],top_n=2)
                nonepahrase = extract_noun_phrases_en(chq)
                
                Is_confidence = match_keyphrases(Keyphrase,nonepahrase)
                if Is_confidence:
                    break

            
            
            Data.append({
                'conversation':[{
                    'human':prompt.format(chq=chq,focus=response['Focus']),
                    'assistant':summary
                }]
            })

        with open(save_path,'w') as f:
            f.write(json.dumps(Data,ensure_ascii=False,indent=None))


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

def extract_noun_phrases_en(text: str):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # noun_chunks 会返回基于依存的名词短语跨度
    return [chunk.text for chunk in doc.noun_chunks]

def match_keyphrases(keyphrases, noun_phrases, threshold=0):
    vectorizer = TfidfVectorizer()
    results = []
    for kp in keyphrases:
        best_np, best_score = None, 0.0
        for np in noun_phrases:
            score = cosine_sim_score(kp, np, vectorizer)
            if score > best_score:
                best_score, best_np = score, np
        if best_score < threshold:
            False
    return True
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_and_path", type=str, default="...", help="model name and path")
    parser.add_argument("--save_path", type=str, default="...", help="save path")
    parser.add_argument("--datset_path", type=str, default="...", help="datset path")
    parser.add_argument("--type", type=str, default="train", help="[train, valid, test]")
    args = parser.parse_args()
    dataset_process(args)