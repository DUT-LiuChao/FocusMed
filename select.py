import json
import argparse
def Selection(path1,path2,path3,path4):
    with open(path1,'r',encoding='utf-8') as f:
        data1 = json.load(f)
    with open(path2,'r',encoding='utf-8') as f:
        data2 = json.load(f)
    with open(path3,'r',encoding='utf-8') as f:
        data3 = json.load(f)
    with open(path4,'r',encoding='utf-8') as f:
        data4 = json.load(f)

    alpha = 0.3 
    beta = 0.3
    gamma = 0.4
    Data = []
    for i in range(len(data1)):
        score_1 = alpha * data1[i]['Fuithness_score'] + beta * data1[i]['Consistency_score'] + gamma * data1[i]['concision_score']
        score_2 = alpha * data2[i]['Fuithness_score'] + beta * data2[i]['Consistency_score'] + gamma * data2[i]['concision_score']
        score_3 = alpha * data3[i]['Fuithness_score'] + beta * data3[i]['Consistency_score'] + gamma * data3[i]['concision_score']
        score_4 = alpha * data4[i]['Fuithness_score'] + beta * data4[i]['Consistency_score'] + gamma * data4[i]['concision_score']

        if max(score_1,score_2,score_3,score_4) == score_1:
            Data.append(data1[i])
        elif max(score_1,score_2,score_3,score_4) == score_2:
            Data.append(data2[i])
        elif max(score_1,score_2,score_3,score_4) == score_3:
            Data.append(data3[i])
        elif max(score_1,score_2,score_3,score_4) == score_4:
            Data.append(data4[i])

    return Data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path1", type=str, default="...", help="data path")
    parser.add_argument("--path2", type=str, default="...", help="data path")
    parser.add_argument("--path3", type=str, default="...", help="data path")
    parser.add_argument("--path4", type=str, default="...", help="data path")
    args = parser.parse_args()
    path1 = args.path1
    path2 = args.path2
    path3 = args.path3
    path4 = args.path4
    Data = Selection(path1,path2,path3,path4)
    with open("FocusMed.json","w") as f:
        json.dump(Data,f,ensure_ascii=False,indent=2)