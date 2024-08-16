import os
import json
import random
import requests
import re
from tqdm import tqdm
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry

def calculate_accuracy(data):
    small_questions = []
    middel_questions = []
    high_questions = []
    
    small_level1_questions = []
    small_level2_questions = []
    small_level3_questions = []

    middel_level1_questions = []
    middel_level2_questions = []
    middel_level3_questions = []

    high_level1_questions = []
    high_level2_questions = []
    high_level3_questions = []

    # 将数据按照题目类型分类
    for item in data:
        diff_type = item['knowledge_point'][0]['name']
        if diff_type == '小学数学':
            small_questions.append(item)
            if item['difficulty']['name'] in ['较易','易']:
                small_level1_questions.append(item)
            elif item['difficulty']['name']=='中档':
                small_level2_questions.append(item)
            elif item['difficulty']['name'] in ['较难','难']:
                small_level3_questions.append(item)
            else:
                print(item['difficulty']['name'])
        elif diff_type == '初中数学':
            middel_questions.append(item)
            if item['difficulty']['name']in ['较易','易']:
                middel_level1_questions.append(item)
            elif item['difficulty']['name']=='中档':
                middel_level2_questions.append(item)
            elif item['difficulty']['name']in ['较难','难']:
                middel_level3_questions.append(item)
            else:
                print(item['difficulty']['name'])
        elif diff_type == '高中数学':
            high_questions.append(item)
            if item['difficulty']['name']in ['较易','易']:
                high_level1_questions.append(item)
            elif item['difficulty']['name']=='中档':
                high_level2_questions.append(item)
            elif item['difficulty']['name']in ['较难','难']:
                high_level3_questions.append(item)
            else:
                print(item['difficulty']['name'])
    #print(len(small_questions),len(middel_questions),len(high_questions))  ##
    #print("小学总计 较易,中档，较难：",len(small_level1_questions)+len(small_level2_questions)+len(small_level3_questions),len(small_level1_questions),len(small_level2_questions),len(small_level3_questions))
    #print("初中总计 较易,中档，较难：",len(middel_level1_questions)+len(middel_level2_questions)+len(middel_level3_questions),len(middel_level1_questions),len(middel_level2_questions),len(middel_level3_questions))
    #print("高中总计 较易,中档，较难：",len(high_level1_questions)+len(high_level2_questions)+len(high_level3_questions),len(high_level1_questions),len(high_level2_questions),len(high_level3_questions))
    # 计算各类题目的准确率
    def calculate_accuracy_for_type(questions):
        total_count = len(questions)
        total_accuracy=0
        for item in questions:
            correct_count = 0
            for i in range(len(item['question_class'])):
                if item['question_class'][i]['result']==True:
                    correct_count += 1
            accuracy=(correct_count / len(item['question_class'])) *100
            total_accuracy+=accuracy          
        average_accuracy = total_accuracy / total_count
        return average_accuracy

    selection_accuracy = calculate_accuracy_for_type(small_questions)
    solution_accuracy = calculate_accuracy_for_type(middel_questions)
    proof_accuracy = calculate_accuracy_for_type(high_questions)

    small1 = calculate_accuracy_for_type(small_level1_questions)
    small2 = calculate_accuracy_for_type(small_level2_questions)
    small3 = calculate_accuracy_for_type(small_level3_questions)

    middel1 = calculate_accuracy_for_type(middel_level1_questions)
    middel2 = calculate_accuracy_for_type(middel_level2_questions)
    middel3 = calculate_accuracy_for_type(middel_level3_questions)

    high1 = calculate_accuracy_for_type(high_level1_questions)
    high2 = calculate_accuracy_for_type(high_level2_questions)
    high3 = calculate_accuracy_for_type(high_level3_questions)
    
    print('小学数学准确率:',selection_accuracy, '初中数学准确率:',solution_accuracy,'高中数学准确率:',proof_accuracy,"\n")
    print('小学容易准确率:',small1 , '小学中档准确率:',small2 ,'小学难准确率:',small3)
    print('初中容易准确率:',middel1 , '初中中档准确率:', middel2 ,'初中难准确率:', middel3)
    print('高中容易准确率:',high1 , '高中中档准确率:',high2 ,'高中难准确率:', high3)



data=[]
with open("parallel_gpt4_ans_with_eval_result.jsonl", 'r') as f:
       for line in f:
            # 跳过空行或只包含空白字符的行
            if not line.strip():
                continue
            one = json.loads(line)
            data.append(one)
if __name__ == '__main__':
    calculate_accuracy(data)