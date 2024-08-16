import os
import json
import random
import requests
import re
from tqdm import tqdm
from concurrent import futures
import numpy as np
from requests.adapters import HTTPAdapter, Retry


def calculate_accuracy(data,cla):
    a_questions = []
    b_questions = []
    c_questions = []
    d_questions = []
    e_questions = []
    # 将数据按照题目类型分类
    for item in data:
        k=cla[0][(item['index'])]['class']
        #print(k)
        if k == '代数':
            a_questions.append(item)
        elif k == '几何':
            b_questions.append(item) 
        elif k == '方程与不等式':
            c_questions.append(item)
        elif k == '函数':
            d_questions.append(item)
        elif k == '概率统计':
            e_questions.append(item)
    #print(len(a_questions),len(b_questions),len(c_questions),len(d_questions),len(e_questions)) 
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

    a_accuracy = calculate_accuracy_for_type(a_questions)
    b_accuracy = calculate_accuracy_for_type(b_questions)
    c_accuracy = calculate_accuracy_for_type(c_questions)
    d_accuracy = calculate_accuracy_for_type(d_questions)
    e_accuracy = calculate_accuracy_for_type(e_questions)
    
    print('代数题准确率:',a_accuracy, '几何题准确率:',b_accuracy,'方程与不等式题准确率:',c_accuracy )
    print('函数题准确率:',d_accuracy, '概率统计题准确率:',e_accuracy)

data=[]
classdata=[]

with open("parallel_glm4v_2_ans1369_with_eval_result.jsonl", 'r') as f:
       for line in f:
            # 跳过空行或只包含空白字符的行
            if not line.strip():
                continue
            one = json.loads(line)
            data.append(one)
with open("question_knowledge.json", 'r') as f:
            one = json.load(f)
            classdata.append(one)

if __name__ == '__main__':
    calculate_accuracy(data,classdata)