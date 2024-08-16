import os
import json
import random
import requests
import re
from tqdm import tqdm
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
os.environ["OPENAI_API_KEY"] = "sk-xxxx"

def calculate_accuracy(data):
    selection_questions = []
    solution_questions = []
    proof_questions = []
    # 将数据按照题目类型分类
    for item in data:
        if 'question_type' not in item['question_class'][0]:
            print("no class",item['id'])
            continue
        else:
            question_type = item['question_class'][0]['question_type']
            if question_type == '选择题':
                selection_questions.append(item)
            elif question_type == '解答题':
                solution_questions.append(item)
            elif question_type == '证明题':
                proof_questions.append(item)
    print("选择题数量:",len(selection_questions),"解答题数量:",len(solution_questions),"证明题数量:",len(proof_questions))  ##
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

    selection_accuracy = calculate_accuracy_for_type(selection_questions)
    solution_accuracy = calculate_accuracy_for_type(solution_questions)
    proof_accuracy = calculate_accuracy_for_type(proof_questions)
    
    print('选择题准确率:',selection_accuracy,"%",'解答题准确率:',solution_accuracy,"%",'证明题准确率:',proof_accuracy,"%",)


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