import os
import json
import random
import requests
import re
from tqdm import tqdm
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
os.environ["OPENAI_API_KEY"] = "sk-xxx"

# openai设置
class openai_config:
    openai_api = os.environ.get('OPENAI_API_ADDR', 'http://xx.xxx.xxx.xxx')
    openai_key = os.environ.get('OPENAI_API_KEY')

# 重试策略
retry_strategy = Retry(
    total=5,  # 最大重试次数（包括首次请求）
    backoff_factor=1,  # 重试之间的等待时间因子
    status_forcelist=[404, 429, 500, 502, 503, 504],  # 需要重试的状态码列表
    allowed_methods=["POST"]  # 只对POST请求进行重试
)

adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=50)
# 创建会话并添加重试逻辑
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

#gpt-4-1106-preview high  gpt-3.5-turbo
def openai_chat(messages, model='gpt-4-0125-preview', temperature=1.0, finish_try=3):
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    while True:

        headers = {'Content-Type': 'application/json'}

        if openai_config.openai_api.startswith('http://xx.xxx.xxx.xxx') and not openai_config.openai_key:
            raise KeyError('Please set openai_key from http://xx.xxx.xxx.xxx')

        if openai_config.openai_key:
            if isinstance(openai_config.openai_key, str):
                api_key = openai_config.openai_key
            else:
                api_key = random.choice(openai_config.openai_key)
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'

        try:
            response = session.post(
                openai_config.openai_api + '/v1/chat/completions',
                headers = headers,
                data = json.dumps(dict(
                    model = model,
                    messages=messages,
                    temperature=temperature
                )))
            assert response.status_code == 200
            response = json.loads(response.text)
            finish_reason = response['choices'][0]['finish_reason']
            response = response['choices'][0]['message']['content']
            finish_try -= 1
            if finish_reason != 'stop' and finish_try:
                raise
            return response, finish_reason
        except Exception as e:
            # print(e, flush=True)
            try:
                error_code = json.loads(e.http_body)['error']['code']
                if error_code in ('billing_not_active', 'context_length_exceeded'):
                    return '', error_code
            except:
                pass
            continue
def parse(info_dict, response):
    if response is not None:
        #info_dict['reason'] = response
        scores = re.findall(r'\[([0-5])\]',response)
        if len(scores) > 0:
            try:
                return int(scores[-1])
            except Exception:
                return None
    return None
GET_PROMPT="""\
你需要抽取出学生每道小题的答案的表达式
学生回答：{response}
你需要输出的内容：
学生答案: {{抽取的学生的答案结果:(1){{学生答案}}(2){{学生答案}}(3){{学生答案}}(4)....}}
"""

input_file= 'math_question_solution_ans.json'
input_file_nomalize='math_with_class.jsonl'
ans_file='parallel_gpt4_ans.json'
out_file='parallel_gpt4_ans_with_eval_result.jsonl'

model_ans_list=json.load(open(ans_file,'r',encoding='utf-8'))
id2model_ans={}
for ans in model_ans_list:
    for k,v in ans.items():
        id2model_ans[k]=v

id2data_with_class = {}
with open(input_file_nomalize, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = json.loads(line)
        id2data_with_class[data['id']] = data


PROMPT_JUDGE="""\

任务描述：评估给定数学问题的学生答案是否正确。

输入：
1. 题目描述：[题目的详细描述，包括必要的数学公式和条件。]{question},
2. 参考答案：[正确答案的详细说明，包括计算过程和结果。]{answer},
3. 学生答案：[学生提供的答案，包括计算过程和结果。]{response},

要求：
- 仔细比较学生答案和参考答案。
- 分析学生答案的正确性，包括计算过程和最终结果。
- 如果学生答案错误，指出错误之处并简要解释错误的原因。
- 提供一个简洁的评估结论，明确学生答案是否正确。

示例：
题目描述：计算三角形的面积，已知底为 6 cm，高为 3 cm。
参考答案：(1)面积 = 0.5 * 底 * 高 = 0.5 * 6 cm * 3 cm = 9 cm²。
学生答案：(1)面积 = 6 cm * 3 cm = 18 cm²。


评估：
(1)False,解释如下：
- 学生的计算过程忽略了面积计算公式中的1/2系数。
- 结果错误，正确的计算应该是 9 cm²，而不是 18 cm²。
- 结论：学生答案错误。

请根据上述任务描述和要求，把参考答案和学生答案按顺序依次对比，谨慎思考是否一致，2、如果学生作答正确，输出True；否则输出False：并给出评估结论。
你需要输出的内容：
只需要输出每题的True 或False，示例：判断结果：(1)True,(2)False,(3)True
解释如下：(1)... （2）... （3）...
"""

#题目为:{question}
#判断结果：(1){{True｜False}},(2){{True｜False}},(3){{True｜False}} (4)....


def calculate_accuracy(results):
    # 使用正则表达式匹配出所有的判断结果
    #results = re.findall(r'\((.*?)\)(True|False)', results)
    pattern = r'\b(True|False)\b'
    results = re.findall(pattern, results)

    # 统计True的个数
    true_count = sum([1 for result in results if result == 'True'])
    total = len(results)
    
    print("resul num", results,total)
    
    if(total==0):
        #print("result??",results)
        total=1

    accuracy = true_count / total * 100
    return accuracy


def judge_jsonl(data):
        if (data['answer']==None):
            print("none")
            standard_ans=data['solution']
        else:
            standard_ans=data['answer']+" "+data['solution']
        conversations = data.get("conversations", [])
        for conversation in conversations:
            if conversation.get("from") == "human":
                question= conversation.get("value")
        model_ans = data['model_output']
        get_prompt = GET_PROMPT.format(response=model_ans)
        model_ans=openai_chat(get_prompt)
        full_prompt = PROMPT_JUDGE.format(question=question, answer=standard_ans, response=model_ans[0])
        judge_response=openai_chat(full_prompt)
        return  judge_response[0],id


jsonl_data=[]
with open("parallel_gpt4_ans_with_eval_result.jsonl", 'r') as f:
       for line in f:
            # 跳过空行或只包含空白字符的行
            if not line.strip():
                continue
            one = json.loads(line)
            jsonl_data.append(one)


def parallel_judge():
    total_accuracy=0
    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        process_futures = []
        for i, _ in enumerate(jsonl_data):  #json
            process_futures.append(executor.submit(judge_jsonl, jsonl_data[i]))       
        for future in futures.as_completed(process_futures):
            judge_response,id = future.result()
            accuracy = calculate_accuracy(judge_response)
            total_accuracy+=accuracy
            print("准确率：",  accuracy, "%")
    average_accuracy = total_accuracy / len(data)
    print("平均准确率：", average_accuracy, "%")
        
if __name__ == '__main__':
    parallel_judge()
