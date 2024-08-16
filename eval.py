import os
import json
import random
import requests
import re
from tqdm import tqdm
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
from jinja2 import Template
from ratelimit import limits, sleep_and_retry

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
def openai_chat(messages, model='gpt-4-1106-preview', temperature=1.0, finish_try=3):
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

GET_PROMPT="""\
你需要抽取出{q_len}小题的最终答案的表达式
回答：{response}
你需要输出的内容为{q_len}条答案：
{{(1){{答案}}(2){{答案}}....}}
"""

PROMPT_JUDGE_OLD="""\
参考答案为：{answer}，
学生答案：{response}
把参考答案和学生答案按顺序依次对比，谨慎思考是否一致，2、如果学生作答正确，输出True；否则输出False：\\
只需要输出每题的True 或False，示例：判断结果：(1)True,(2)False,(3)True
"""

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
# 根据GPT-4的答案和正确答案进行评分


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
        total=1

    accuracy = true_count / total * 100
    return accuracy

TIME = 10
@sleep_and_retry
@limits(calls=5, period=TIME) # 
def call_api(prompts):
    response,_ = openai_chat(prompts)
    return response,_

def judge(q_len,key, data,model_ans):

    questions = data["question_class"]
    ocr=data["conversations"][0]["value"]
    
    #print("stand ans",model_ans)
    prompts="""
数学问题有{{q_len}}条参考答案和对应的解题过程如下：
{% for question in questions %}
 类型：{{ question.question_type }}
{% endfor %}
{% for question in questions %}
参考答案：{{ question.answer }}}
{% endfor %}
{% for question in questions %}
解题过程：{{ question.solution }}
{% endfor %}
学生的解题过程和最终答案如下。```\n解题过程与最终答案\n```\n{{ model_ans }}\n```
首先请直接按参考答案和学生答案的顺序进行{{q_len}}次对比，判断是否一致，
如果学生的答案正确，则无需对比解题过程。
如果学生的答案错误，则需要对比参考解题过程和学生解题过程，找出差异并解释原因。
如果是证明题，则需要判断每一步推理步骤的正确或错误，及其原因。
并输出以下格式的JSON结果 ：
[
    {
        "reason": "学生答案正确或者错误的原因是",
        "result": true or false
    },
    ...
]
输出示例：
[
    {
        "reason": "学生答案的90m与参考答案的220m不一致",
        "result": true 
    },
    ...
]
请确保输出的JSON字符串只包含{{ q_len }}条"reason"和"result",不要包含任何其他内容。
"""
    prompts = Template(prompts)
    prompts = prompts.render(q_len=q_len,ocr=ocr, questions=questions, model_ans=model_ans)
    res,_ = call_api(prompts) 
    return  res, key,data, model_ans

input_file= 'math_question_solution_ans.json'
input_file_nomalize='math_with_class.jsonl'

ans_file='parallel_glm4v_ans.jsonl'
out_file='parallel_glm4v_ans_with_eval_result.jsonl' 

id2model_ans={}
with open(ans_file,'r',encoding='utf-8') as f:
    for line in f:
        model_ans_list=json.loads(line)
        for k,v in model_ans_list.items():
            id2model_ans[k]=v

id2data_with_class = {}
with open(input_file_nomalize, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = json.loads(line)
        id2data_with_class[data['id']] = data


def parallel_call_cache(dataset,input_file, out_file, max_workers=1):
    '''多线程，实时缓存结果'''
    cnt = 0
    pbar = tqdm(total=len(dataset))
    id_set = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data = json.loads(line)
                id_set.add(data['id'])
        
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:    
            process_futures = []
            for i, _ in enumerate(dataset):
                key=dataset[i].keys()
                key=(list(key)[0])
                id = dataset[i][key]["image_id"]#dataset[i]["id"]
                if id in id_set:
                    continue
                model_ans = id2model_ans[key]["ans"] #id2model_ans[str(i+1)]["ans"]
                question_data = id2data_with_class[id]
                q_len=len(question_data["question_class"])
                get_prompt = GET_PROMPT.format(q_len=q_len,response= model_ans,)
                model_ans= call_api(get_prompt)
                model_ans=model_ans[0]
                process_futures.append(executor.submit(judge,q_len,key,question_data, model_ans))
                cnt += 1
            pbar = tqdm(total=cnt)
            for future in futures.as_completed(process_futures):
                res, k, data, model_ans = future.result()
                try:
                    data["index"]=k
                    data["model_output"] = model_ans
                    if res is not None:                  
                        # 后处理res
                        json_pattern = r'\[.*\]'
                        match = re.search(json_pattern, res, re.DOTALL)
                        res = match.group(0)

                        try:
                            res_data = json.loads(res)
                        except:
                            # 将单独的转义字符\替换为\\
                            res = re.sub(r'(?<!\\)\\(?!\\|")', r'\\\\', res)
                            res_data = json.loads(res)
                        if(len(data["question_class"]) != len(res_data)):
                            #print("id ",k)
                            print(data["question_class"],model_ans,res_data)
                            print(len(data["question_class"]),len(res_data))
                        assert(len(data["question_class"]) == len(res_data))

                        for i in range(len(data["question_class"])):
                            data["question_class"][i]["result"] = res_data[i]["result"]
                            data["question_class"][i]["reason"] = res_data[i]["reason"]

                        data_json_str = json.dumps(data, ensure_ascii=False)
                        with open(out_file, "a", encoding="utf-8") as f:
                            f.write(data_json_str + "\n")
                        #print(k,"OK ")
                except Exception as e:
                    print("pass",k)
                    print(repr(e),str(e))
                    print(data["question_class"],model_ans,res_data,len(data["question_class"]),len(res_data))
                    with open("pass_gpt4_id.json", "a") as file:
                            json.dump(k, file)
                            file.write(' '+"\n")
                    continue
                
                pbar.update(1)
    pbar.close()

def get_parallel_answer():
    with open(input_file, 'r') as f:
        data = json.load(f)
    parallel_call_cache(data,input_file,out_file,5)
        
if __name__ == '__main__':
    get_parallel_answer()
