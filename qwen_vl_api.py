import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from http import HTTPStatus
import dashscope
import requests
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
import time
from ratelimiter import RateLimiter
from ratelimit import limits, sleep_and_retry

import json
import os
from PIL import Image
import base64
from io import BytesIO

dashscope.api_key="sk-xxx"

max_requests_per_second = 2
rate_limiter = RateLimiter(max_calls=max_requests_per_second, period=1)


def get_base64_image(path):
    if os.path.exists(path): 
        image = Image.open(path).convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes)
        img_str = img_base64.decode('utf-8')
        return img_str
    else:
       return None  # 如果文件不存在

TIME = 50
@sleep_and_retry
@limits(calls=5, period=TIME) # 
def call_api(model_name,message):
    response = dashscope.MultiModalConversation.call(model=model_name,
                                                                messages=message)
    if response.status_code != 200:
        raise Exception('API response: {}'.format(response.status_code))
    return response

def process_ID(data):
    system="你将扮演一个擅长做数学题的解题助手，根据图文信息解答数学问题，你需要理解图像题目内容含义并结合从图像中识别出来的文字，对问题按照步骤进行解答"     
    demand="你需要结合文本和图片有一个全面综合的理解，然后回答文本中的问题\n------------------------\n注意：回答格式如下：“每道题的解题过程+'\n\n'+每道（）小题的最终答案:【XXX】”。例如有（1），（2），（3）道问题，最终答案需要包括这全部小题的答案如：（1）xx （2）xx （3）xx"     
    #demand="你需要结合文本和图片有一个全面综合的理解，然后回答文本中的问题\n------------------------\n注意：最后输出的答案用json格式，参考格式为{ \"solution\": \"解题过程……\" , \"answer\": \"最终答案\"}"
    
    for key,value in data.items():
        print("item",key)
        id=key
        imageid = value.get("image_id") 
        image_name=imageid+'.png'
        image_path='photograph'+image_name

        ocr=value.get("question")
        standard_ans=value.get("standard_ans")

        img_str = get_base64_image(image_path)  
        if img_str is not None:        
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"image": image_path},
                            {"text": ocr+system+demand}
                        ]
                    }
                ]

                response = call_api('qwen-vl-max',messages) #'qwen-vl-plus'
                print(id,"response ok")
                return response.output.choices[0].message.content[0]['text'], id, ocr,standard_ans
        else:
            # 如果图片不存在，可以选择跳过当前迭代或记录错误等操作
            print(f"Image not found: {image_path}")
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {"text": ocr+system+demand}
                        ]
                    }
                ]
            response = call_api('qwen-vl-max',messages)   
            return response.output.choices[0].message.content[0]['text'], id, ocr,standard_ans
            #continue 


def parallel_call_cache(data,input_file,out_file, max_workers=10):
    '''多线程，实时缓存结果'''
    records=[] 
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_futures = []
        for i, _ in enumerate(data):
            process_futures.append(executor.submit(process_ID, data[i]))
        for future in futures.as_completed(process_futures):
            res,id,ocr,standard_ans = future.result()
            out={id:{'ocr':ocr,'standard_ans':standard_ans,'ans':res}}
            with open("parallel_qwen_plus_ans.jsonl", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(out, ensure_ascii=False) + '\n')
            records.append(out)
    print ("%s done!" % out_file)
input_file='math_question_solution_ans.json'
out_file=''

def get_parallel_answer():
    with open(input_file, 'r') as f:
        data = json.load(f)
    parallel_call_cache(data,input_file,out_file,1)


if __name__ == '__main__':
    get_parallel_answer()


