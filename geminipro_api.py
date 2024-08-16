import requests
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
import json
import os
from PIL import Image
import base64
from io import BytesIO

import google.generativeai as genai
import os
import json
import PIL.Image
from IPython.display import display
from IPython.display import Markdown
from ratelimit import limits, sleep_and_retry

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HTTP_PROXY"] = "http://xx.xxx.xxx.xxx:xxxxx"
os.environ["HTTP_PROXYS"] = "http://xx.xxx.xxx.xxx:xxxxx"


# 60次/min
os.environ["GEMINI_API_KEY"]="xxxxx"

 
gemini_api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key = gemini_api_key)



TIME = 10
@sleep_and_retry
@limits(calls=2, period=TIME) # 10,100 / s   google free: 10 次/s
def call_api(model_name,message):
    model = genai.GenerativeModel(model_name)#('models/gemini-pro-vision')
    response = model.generate_content(message,stream=True)
    return response


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

def process_ID(data):
    res=None
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
        model = genai.GenerativeModel('gemini-pro-vision')
        img_str = get_base64_image(image_path)  
        if img_str is not None:        
                img = PIL.Image.open(image_path)
                response = call_api('models/gemini-pro-vision',[ocr+system+demand,img])
                while not response:
                    response = call_api('models/gemini-pro-vision',[ocr+system+demand,img])

                response.resolve()
                print(id,"response ok")

                return response.candidates[0].content.parts[0].text, id, ocr,standard_ans
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
            response = call_api('models/gemini-pro',ocr+system+demand)
            while not response:
                    response = call_api('models/gemini-pro',ocr+system+demand)
                    response.resolve()
            response.resolve()
            return response.candidates[0].content.parts[0].text, id, ocr,standard_ans

# 检查id是否已经存在于文件中
def check_id_exists(file_name, id):
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if str(id) in line:
                return True
    return False


def parallel_call_cache(data,input_file,out_file, max_workers=10):
    '''多线程，实时缓存结果'''
    cnt = 0
    #answers_dict={}
    records=[] 
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_futures = []
        for i, _ in enumerate(data):
            process_futures.append(executor.submit(process_ID, data[i]))
        for future in futures.as_completed(process_futures):
            res,id,ocr,standard_ans = future.result()
            out={id:{'ocr':ocr,'standard_ans':standard_ans,'ans':res}}
            with open("parallel_geminipro_ans.jsonl", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(out, ensure_ascii=False) + '\n')
            records.append(out)
            cnt += 1
    print ("%s done!" % out_file)

input_file='math_question_solution_ans.json'
out_file=''

def get_parallel_answer():
    with open(input_file, 'r') as f:
        data = json.load(f)
    parallel_call_cache(data,input_file,out_file,1)

    
        

if __name__ == '__main__':
    get_parallel_answer()



