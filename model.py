import json
import random
from concurrent import futures
from tqdm import tqdm
from session import session
from abc import ABC, abstractmethod
from utils.crawlers import get_RMB_exchange
from utils.utils import uuid
import openai
print(openai.__file__)
import copy
import time
import os
import api_manager
from ratelimit import limits, sleep_and_retry

from PIL import Image
import base64
from io import BytesIO

import requests
from concurrent import futures
from requests.adapters import HTTPAdapter, Retry
os.environ["OPENAI_API_KEY"] = "xxx"
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



# 1. gpt-4-1106-preview
# 2. gpt-4-turbo-2024-04-09
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


def openai_parallel(dataset, processor, threads=5, name=None):
    with futures.ThreadPoolExecutor(threads) as executor, \
         tqdm(total=len(dataset), ncols=0, desc=name) as pbar:
        process_futures = [
            executor.submit(processor, data)
            for data in dataset
        ]
        for future in futures.as_completed(process_futures):
            yield future.result()
            pbar.update()

def thread_parallel(processor, dataset, threads=10, name=None, extra_paras=None):
    with futures.ThreadPoolExecutor(threads) as executor:
        if extra_paras is None:
            process_futures = [executor.submit(processor, data)for data in dataset]
        else:
            process_futures = [executor.submit(processor, data, *extra_paras)for data in dataset]
        for future in tqdm(futures.as_completed(process_futures), desc=name, total=len(dataset)):
            yield future.result()

def process_parallel(processor, dataset, threads=10, name=None, extra_paras=None):
    with futures.ProcessPoolExecutor(threads) as executor:
        if extra_paras is None:
            process_futures = [executor.submit(processor, data)for data in dataset]
        else:
            process_futures = [executor.submit(processor, data, *extra_paras)for data in dataset]
        for future in tqdm(futures.as_completed(process_futures), desc=name, total=len(dataset)):
            yield future.result()


class Model(ABC):
    def __init__(self, server, name, stream=True, timeout=120, temperature=0.3):
        self.server = server
        self.name = name
        self.stream = stream
        self.timeout = timeout
        self.temperature = temperature
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2)
    
    @property
    def url(self):
        url = f'http://{self.server}'
        if self.stream:
            url += '/generate_stream'
        else:
            url += '/generate'
        return url
    
    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_stream(self, stream):
        self.stream = stream
    
    @abstractmethod
    def call(self, prompt, history=[], system=None, qtype=''):
        return None

    def process_data(self, data, use_model_gen=True):
        try:
            if 'messages' in data or isinstance(data, list): # it's multi turn
                prompt, qtype, history = None, '', []
                # write ans to 'model' key if all use model gen(which data usually don't have this key), else 'content' key(which data usually have)
                insert_key = 'model' if use_model_gen else 'content'
                system = None
                for message in data['messages'] if 'messages' in data else data:
                    if message['role'] == 'user':
                        # if the user speak continuously
                        if prompt is not None:
                            history.append([prompt, None])
                        prompt = message['content']
                        qtype = message.get('qtype', '')
                    elif message['role'] == 'assistant':
                        if insert_key not in message:
                            message[insert_key] = self.call(prompt=prompt, history=history, system=system, qtype=qtype)
                        history.append([prompt, message[insert_key]])
                        prompt = None
                    # currently only support 1 turn system
                    elif message['role'] in {'system', 'user_system'} and system is None:
                        system = message['content']
                # if the role of last turn is 'user', we should generate answer for it
                if prompt is not None:
                    last_turn_answer = {'role': 'assistant', insert_key: self.call(prompt=prompt, history=history, system=system, qtype=qtype)}
                    if 'messages' in data:
                        data['messages'].append(last_turn_answer)
                    else:
                        data.append(last_turn_answer)
            else: # single turn
                if 'model' not in data:
                    data['model'] = self.call(prompt=data['content'], history=[], system=data.get('system', None), qtype=data.get('qtype', ''))
        except Exception as e:
            print(f"{data}发生错误：{e}", flush=True)
        return data


class BCDockerModel(Model):
    def __init__(
        self,
        server,
        name,
        stream = True,
        timeout = 120,
        max_new_tokens = 2048,
        temperature = 0.3,
        top_k = 5,
        top_p = 0.85,
        repetition_penalty = 1.05,
        do_sample = True
    ):
        super().__init__(server, name, stream, timeout, temperature)
        self.parameters = {
            'max_new_tokens': max_new_tokens,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'do_sample': do_sample
        }
        self.price = 0.01 / 1000
        self.rdm = random.Random()

    def set_top_p(self, top_p):
        self.parameters['top_p'] = top_p

    def set_top_k(self, top_k):
        self.parameters['top_k'] = top_k

    def call_prompt_details(self, input_prompt, parameters=None):
        call_parameters = self.parameters
        call_parameters['temperature'] = self.temperature
        call_parameters['details'] = True
        if parameters is not None and isinstance(parameters, dict):
            call_parameters.update(parameters)
        rdm_int = self.rdm.randint(-65535, 65535)
        response = session.post(
            self.url,
            json={'inputs': input_prompt, 'parameters': call_parameters, 'seed': rdm_int},
            stream=self.stream,
            timeout=self.timeout,
        )
        if self.stream:
            answer = ''
            for byte_line in response.iter_lines():
                byte_line = byte_line.rstrip(b'\n')
                if byte_line.startswith(b'data:'):
                    prompt = json.loads(byte_line.decode().lstrip('data:'))
                    #answer += prompt['token']['text']
            answer = prompt
        else:
            answer = response.json()
        
        res = {}
        res['status'] = 'finish'
        res['text'] = answer['generated_text']
        prompt_tokens = answer['details']['prefill_tokens']
        completion_tokens = answer['details']['generated_tokens']
        res['prompt_tokens'] = prompt_tokens
        res['completion_tokens'] = completion_tokens
        res['details'] = answer['details']
        res['cost'] = self.price * (prompt_tokens + completion_tokens)
        return res
    
    def get_top_responses(self, prompt, top_k, history=[], system=None):
        if self.stream:
            print('Warrning. The `get_top_responses` method need to be launched with stream=False, but the current setting is True. Changed to false now')
            self.stream = False
        parameters = {'best_of': top_k}
        resp = self.call_details(prompt, history=history, system=system, parameters=parameters)
        out_res = [resp['text']]
        for res in resp['details']['best_of_sequences']:
            out_res.append(res['generated_text'])
        return out_res
    
    def call_details(self, prompt, history=[], system=None, qtype='', parameters=None):

        print("?what",type(prompt))
        input_str = ""
        if system is not None:
            input_str += '<B_SYS>' + system
        for query, answer in history:
            input_str += '<H_Q>' + query
            if answer is not None:
                input_str += '<H_A>' + answer
        input_str += '<C_Q>' + prompt + '<C_A>'
        return self.call_prompt_details(input_str, parameters)

    def call(self, prompt, history=[], system=None, qtype=''):
        return self.call_details(prompt, history, system, qtype)['text']
    
    def get_ans_logps(self, prompt, response, history=[], system=None):
        input_str = ""
        if system is not None:
            input_str += '<B_SYS>' + system
        for query, answer in history:
            input_str += '<H_Q>' + query
            if answer is not None:
                input_str += '<H_A>' + answer
        input_str += '<C_Q>' + prompt + '<C_A>'

        parameters = self.parameters
        parameters['temperature'] = self.temperature
        parameters['details'] = True
        parameters['decoder_input_details'] = True
        parameters['max_new_tokens'] = 1
        rdm_int = self.rdm.randint(-65535, 65535)
        response = session.post(
            f'http://{self.server}/generate',
            json={'inputs': input_str + response + '<|endoftext|>', 'parameters': parameters, 'seed': rdm_int},
            stream=False,
            timeout=self.timeout,
        )
        prefill_logp = response.json()['details']['prefill']
        for i, logp_ele in enumerate(prefill_logp):
            if logp_ele['id'] == 196:
                return [k['logprob'] for k in prefill_logp[i+1:]]
        return None
        
    
class BCProxyModel(Model):
    def __init__(self, server, name, stream=True, timeout=120, temperature=0.3):
        super().__init__(server, name, stream, timeout, temperature)
        self.rdm = random.Random()

    def call(self, prompt, history=[], system=None, qtype=''):
        rdm_int = self.rdm.randint(-65535, 65535)
        response = session.post(
            self.url,
            json={
                'req_id': str(random.randint(10000000, 99999999)),
                'input': prompt,
                'history': history,
                'seed': rdm_int,
                'temperature': self.temperature,
                'qtype': qtype
            },
            stream=self.stream,
            timeout=self.timeout,
        )
        if self.stream:
            answer = ''
            for byte_line in response.iter_lines():
                prompt = json.loads(byte_line)
                answer += prompt['data']['word'].replace('<emo>','')
        else:
            answer = response.json()['generated_text']
        return answer
    

class VLLMModel(Model):
    def __init__(
        self,
        server,
        name,
        stream = True,
        timeout = 120,
        max_new_tokens = 2048,
        temperature = 0.3,
        top_k = 5,
        top_p = 0.85,
        repetition_penalty = 1.05,
        do_sample = True
    ):
        super().__init__(server, name, stream, timeout, temperature)
        self.parameters = {
            'max_new_tokens': max_new_tokens,
            'top_k': top_k,
            'top_p': top_p,
            'repetition_penalty': repetition_penalty,
            'do_sample': do_sample
        }
        self.rdm = random.Random()

    def set_api_key(self, api_key):
        self.api_key = api_key
        api_manager.add_key("openAI", api_key)
    
    @property
    def url(self):
        return f'http://{self.server}/v1/chat/completions'
    
    def call(self, prompt, history=[], system=None, qtype=''):
        if system:
            messages = [{'role': 'system', 'content': system}]
        else:
            messages = []
        for user, assistant in history:
            messages.extend([
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant}
            ])
        messages.append({'role': 'user', 'content': prompt})
        rdm_int = self.rdm.randint(-65535, 65535)

        response = session.post(
                url = self.url,
                json = {
                    'model': self.name,
                    'messages': messages,
                    'seed': rdm_int,
                    'temperature': self.temperature,
                    **self.parameters
                },
                stream=self.stream,
                timeout=self.timeout
        )
        print(response)
        return response.json()['choices'][0]['message']['content']
    
    
class openAIModel(Model):
    def __init__(self, server='xx.xxx.xxx.xxx:xx', name='gpt-4-1106-preview', stream=False, timeout=180, temperature=1.0, json_mode=False):
        super().__init__(server, name, stream, timeout, temperature)
        self.json_mode = json_mode
        self.api_key = api_manager.get_keys('openAI')
        self.input_price, self.output_price = self.init_price()

    @property
    def url(self):
        return f'http://{self.server}/v1/chat'

    def set_api_key(self, api_key):
        self.api_key = api_key
        api_manager.add_key("openAI", api_key)

    def init_price(self):
        dollar = 7.1#get_RMB_exchange('美元')
        if self.name == 'gpt-4-1106-preview' or self.name == 'gpt-4-1106-vision-preview':
            return dollar * 0.01 / 1000, dollar * 0.03 / 1000
        elif self.name == 'gpt-4':
            return dollar * 0.03 / 1000, dollar * 0.06 / 1000
        elif self.name == 'gpt-4-32k':
            return dollar * 0.06 / 1000, dollar * 0.12 / 1000
        elif self.name == 'gpt-3.5-turbo-1106':
            return dollar * 0.001 / 1000, dollar * 0.002 / 1000
        elif self.name == 'gpt-3.5-turbo-instruct':
            return dollar * 0.0015 / 1000, dollar * 0.002 / 1000
        # default return gpt3.5-turbo cost
        return dollar * 0.001 / 1000, dollar * 0.002 / 1000
    
    def call_details(self, prompt, history=[], system=None, qtype=''):
        openai.api_base = self.url
        messages = []
        if system is not None:
            messages.append({'role': 'system', 'content': system})
        for query, answer in history:
            messages.append({'role': 'user', 'content': query})
            if answer:
                messages.append({'role': 'assistant', 'content': answer})
        messages.append({'role': 'user', 'content': prompt})

        data = dict(
                model=self.name,
                messages=messages,
                temperature=self.temperature
            )
        if self.json_mode:
            data['response_format'] = { "type": "json_object" }

        api_key = None
        if self.api_key is not None:
            if isinstance(self.api_key, str):
                api_key = self.api_key
            else:
                api_key = random.choice(self.api_key)
        if api_key:
            openai.api_key = api_key
        else:
            raise 'No aviliable "openAI" api_key provided in the config/api_keys.json! You should provide them in the file'

        ntry = 0
        while ntry < 1000:
            try:
                return self.chat_func(data)
            except Exception as e:
                if type(e) not in [openai.error.ServiceUnavailableError, openai.error.RateLimitError, openai.error.APIConnectionError]:
                    print("access openai error, %s" % (e), type(e))
                    return {'status': 'failed', 'text': None}
                ntry += 1
        print('try too much times')
        return {'status': 'failed', 'text': None}
    
    def chat_func(self, data):
        response = openai.Completion.create(**data)
        if 'choices' not in response:
            print("access openai error, status code: %s，errmsg: %s" % (response.status_code, response.text))
            return {'status': 'failed', 'text': None}
        finish_reason = response['choices'][0]['finish_reason']
        if finish_reason != 'stop':
            print(f'Unexpected finish reason! The finish_reason={finish_reason}')
            return {'status': 'unfinish', 'text': None}
        res = {}
        res['status'] = 'finish'
        res['text'] = response['choices'][0]['message']['content']
        return res
    
    def call(self, prompt, history=[], system=None, qtype=''):
        response = self.call_details(prompt, history, system, qtype)
        return response['text']


class GLMModel(Model):
    def __init__(self, server, name, timeout=180):
        super().__init__(server, name, stream=True, timeout=timeout)
        self.api_key = api_manager.get_keys('glm')
        self.access_key = {}
        self.header = {
            'Accept' : 'text/event-stream',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection':'keep-alive',
            'Content-Length':'411',
            'Content-Type':'application/json',
            'Host':'chatglm.cn',
            'Origin':'https://chatglm.cn',
            'Referer':'https://chatglm.cn/main/alltoolsdetail',
            'Sec-Ch-Ua':'"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile':'?0',
            'Sec-Ch-Ua-Platform':'"macOS"',
            'Sec-Fetch-Dest':'empty',
            'Sec-Fetch-Mode':'cors',
            'Sec-Fetch-Site':'same-origin',
            'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        self.input_template = {
            "assistant_id": "65940acff94777010aa6b796",
            "conversation_id": "",
            "meta_data": {
                "is_test":False,
                "input_question_type": "xxxx",
                "channel": ""
            }
        }

    @property
    def url(self):
        return f'https://{self.server}/chatglm/backend-api/assistant/stream'

    def set_api_key(self, api_key):
        self.api_key = api_key
        api_manager.add_key('glm', api_key)

    def get_access_key(self, api_key, force_refresh=False):
        current_time = int(time.time())
        # refresh the access_key 10s before it expires
        if api_key not in self.access_key or self.access_key[api_key][1] - 10 < current_time or force_refresh:
            header = copy.deepcopy(self.header)
            header['referer'] = 'https://chatglm.cn/main/alltoolsdetail'
            header['Authorization'] = f'Bearer {api_key}'
            header['X-Device-Id'] = uuid(no_slash=True)
            header['X-Request-Id'] = uuid(no_slash=True)
            # try 3 times, if all failed, delete that api_key
            request_failed = True
            for _ in range(3):
                try:
                    refresh_resp = session.post(url='https://chatglm.cn/chatglm/backend-api/v1/user/refresh', 
                                                json={},
                                                headers=header, 
                                                timeout=30)
                    if refresh_resp.status_code == 200:
                        data = refresh_resp.json()['result']
                        if 'accessToken' in data:
                            key, expire_time = data['accessToken'], int(time.time()) + 3600
                            self.access_key[api_key] = [key, expire_time]
                            request_failed = False
                            break
                    else:
                        print(f'Error when trying to glm api access token, response_code={refresh_resp.status_code}, data={refresh_resp.json()}')
                except Exception as e:
                    print(f'Error when trying to glm api access token, error={e}')
                    return None
            if request_failed:
                print(f'Error: refresh glm api tokens failed. You may need to try to get it using browser. Now delete the key {api_key}')
                api_manager.delete_key('glm', api_key)
                return None
        return self.access_key[api_key][0]

    def retry_and_call(self, func, api_key, retry_num=3, **kwargs):
        for _ in range(retry_num):
            response = func(**kwargs)
            if response.status_code == 401:
                access_key = self.get_access_key(api_key, force_refresh=True)
                kwargs['headers']['Authorization'] = f'Bearer {access_key}'
            elif response.status_code == 200:
                return response
            else:
                print("access glm error, status code: %s，errmsg: %s" % (response.status_code, response.text))
                time_sleep = random.uniform(0.2, 2)
                time.sleep(time_sleep)
                return {'status': 'failed', 'text': None}
                
    def call_details(self, prompt, history=[], system=None, qtype=''):
        if self.api_key is not None:
            if isinstance(self.api_key, str):
                api_key = self.api_key
            else:
                api_key = random.choice(self.api_key)
        
        header = copy.deepcopy(self.header)    
        if api_key:
            access_key = self.get_access_key(api_key)
            header['Authorization'] = f'Bearer {access_key}'
            header['X-Device-Id'] = uuid(no_slash=True)
            header['X-Request-Id'] = uuid(no_slash=True)
        else:
            raise 'No aviliable "glm" api_key provided in the config/api_keys.json! You should provide them in the file'
        messages = []
        content = {
            "type": "text",
            "text": prompt
        }
        for query, answer in history:
            messages.append({'role': 'user', 'content': query})
            if answer:
                messages.append({'role': 'assistant', 'content': answer})
        messages.append({"role": "user", "content": [content]})

        data = copy.deepcopy(self.input_template)
        data['messages'] = messages
        answer = ''
        answer_details = []
        res = {}
        try:
            response = self.retry_and_call(session.post, api_key, 3,
                url=self.url,
                headers = header,
                json=data,
                timeout=self.timeout,
                stream=self.stream
            )
            if response.status_code != 200:
                print("access chatglm error, status code: %s，errmsg: %s" % (response.status_code, response.text))
                time_sleep = random.uniform(0.2, 2)
                time.sleep(time_sleep)
                return {'status': 'failed', 'text': None}
            for byte_line in response.iter_lines():
                byte_line = byte_line.rstrip(b'\n')
                if byte_line.startswith(b'data:'):
                    prompt = json.loads(byte_line.decode().lstrip('data:'))
                    # remove the last generated one since it have two lines with finish
                    if prompt.get('status', 'init') == 'finish':
                        continue
                    if 'parts' in prompt and len(prompt['parts']) > 0 and 'status' in prompt['parts'][-1] and prompt['parts'][-1]['status'] == 'finish':
                        print(json.dumps(prompt, ensure_ascii=False))
                        c = ''
                        data = prompt['parts'][-1]['content'][-1]
                        answer_details.append(data)
                        if data['type'] == 'text' or data['type'] == 'code':
                            c = data[data['type']]
                        elif data['type'] == 'image':
                            c = data['image'][-1]['image_url']
                        elif data['type'] == 'browser_result' or data['type'] == 'execution_output':
                            c = data['content']
                        elif data['type'] == 'system_error' or data['type'] == 'tool_calls' or data['type'] == 'quote_result':
                            pass
                        else:
                            print(f'{data["type"]} is not processed')
                            print('input prompt=', prompt)
                            print(json.dumps(data, ensure_ascii=False))
                            break
                        answer += c
            res['status'] = 'finish'
            res['text'] = answer
            res['details'] = answer_details
            res['cost'] = 0
        except Exception as e:
            print(f'call model failed. Error is {e}')
            print(f'res_data = {prompt}')
        time_sleep = random.uniform(0.2, 2)
        time.sleep(time_sleep)
        return res if res else {'status': 'failed', 'text': None}
    
    def call(self, prompt, history=[], system=None, qtype=''):
        response = self.call_details(prompt, history, system, qtype)
        return response['text']


class MoonshotModel(Model):
    def __init__(self, server, name, use_search=False, timeout=180):
        super().__init__(server, name, stream=True, timeout=timeout)
        self.header = {
            'authority': 'kimi.moonshot.cn',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'content-type': 'application/json',
            'origin': 'https://kimi.moonshot.cn',
            'r-timezone': 'Asia/Shanghai',
            'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        }
        self.input_template = {}
        self.set_use_search(use_search)
        self.api_key = api_manager.get_keys('moonshot')
        self.access_key = {}

    def set_use_search(self, use=True):
        self.input_template["use_search"] = use

    @property
    def url(self):
        return 'https://' + self.server + '/api/chat/{conv_id}/completion/stream'
    
    def get_access_key(self, api_key, force_refresh=False):
        current_time = int(time.time())
        # refresh the access_key 10s before it expires
        if api_key not in self.access_key or self.access_key[api_key][1] - 10 < current_time or force_refresh:
            header = copy.deepcopy(self.header)
            header['referer'] = 'https://kimi.moonshot.cn/'
            header['Authorization'] = f'Bearer {api_key}'
            # try 3 times, if all failed, delete that api_key
            request_failed = True
            for _ in range(3):
                try:
                    refresh_resp = session.get(url='https://kimi.moonshot.cn/api/auth/token/refresh', headers=header, timeout=10)
                    if refresh_resp.status_code == 200:
                        data = refresh_resp.json()
                        if 'access_token' in data:
                            key, expire_time = data['access_token'], int(time.time()) + 300
                            self.access_key[api_key] = [key, expire_time]
                            request_failed = False
                            break
                    else:
                        print(f'Error when  trying to moonshot api access token, response_code={refresh_resp.status_code}, data={refresh_resp.json()}')
                except Exception as e:
                    print(f'Error when trying to moonshot api access token, error={e}')
                    return None
            if request_failed:
                print(f'Error: refresh moonshot api tokens failed. You may need to try to get it using browser. Now delete the key {api_key}')
                api_manager.delete_key('moonshot', api_key)
                return None
        return self.access_key[api_key][0]

    def set_api_key(self, api_key):
        self.api_key = api_key
        api_manager.add_key('moonshot', api_key)

    def retry_and_call(self, func, api_key, retry_num=3, **kwargs):
        for _ in range(retry_num):
            response = func(**kwargs)
            if response.status_code == 401:
                access_key = self.get_access_key(api_key, force_refresh=True)
                kwargs['headers']['Authorization'] = f'Bearer {access_key}'
            elif response.status_code == 200:
                return response
            else:
                print("access moonshot error, status code: %s，errmsg: %s" % (response.status_code, response.text))
                time_sleep = random.uniform(0.2, 2)
                time.sleep(time_sleep)
                return {'status': 'failed', 'text': None}

    def create_new_conv(self, api_key):
        try:
            header = copy.deepcopy(self.header)
            header['referer'] = 'https://kimi.moonshot.cn/'
            access_key = self.get_access_key(api_key)
            header['Authorization'] = f'Bearer {access_key}'
            data = {
                'is_example': False,
                'name': '未命名会话'
            }
            response = self.retry_and_call(session.post, api_key, 3,
                                           url='https://kimi.moonshot.cn/api/chat', 
                                           headers=header, 
                                           json=data, 
                                           timeout=self.timeout)
            return response.json()['id']
        except Exception as e:
            print(f'Failed to create the new conversation. Error={e}')

    def parse_response(self, response):
        resp = {'details': {}}
        resp_list = []
        for byte_line in response.iter_lines():
            byte_line = byte_line.rstrip(b'\n')
            if byte_line.startswith(b'data:'):
                chunk = json.loads(byte_line.decode().lstrip('data:'))
                if 'event' not in chunk:
                    continue
                if chunk['event'] == 'cmpl':
                    resp_list.append(chunk['text'])
                elif chunk['event'] == 'all_done':
                    resp['status'] = 'finish'
                elif chunk['event'] == 'error':
                    resp['status'] = 'interrupted'
                elif chunk['event'] == 'search_plus' and 'msg' in chunk and chunk['msg']['type'] == 'get_res':
                    resp['details']['refer'] = {'title': chunk['msg']['title'], 'url': chunk['msg']['url']}
        resp['text'] = ''.join(resp_list)
        resp['cost'] = 0
        return resp
    
    def call_details(self, prompt, history=[], system=None, qtype=''):
        if self.api_key is not None:
            if isinstance(self.api_key, str):
                api_key = self.api_key
            else:
                api_key = random.choice(self.api_key)
        if api_key:
            access_key = self.get_access_key(api_key)
            header = copy.deepcopy(self.header)
            header['Authorization'] = f'Bearer {access_key}'
        else:
            raise 'No aviliable "moonshot" api_key provided in the config/api_keys.json! You should provide them in the file'

        messages = []
        if system is not None:
            print('Warning: The moonshot model don\'t support the system, your system msg won\'t take effect')
        for query, answer in history:
            messages.append({'role': 'user', 'content': query})
            if answer:
                messages.append({'role': 'assistant', 'content': answer})
        messages.append({"role": "user", "content": prompt})

        data = copy.deepcopy(self.input_template)
        data['messages'] = messages
        res = {}
        try:
            new_conv = self.create_new_conv(api_key)
            header['referer'] = f'https://kimi.moonshot.cn/chat/{new_conv}'
            response = self.retry_and_call(session.post, api_key, 3, 
                                           url=self.url.format(conv_id=new_conv), 
                                           headers=header, 
                                           json=data, 
                                           timeout=self.timeout)
            res = self.parse_response(response)
        except Exception as e:
            print(f'call model failed. Error is {e}')
            print(f'res_data = {prompt}')
        time_sleep = random.uniform(0.2, 2)
        time.sleep(time_sleep)
        return res if res else {'status': 'failed', 'text': None}
    
    def call(self, prompt, history=[], system=None, qtype=''):
        response = self.call_details(prompt, history, system, qtype)
        return response['text']
    

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



TIME = 10
@sleep_and_retry
@limits(calls=10, period=TIME) 
def call_api(prompt,sys):
    model = get_model('gpt4-turbo-vision')
    response = model.call(prompt, system=sys, history=[])
    return response

def process_ID(data):
    res=None
    system="你将扮演一个擅长做数学题的解题助手，根据图文信息解答数学问题，你需要理解图像题目内容含义并结合从图像中识别出来的文字，对问题按照步骤进行解答"     
    demand="你需要结合文本和图片有一个全面综合的理解，然后回答文本中的问题\n------------------------\n注意：最后输出的答案用json格式，参考格式为{ \"solution\": \"解题过程……\" , \"answer\": \"最终答案\"}"      
    model = get_model('gpt4-turbo-vision')
    for key,value in data.items():
        id=key
        imageid = value.get("image_id") 
        image_name=imageid+'.png'
        image='/photograph/'+image_name
        ocr=value.get("question")
        standard_ans=value.get("standard_ans")
        img_str = get_base64_image(image)  
        if img_str is not None:        
            # 如果成功获取到图片，继续后续操作
                prompt = [
                { 
                    "type": "text",
                    "text": f"下面是是图片和OCR文字```\nOCR文字\n```\n{ocr}\n```\nOCR文字描述了数学问题的内容."
            
            
                },
                { 
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{get_base64_image(image)}", 
                        'detail': 'high'
                    } 
                },
                ]           
                res= model.call(prompt, system=system+demand, history=[])
                while not res:
                    res = model.call(prompt, system=system+demand, history=[])

                return res,id, ocr,standard_ans
        else:
            # 如果图片不存在，可以选择跳过当前迭代或记录错误等操作
            print(f"Image not found: {image}")
            res= model.call(ocr, system=system+demand, history=[])
            while not res:
                res = model.call(ocr, system=system+demand, history=[])
            return res, id, ocr,standard_ans

def parallel_call_cache(data,input_file,out_file, max_workers=10):
    '''多线程，实时缓存结果'''
    cnt = 0
    records=[] 
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_futures = []
        for i, _ in enumerate(data):
            process_futures.append(executor.submit(process_ID, data[i]))
        for future in futures.as_completed(process_futures):
            res,id,ocr,standard_ans = future.result()
            out={id:{'ocr':ocr,'standard_ans':standard_ans,'ans':res}}
            with open("parallel_gpt4v_ans.jsonl", 'a', encoding='utf-8') as f:
                    f.write(json.dumps(out, ensure_ascii=False) + '\n')
            records.append(out)
            cnt += 1         
    print ("%s done!" % out_file) 

def get_model(model_name, temperature=None, timeout=None):
    global ALL_MODELS
    if 'ALL_MODELS' not in globals():
        with open('config/models.json') as f_in:
            ALL_MODELS = json.load(f_in)
    if model_name not in ALL_MODELS:
        raise Exception(f"No model named {model_name} in models/models.json")
    model_config = ALL_MODELS[model_name]
    if model_config['type'] == 'openAI':
        return openAIModel(server=model_config.get('server', 'xx.xxx.xxx.xxx:xx'), 
                            name=model_config['name'], 
                            stream=model_config.get('stream', False), 
                            timeout=timeout if timeout else model_config.get('timeout', 220), #120->220
                            temperature=temperature if temperature else model_config.get('temperature', 0.3),
                            json_mode=model_config.get('json_mode', False))
    elif model_config['type'] == 'BCDocker':
        return BCDockerModel(server=model_config['server'], 
                            name=model_name, 
                            stream=model_config.get('stream', True), 
                            timeout=timeout if timeout else model_config.get('timeout', 120),
                            max_new_tokens=model_config.get('max_new_tokens', 2048),
                            temperature=temperature if temperature else model_config.get('temperature', 0.3),
                            top_k=model_config.get('top_k', 5),
                            top_p=model_config.get('top_p', 0.85),
                            repetition_penalty=model_config.get('repetition_penalty', 1.05),
                            do_sample=model_config.get('do_sample', True))
    elif model_config['type'] == 'vllm':
        return VLLMModel(server=model_config['server'], 
                            name=model_name, 
                            stream=model_config.get('stream', True), 
                            timeout=timeout if timeout else model_config.get('timeout', 120),
                            max_new_tokens=model_config.get('max_new_tokens', 2048),
                            temperature=temperature if temperature else model_config.get('temperature', 0.3),
                            top_k=model_config.get('top_k', 5),
                            top_p=model_config.get('top_p', 0.85),
                            repetition_penalty=model_config.get('repetition_penalty', 1.05),
                            do_sample=model_config.get('do_sample', True))
    elif model_config['type'] == 'glm':
        return GLMModel(server=model_config['server'], 
                        name=model_config['name'], 
                        timeout=timeout if timeout else model_config.get('timeout', 180))
    elif model_config['type'] == 'moonshot':
        return MoonshotModel(server=model_config['server'], 
                            name=model_config['name'], 
                            use_search=model_config.get('use_search', False),
                            timeout=timeout if timeout else model_config.get('timeout', 240))
    elif model_config['type'] == 'BCProxy':
        return BCProxyModel(server=model_config['server'], 
                            name=model_name)

input_file='math_question_solution_ans.json'
out_file=''

def get_parallel_answer():
    with open(input_file, 'r') as f:
        data = json.load(f)
    parallel_call_cache(data,input_file,out_file,1)


if __name__ == '__main__':
    # 0.28.0 version for openai when using this codebase
    get_parallel_answer()
 

   
    