
from model import get_model
import copy
from abc import ABC, abstractmethod

class Judge(ABC):
    def __init__(self, model_name, prompt, parse_func, retry_num=3, timeout=None, return_details=False):
        self.prompt = prompt
        self.model = get_model(model_name, timeout=timeout)
        self.retry_num = retry_num
        self.parse_func = parse_func
        self.return_details = return_details

    def set_model(self, model_name, timeout=None):
        self.model = get_model(model_name, timeout=timeout)

    def get_model(self):
        return self.model
    
    def get_prompt(self):
        return self.prompt
    
    def call_model(self, input_prompt, info_dict, system=None):
        if self.return_details:
            response = self.model.call_details(input_prompt, history=[], qtype=info_dict.get('qtype', ''), system=system)
            result = self.parse_func(info_dict, response['text'])
            if result is None:
                return None
            detailed_result = {'result': result}
            for k, v in response.items():
                if k != 'text':
                    detailed_result[k] = v
            return detailed_result
        else:
            response = self.model.call(input_prompt, history=[], qtype=info_dict.get('qtype', ''))
            return self.parse_func(info_dict, response)

    @abstractmethod
    def __call__(self, info_dict):
        return None

class OneStageJudge(Judge):
    def __init__(self, model_name, prompt, parse_func, retry_num=3, timeout=None, return_details=False):
        super().__init__(model_name, prompt, parse_func, retry_num, timeout, return_details)

    def __call__(self, info_dict, system=None):
        input_prompt = self.prompt.format_map(info_dict)
        for _ in range(self.retry_num):
            result = self.call_model(input_prompt, info_dict, system=system)
            if result is not None:
                return result
        return None


class TwoStageJudge(Judge):
    def __init__(self, model_name, expert_num, prompt, prompt_stage2, parse_func, retry_num=3, timeout=None, return_details=False):
        super().__init__(model_name, prompt, parse_func, retry_num, timeout, return_details)
        self.expert_num = expert_num
        self.prompt_stage2 = prompt_stage2
    
    def __call__(self, info_dict, system=None):
        experts = []
        if self.return_details:
            total_res = {'cost': 0} 
        while len(experts) < self.expert_num:
            expert_res = self._judge(info_dict)
            if expert_res:
                if self.return_details:
                    total_res['cost'] += expert_res['cost']
                    expert = expert_res['result']
                else:
                    expert = expert_res
                experts.append(f'专家打分{len(experts)+1}：{expert}')
        tmp_dict = info_dict
        tmp_dict['expert_num'] = len(experts)
        tmp_dict['experts'] = '\n'.join(experts)
        input_prompt = self.prompt_stage2.format_map(info_dict)

        for _ in range(self.max_try):
            result = self.call_model(input_prompt, info_dict, system=system)
            if result is not None:
                if self.return_details:
                    total_res['cost'] += result['cost']
                    total_res['result'] = result
                    return total_res
                else:
                    return result
        return None

    def _judge(self, info_dict, system=None):
        input_prompt = self.prompt.format_map(info_dict)
        for _ in range(self.max_try):
            response = self.call_model(input_prompt, info_dict, system=system)
            if response is not None:
                return response
        return None

class RepeatJudge(Judge):
    def __init__(self, model_name, prompt, parse_func, repeat_num=3, retry_num=3, timeout=None, return_details=False):
        super().__init__(model_name, prompt, parse_func, retry_num, timeout, return_details)
        self.repeat_num = repeat_num

    def __call__(self, info_dict, system=None):
        input_prompt = self.prompt.format_map(info_dict)
        results = []
        if self.return_details:
            total_res = {'cost': 0} 
        for _ in range(self.retry_num):
            for _ in range(self.repeat_num):
                res = self.call_model(input_prompt, info_dict, system=system)
                if res is not None:
                    if self.return_details:
                        total_res['cost'] += res['cost']
                        results.append(res['result'])
                        if len(results) >= self.repeat_num:
                            total_res['result'] = int(0.5 + sum(results) / len(results))
                            return total_res
                    else:
                        results.append(res)
                        if len(results) >= self.repeat_num:
                            return sum(results) / len(results)
        return None


class CompareJudge(Judge):
    def __init__(self, model_name, prompt, parse_func, reverse_func, combine_func, retry_num=3, timeout=None, return_details=False):
        super().__init__(model_name, prompt, parse_func, retry_num, timeout, return_details)
        self.reverse_func = reverse_func
        self.combine_func = combine_func

    def __call__(self, info_dict, system=None):
        tmp_dict1 = copy.deepcopy(info_dict)
        scores1 = self._call_step(tmp_dict1, self.prompt.format_map(tmp_dict1), system=system)
        tmp_dict2 = copy.deepcopy(info_dict)
        self.reverse_func(tmp_dict2)
        scores2 = self._call_step(tmp_dict2, self.prompt.format_map(tmp_dict2), system=system)
        if not self.return_details:
            return self.combine_func(scores1, scores2)
        else:
            if scores1 is None and scores2 is None:
                return None
            elif scores1 is None:
                return {'cost': scores2['cost'], 'result': scores2['result']}
            else:
                total_res = {'cost': scores1['cost'] + scores2['cost'], 'result': self.combine_func(scores1['result'], scores2['result'])}
                return total_res 
        return None

    def _call_step(self, info_dict, input_prompt, system):
        for _ in range(self.retry_num):
            result = self.call_model(input_prompt, info_dict, system=system)
            if result is not None:
                return result
        return None