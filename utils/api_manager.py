import json


def get_api_datas():
    global MODEL_API_LIB
    if 'MODEL_API_LIB' not in globals():
        with open('config/api_keys.json') as f_in:
            MODEL_API_LIB = json.load(f_in)
    return MODEL_API_LIB

def get_keys(model_type):
    api_lib = get_api_datas()
    if model_type not in api_lib or len(api_lib[model_type]) == 0:
        raise "No aviliable api found for {model_type} in models/api_keys.json, you should fetch it by yourself"
    return api_lib[model_type]

def add_key(model_type, key):
    api_lib = get_api_datas()
    if model_type in api_lib:
        # in case some stupid action add duplicate api keys
        old_set = set(api_lib[model_type])
        if isinstance(key, list):
            old_set.union(set(key))
        else:
            old_set.add(key)
        api_lib[model_type] = list(old_set)
    else:
        if isinstance(key, list):
            api_lib[model_type] = list(set(key))
        else:
            api_lib[model_type] = [key]
    save_api_datas()

def delete_key(model_type, key):
    api_lib = get_api_datas()
    if model_type in api_lib:
        # in case some stupid action add duplicate api keys
        old_set = set(api_lib[model_type])
        if isinstance(key, list):
            api_lib[model_type] = list(old_set - set(key))
        else:
            old_set.discard(key)
            api_lib[model_type] = list(old_set)
        save_api_datas()

def save_api_datas():
    with open('config/api_keys.json', 'w') as f_out:
        json.dump(get_api_datas(), f_out, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    api_datas = get_api_datas()
    print(api_datas)
    add_key('openAI', 'new')
    print(get_api_datas())
    delete_key('openAI', 'new')
    print(get_api_datas())
    delete_key('openAI', 'not_exist')
    delete_key('not_exist', 'not_exist')
    print(get_api_datas())

