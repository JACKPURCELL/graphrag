import pandas as pd
import re
import glob
import os
import json

def apply_ifelse(row):
    dic = {"GEO":"G",
           "PERSON":"P",
           "EVENT":"E",
           "ORGANIZATION":"G",
           "Unknown":"U"}
    index = row.name
    return dic[row["type"]] + str(index).zfill(4)

def match_case(word, code):
    if word.isupper():
        return code.upper()
    elif word.istitle():
        return code.title()
    else:
        return code.lower()


def gen_replacements(csv_file):
    df = pd.read_csv(csv_file)
    df_name_type = df.loc[:,["name","type"]]
    df_name_type = df_name_type.fillna("Unknown")
    df_name_type['fake_id'] = df_name_type.apply(apply_ifelse, axis=1)

    replacement_list = list(zip(df_name_type['name'], df_name_type['fake_id']))
    replacement_list_sorted = sorted(replacement_list, key = lambda x: (len(x[0]), x[0]), reverse= True)
    return replacement_list_sorted


def replace_terms(text, replacements):
    for term, code in replacements:
        text = re.sub(r'\b' + re.escape(term) + r'\b', 
                       lambda match: match_case(match.group(0), code), 
                       text, flags=re.IGNORECASE)
    return text



def process_txt_files(input_folder, output_folder, replacements):
    # 遍历文件夹下的所有txt文件
    idx = 1
    for file_path in glob.glob(os.path.join(input_folder, '*.txt')):
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # 替换文本
        new_text = replace_terms(text, replacements)
        
        # 创建输出文件路径
        output_file = os.path.join(output_folder, f"{idx}.txt")
        
        # 写入替代后的文本到新文件
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(new_text)
        idx += 1

def replace_in_json(data, replacements):
    if isinstance(data, dict):
        # 如果是字典，递归处理每个键值对
        return {key: replace_in_json(value, replacements) for key, value in data.items()}
    elif isinstance(data, list):
        # 如果是列表，递归处理每个元素
        return [replace_in_json(item, replacements) for item in data]
    elif isinstance(data, str):
        # 如果是字符串，执行替换操作
        return replace_terms(data, replacements)
    else:
        # 其他类型的数据（如数字）直接返回
        return data

def process_json_file(input_file_path, output_file_path, replacements):
    # 读取 JSON 文件
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 替换 JSON 文件中的内容
    new_data = replace_in_json(data, replacements)
    
    # 将修改后的内容写回新的 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #############先跑graphrag###############
    #############再跑test_context_inspect.ipynb 中 entity_embedding_df.to_csv(xxxxx)保存读取的实体为csv##############
    #########利用csv生成密码##########
    replacement_list_sorted = gen_replacements("/data/yuhui/6/adddata/location_data/entity.csv")
    ############替换txt##########
    process_txt_files("/data/yuhui/6/graphrag/alltest/location_dataset/dataset_4_revised/input", #读取包含txt的文件夹
                  "/data/yuhui/6/graphrag/alltest/location_dataset/dataset_4_fake/input", #生成的文件夹
                  replacement_list_sorted)
    ############替换json###########
    process_json_file("location_data/true_revised/question_multi_v3.json",
                      "location_data/true_revised/question_multi_v3_fake.json",
                      replacement_list_sorted)


