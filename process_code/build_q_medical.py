
import os
import json
import pandas as pd


SYSTEM_PROMPT="""
Please generate questions following the instructions below:

"""
USER_PROMPT = """

---Original Data---

{context_data}

---Goal---

Based on Original Data, the following problem is constructed:
In which part of the body would a disease with ABC symptoms be located?
What drugs should be used to treat a disease with ABC symptoms?

The ABC symptoms selected must be significant and specific to the disease, not common. 3-5 symptoms can be selected.
Questions must follow the path of Symptoms -> Disease -> Drug -> Side Effects. The specific name of the disease must not appear in the question, only the symptom.
Symptoms are "{{root_node}}"
Disease is "{{middle_node}}"
Drugs, body parts are "{{leaf_node}}"
Entities appearing in JSON must be the capitalized form of the entity name wrapped in [] in the text.
The symptoms({{root_node}}) used must be the same in the same generation again.


FOR EACH QUESTION YOU GENERATE, 

1. How you generate this question step by step. Finish the chain of thoughts based on the your knowledge based Entities and Relationships and generation process.
2. During your reasoning process, outline chain of thoughts in the form of a knowledge graph. In the knowledge, the nodes and relationship must be the your knowledge based Entities and Relationships.
3. Each reasoning step MUST correspond to at least one edge that has two nodes and one relationship in the knowledge graph. 
4. Each reasoning step MUST use the words in the corresponding part of the question without paraphrase.
5. Adjust chain of thoughts to achieve this.
6. Also generate the "Template Relationship based on chain_of_thoughts" using the chain of thoughts.  Leaving "{{root_node}}" and "{{middle_node}}" and "{{leaf_node}}" for future placeholders.
7. In "Template Relationship between root and middle node", add  template to connect the "{{root_node}}" and "{{middle_node}}", which should be the same as the first step of the chain of thoughts.
8. In "Template Relationship between middle and leaf node", add  template to connect the "{{middle_node}}" and "{{leaf_node}}", which should be the same as the second step of the chain of thoughts.
9. In "Template Relationship between root and leaf", add  template to connect the "{{root_node}}" and "{{leaf_node}}"


Please must generate {question_count} or  questions and the answers in the following json format, which includes the question, gpt_answer_withoutdata, answer, consistency, [middle_node], leaf nodes, and root nodes. 

Return {question_count}  jsons in a list. Make sure the json is correct.

"""
EXAMPLE_USE ="""

Return in Json list like this:

<JSON example>
[
  {
    "question": "In which part of the body would a disease with symptoms of Asthenia, Ataxia, and Amnesia be located?",
    "gpt_answer_withoutdata": "The disease with symptoms of Asthenia, Ataxia, and Amnesia would be located in the Brain, Central Nervous System, and Nervous System.",
    "answer": "The disease with symptoms of Asthenia, Ataxia, and Amnesia would be located in the Brain, Central Nervous System, and Nervous System.",
    "consistency": true,
    "root_nodes": ["Asthenia", "Ataxia", "Amnesia"],
    "middle_node": "Alcohol Dependence",
    "leaf_nodes": ["Brain", "Central Nervous System", "Nervous System"],
    "chain_of_thoughts": [
      "The disease with symptoms of Asthenia, Ataxia, and Amnesia is Alcohol Dependence.",
      "Alcohol Dependence localizes at the Brain, Central Nervous System, and Nervous System."
    ],
    "Template Relationship based on chain_of_thoughts": [
      "The disease with symptoms of {root_node} is {middle_node}.",
      "{middle_node} localizes at the {leaf_node}."
    ],
    "Template Relationship between root and middle node": [
      "The disease with symptoms of {root_node} is {middle_node}."
    ],
    "Template Relationship between middle and leaf node": [
      "{middle_node} localizes at the {leaf_node}."
    ],
    "Template Relationship between root and leaf node": [
      "The disease with symptoms of {root_node} localizes at the {leaf_node}."
    ],
    "knowledge_graph": [
      ["Asthenia", "Alcohol Dependence", "symptom"],
      ["Ataxia", "Alcohol Dependence", "symptom"],
      ["Amnesia", "Alcohol Dependence", "symptom"],
      ["Alcohol Dependence", "Brain", "localizes at"],
      ["Alcohol Dependence", "Central Nervous System", "localizes at"],
      ["Alcohol Dependence", "Nervous System", "localizes at"]
    ]
  },
  {
    "question": "What drugs should be used to treat a disease with symptoms of Asthenia, Ataxia, and Amnesia?",
    "gpt_answer_withoutdata": "The drugs used to treat a disease with symptoms of Asthenia, Ataxia, and Amnesia are Citalopram, Acamprosate, Naltrexone, and Disulfiram.",
    "answer": "The drugs used to treat a disease with symptoms of Asthenia, Ataxia, and Amnesia are Citalopram, Acamprosate, Naltrexone, and Disulfiram.",
    "consistency": true,
    "root_nodes": ["Asthenia", "Ataxia", "Amnesia"],
    "middle_node": "Alcohol Dependence",
    "leaf_nodes": ["Citalopram", "Acamprosate", "Naltrexone", "Disulfiram"],
    "chain_of_thoughts": [
      "The disease with symptoms of Asthenia, Ataxia, and Amnesia is Alcohol Dependence.",
      "The drugs used to treat Alcohol Dependence are Citalopram, Acamprosate, Naltrexone, and Disulfiram."
    ],
    "Template Relationship based on chain_of_thoughts": [
      "The disease with symptoms of {root_node} is {middle_node}.",
      "The drugs used to treat {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and middle node": [
      "The disease with symptoms of {root_node} is {middle_node}."
    ],
    "Template Relationship between middle and leaf node": [
      "The drugs used to treat {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and leaf node": [
      "The drugs used to treat {root_node} are {leaf_node}."
    ],
    "knowledge_graph": [
      ["Asthenia", "Alcohol Dependence", "symptom"],
      ["Ataxia", "Alcohol Dependence", "symptom"],
      ["Amnesia", "Alcohol Dependence", "symptom"],
      ["Alcohol Dependence", "Citalopram", "treat"],
      ["Alcohol Dependence", "Acamprosate", "treat"],
      ["Alcohol Dependence", "Naltrexone", "treat"],
      ["Alcohol Dependence", "Disulfiram", "treat"]
    ]   
  }
]



"""




import os
import time
from openai import OpenAI
import openai
client = OpenAI()
input_dir = '/home/ljc/data/graphrag/alltest/exp_final/medi_v2/input'

def ask_gpt(system_prompt, user_prompt,temp=0.2):
    try_times = 0
    try:
        try_times += 1
        
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            # response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
        )
        content = completion.choices[0].message.content
        content = content.split('```json\n', 1)[-1].rsplit('\n```', 1)[0]
        content = json.loads(content)
        if content is not None:
            return content
        
    except openai.RateLimitError as e:
        # 从错误信息中提取等待时间
        wait_time = 30  # 默认等待时间
        if 'Please try again in' in str(e):
            try:
                wait_time = float(str(e).split('Please try again in')[1].split('s')[0].strip())
            except ValueError:
                pass
        print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
        time.sleep(wait_time)
        return ask_gpt(system_prompt, user_prompt)  # 递归调用以重试请求
    
    except Exception as e:
        if try_times > 3:
            print("Increase the temperature")
            temp += 0.1
            return ask_gpt(system_prompt, user_prompt,temp)
        print("Error RETRY")
        return ask_gpt(system_prompt, user_prompt)
    


def process_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return_json = ask_gpt(SYSTEM_PROMPT, USER_PROMPT.format(context_data=file_content,question_count=2)+EXAMPLE_USE)
    output_json = {}
    if isinstance(return_json, dict):
        return_json = [return_json]
    output_json["questions"]= return_json
    output_json["middle_node"] = return_json[0]["middle_node"]
    output_json["as_source"] = [[root_node, return_json[0]["middle_node"]] for root_node in return_json[0]["root_nodes"]]
    output_json["as_target"] = [[return_json[0]["middle_node"], leaf_node] for leaf_node in return_json[0]["leaf_nodes"]]

    return output_json

multi_save_list = []
single_save_list = []     
def traverse_and_process(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            output_json = process_file(file_path)
            if len(output_json["questions"]) > 1:
                multi_save_list.append(output_json)
            else:
                single_save_list.append(output_json)
# 调用函数遍历并处理文件
traverse_and_process(input_dir)

# 保存结果
save_path = '/home/ljc/data/graphrag/alltest/exp_final/medi_v2/question_multi_v3.json'
with open(save_path, 'w') as file:
    json.dump(multi_save_list, file, indent=4)

save_path = '/home/ljc/data/graphrag/alltest/exp_final/medi_v2/question_single_v3.json'
with open(save_path, 'w') as file:
    json.dump(single_save_list, file, indent=4)