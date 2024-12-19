# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Local question generation."""

import json
import logging
import random
import time
from typing import Any
from tqdm import tqdm
import tiktoken
from unsloth import FastLanguageModel 

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.question_gen.base import BaseQuestionGen, QuestionResult
# from graphrag.query.question_gen_by_entity.system_prompt import QUESTION_SYSTEM_PROMPT
from openai import OpenAI
import time
client = OpenAI()
print(client.api_key)
log = logging.getLogger(__name__)
import concurrent.futures
import openai 
from tqdm import tqdm



    
def ask_gpt( system_prompt, user_prompt,temp=0.2):
    try_times = 0
    try:
        try_times += 1
        recent_time = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
        )
        content = completion.choices[0].message.content
        print(f"Time for ask_gpt: {time.time()-recent_time}")
        # content = content.split('OUTPUT_START', 1)[-1].rsplit('OUTPUT_END', 1)[0]
        content = json.loads(content)
        
        print(content)
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
        time.sleep(wait_time)
        return ask_gpt(system_prompt, user_prompt)  # 递归调用以重试请求
    
    except Exception as e:
        print(f"Error: {e}")
        if try_times > 3:
            print("Increase the temperature")
            temp += 0.1
            return ask_gpt(system_prompt, user_prompt,temp)
        print("Error RETRY")
        return ask_gpt(system_prompt, user_prompt)


    
USER_PROMPT_MULTI_ROOT = """
Please MUST generate {question_count}  questions 

---Data tables---

{context_data}

---Goal---
First you need to find out what disease this TXT is talking about, let's take the disease as [middle_node] let's take the symptoms of the disease as [root_node], let's take the medication for the disease and the part of the body where the disease is located as [leaf_node].



1. Formulate questions along the path of [root_node] -> [middle_node] -> [leaf_node]. The questions should only include the root nodes without mentioning the [middle_node]. Try to hide the [middle_node] with another representation using the root nodes in the question.

2. The candidate questions should reflect the important or urgent information or themes within the data tables.

3. These questions should be answerable using the provided data tables but should not explicitly reference any specific data fields or tables in the question text.

4. The answers to these questions should be the [leaf_node] itself or some simple content in [leaf_node]' description 

5. At the same time you need to ask the questions to yourself without the Data tables, And then Answer within 50 words, write the answer in the gpt_answer_withoutdata field. then compare the answer with the data table to see if the answer is consistent with the data table. If the answer is not consistent approximately with the data table, you need to reconsider the question.

6. Don't use other entities and relationships(out of the txt) information to generate the questions.

7. Should include the two or more root nodes in the question. Should not include the [middle_node] and [leaf nodes] in the question.

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
10. Must include two or more root nodes in the question. The root node you use in different questions MUST BE THE SAME 
11. when [middle_node] is the source, the [leaf_node] is the target, put the relationship between [middle_node] and [leaf_node] in the as_target. When [middle_node] is the target, the [root_node] is the source, put the relationship between [middle_node] and [leaf_node] in the as_source. Don't lost this part in json

Please must generate {question_count}  questions and the answers in the following json format, which includes the question, gpt_answer_withoutdata, answer, consistency, [middle_node], leaf nodes, and root nodes. 

Return {question_count}  jsons in a list.
Just output json, don't saw any other information.
"""

EXAMPLE_USE_MULTI_ROOT = """


<JSON example>

{
    
"questions": [{
    "question": "What medication should be used to treat a patient who may has combination symptoms of Facial Paralysis, Hypesthesia, Muscular Atrophy and Neuralgia?", 

    "gpt_answer_withoutdata":"...",
    "answer": "...",
    "type": "multi_root_normal",
    "consistency": true,
    "root_nodes": ["Facial Paralysis", "Hypesthesia", "Muscular Atrophy", "Neuralgia"],
    "middle_node": "Leprosy",
    "leaf_nodes": ["Rifampicin", "Dapsone", "Thalidomide"],
    "chain_of_thoughts": [
    "The disease have combination symptom of Facial Paralysis, Hypesthesia, Muscular Atrophy and Neuralgia is Leprosy.",
    "The treatment drugs of Leprosy are Rifampicin, Dapsone, Thalidomide."
    ],
    "Template Relationship based on chain_of_thoughts": [
        "The disease have symptom of {root_node} is {middle_node}.",
        "The treatment drugs of {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and middle node": [
        "The disease have symptom of {root_node} is {middle_node}.",
    ],
    "Template Relationship between middle and leaf node": [
        "The treatment drugs of {middle_node} are {leaf_node}.",
    ],
    "Template Relationship between root and leaf node": [
        "The treatment drugs of {root_node} are {leaf_node}."
    ],
    "knowledge_graph": [
        ["Facial Paralysis", "Leprosy", "disease"],
        ["Leprosy", "Rifampicin", "treatment drugs"],
        ["Leprosy", "Dapsone", "treatment drugs"],
        ["Leprosy", "Thalidomide", "treatment drugs"]
    ]
    },
    {

    "question": ""In a patient presenting with a combination of symptoms such as facial paralysis, hypesthesia, muscular atrophy, and neuralgia, where does this disease typically localize in the human body?", 
    "gpt_answer_withoutdata":"...",
    "answer": "...",
    "type": "multi_root_normal",
    "consistency": true,
    "root_nodes": ["Facial Paralysis", "Hypesthesia", "Muscular Atrophy", "Neuralgia"],
    "middle_node": "Leprosy",
    "leaf_nodes": ["Nose", "Peripheral Nervous System", "Appendage", "Tendon", "Testis", "Eye", "Skin Epidermis", "Nervous System", "Nerve", "Sheath Of Schwann"],
    "chain_of_thoughts": [
    "The disease have combination symptom of Facial Paralysis, Hypesthesia, Muscular Atrophy and Neuralgia is Leprosy.",
    "The disease of Leprosy typically localize in the human body are Nose, Peripheral Nervous System, Appendage, Tendon, Testis, Eye, Skin Epidermis, Nervous System, Nerve, Sheath Of Schwann."
    ],
    "Template Relationship based on chain_of_thoughts": [
        "The disease have symptom of {root_node} is {middle_node}.",
        "The disease of {middle_node} typically localize in the human body are {leaf_node}."          
    ],
    "Template Relationship between root and middle node": [
        "The disease have symptom of {root_node} is {middle_node}.",
    ],
    "Template Relationship between middle and leaf node": [
            "The disease of {middle_node} typically localize in the human body are {leaf_node}.",
    ],
    "Template Relationship between root and leaf node": [
        "The disease have symtoms of {root_node} typically localize in the human body are {leaf_node}."        
    ],
    "knowledge_graph": [
        ["Facial Paralysis", "Leprosy", "disease"],
        ["Leprosy", "Nose", "localize in the human body"],
        ["Leprosy", "Peripheral Nervous System", "localize in the human body"]        
    ]}
    ],
"middle_node": "Leprosy",
"as_source": [["Leprosy", "Rifampicin"],[ "Leprosy", "Dapsone"], ["Leprosy", "Thalidomide"]],
"as_target": [["Nose","Leprosy"],["Peripheral Nervous System","Leprosy"],["Appendage","Leprosy"],["Tendon","Leprosy"],["Testis","Leprosy"],["Eye","Leprosy"],["Skin Epidermis","Leprosy"],["Nervous System","Leprosy"],["Nerve","Leprosy"],["Sheath Of Schwann","Leprosy"]]


}
"""


import os



directory = "/home/ljc/data/graphrag/alltest/new_med_1204/medi_v3/input"



to_save_list = []
for root, dirs, files in os.walk(directory):
    for file in tqdm(files):
        each_txt = os.path.join(root, file)
        system_prompt = USER_PROMPT_MULTI_ROOT.format(question_count=2, context_data=open(each_txt).read())
        user_prompt = "The output json example is as follows \n" + EXAMPLE_USE_MULTI_ROOT
        
        return_json = ask_gpt(system_prompt, user_prompt)
        for as_source in return_json["as_source"]:
            try:
                if as_source[0] != return_json["middle_node"]:
                    
                    print(f"\nerror: {as_source}")
            except:
                continue
        for as_target in return_json["as_target"]:
            try:
                if as_target[1] != return_json["middle_node"]:
                    print(f"\nerro: {as_target}")
            except:
                continue
        to_save_list.append(return_json)
        
with open("/home/ljc/data/graphrag/alltest/new_med_1204/medi_v3/question_multi_v3.json", "w") as f:
    json.dump(to_save_list, f)