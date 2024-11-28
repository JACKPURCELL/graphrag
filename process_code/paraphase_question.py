import json
import os
import time
from openai import OpenAI

client = OpenAI()
import openai
import os
from tqdm import tqdm
def ask_gpt(system_prompt, user_prompt,temp=0.1):
    try_times = 0
    try:
        try_times += 1
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
        )
        content = completion.choices[0].message.content
        # content = content.split('```json\n', 1)[-1].rsplit('\n```', 1)[0]
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
        print(f"Error: {e}")
        if try_times > 3:
            print("Increase the temperature")
            temp += 0.1
            return ask_gpt(system_prompt, user_prompt,temp)
        print("Error RETRY")
        return ask_gpt(system_prompt, user_prompt)
    
question_path = '/home/ljc/data/graphrag/alltest/defense_para/cyber_dataset_v2_only1_black/question_multi_v3.json'

with open(question_path, 'r') as f:
    question_sets = json.load(f)
    

for set in tqdm(question_sets):
    for q in set["questions"]:
        question = q["question"]
        system_prompt = "You're a helpful assistant."
        user_prompt = f" This is my question: [{question}]. Please craft 1 paraphrased version for the question. Give your reply as a JSON formatted string. The reply should use “paraphrased_question” as key, paraphase_question as value."
        new_question = ask_gpt("Paraphrase the question", user_prompt.format(question=question),1.0)
        
        
        q["question"] = new_question["paraphrased_question"]
    if len(set["pre_node_pending_questions"]) == 0:
        continue
    for sss in set["pre_node_pending_questions"]:
        for q in sss["questions"]:
            if len(q["question"]) == 0:
                continue
            question = q["question"]
            system_prompt = "You're a helpful assistant."
            user_prompt = f" This is my question: [{question}]. Please craft 1 paraphrased version for the question. Give your reply as a JSON formatted string. The reply should use “paraphrased_question” as key, paraphase_question as value."

            new_question = ask_gpt("Paraphrase the question", user_prompt.format(question=question),1.0)
            q["question"] = new_question["paraphrased_question"]
        
with open(question_path, 'w') as f:
    json.dump(question_sets, f, indent=4)
        
    