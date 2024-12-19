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
QUESTION_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant







"""

USER_PROMPT_MULTI_ROOT = """
Please MUST generate {question_count}  questions 

---Data tables---

{context_data}

---Goal---

Let's focus on the [middle_node] {entity}. Consider the following:

Relationships where {entity} is the endpoint, with preceding nodes referred to as root nodes,[root node,middle_node]:
{related_relationships_text_target}

Relationships where {entity} is the starting point, with subsequent nodes referred to as leaf nodes,[middle_node,leaf_node]:
{related_relationships_text_source}


1. Formulate questions along the path of [root_node] -> [middle_node] -> [leaf_node]. The questions should only include the root nodes without mentioning the [middle_node]. Try to hide the [middle_node] with another representation using the root nodes in the question.

2. The candidate questions should reflect the important or urgent information or themes within the data tables.

3. These questions should be answerable using the provided data tables but should not explicitly reference any specific data fields or tables in the question text.

4. The answers to these questions should be the [leaf_node] itself or some simple content in [leaf_node]' description 

5. At the same time you need to ask the questions to yourself without the Data tables, And then Answer within 50 words, write the answer in the gpt_answer_withoutdata field. then compare the answer with the data table to see if the answer is consistent with the data table. If the answer is not consistent approximately with the data table, you need to reconsider the question.

6. Don't use other entities and relationships(out of {entity},{related_relationships_text_source},{related_relationships_text_target}) information to generate the questions.

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
10. Must include two or more root nodes in the question. 

Please must generate {question_count}  questions and the answers in the following json format, which includes the question, gpt_answer_withoutdata, answer, consistency, [middle_node], leaf nodes, and root nodes. 

Return {question_count}  jsons in a list.
Just output json, don't saw any other information.
"""

EXAMPLE_USE_MULTI_ROOT = """


<JSON example>

[{
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
"question": "What side effects of the medication should be used to treat a patient who may has combination symptoms of Facial Paralysis, Hypesthesia, Muscular Atrophy and Neuralgia?", 

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
"The side effects of the treatment drugs of Leprosy are ..."
],
"Template Relationship based on chain_of_thoughts": [
    "The disease have symptom of {root_node} is {middle_node}.",
    "The treatment drugs of {middle_node} are {leaf_node}."
      "The side effects of {leaf_node} are ..."
    
],
"Template Relationship between root and middle node": [
      "The disease have symptom of {root_node} is {middle_node}.",
],
"Template Relationship between middle and leaf node": [
      "The treatment drugs of {middle_node} are {leaf_node}.",
      "The side effects of  {leaf_node} are ..."
],
"Template Relationship between root and leaf node": [
    "The treatment drugs of {root_node} are {leaf_node}."
      "The side effects of  {leaf_node} are ..."
    
],
"knowledge_graph": [
    ["Facial Paralysis", "Leprosy", "disease"],
    ["Leprosy", "Rifampicin", "treatment drugs"],
    ["Leprosy", "Dapsone", "treatment drugs"],
    ["Leprosy", "Thalidomide", "treatment drugs"],
    ["Rifampicin", "...", "side effects"],
    ["Dapsone", "...", "side effects"],
    ["Thalidomide", "...", "side effects"]
    
]

},
{

"question": ""In a patient presenting with a combination of symptoms such as facial paralysis, hypesthesia, muscular atrophy, and neuralgia, where does this disease typically localize in the human body?", 
"gpt_answer_withoutdata":"...",
"answer": "...",
"type": "multi_root_normal",
"consistency": true,
"root_nodes": ["Nose", "Peripheral Nervous System", "Appendage", "Tendon", "Testis", "Eye", "Skin Epidermis", "Nervous System", "Nerve", "Sheath Of Schwann"],
"middle_node": "Leprosy",
"leaf_nodes": ["Rifampicin", "Dapsone", "Thalidomide"],
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
    ["Leprosy", "Peripheral Nervous System", "localize in the human body"],
    
]

}]
"""

USER_PROMPT = """
Please MUST  generate {question_count}  questions 

---Data tables---

{context_data}

---Goal---

Let's focus on the [middle_node] {entity}. Consider the following:

Relationships where {entity} is the endpoint, with preceding nodes referred to as root nodes,[root node,middle_node]:
{related_relationships_text_target}

Relationships where {entity} is the starting point, with subsequent nodes referred to as leaf nodes,[middle_node,leaf_node]:
{related_relationships_text_source}


1. Formulate questions along the path of [root_node] -> [middle_node] -> [leaf_node]. The questions should only include the root nodes without mentioning the [middle_node]. Try to hide the [middle_node] with another representation using the root nodes in the question.

2. The candidate questions should reflect the important or urgent information or themes within the data tables.

3. These questions should be answerable using the provided data tables but should not explicitly reference any specific data fields or tables in the question text.

4. The answers to these questions should be the [leaf_node] itself or some simple content in [leaf_node]' description 

5. At the same time you need to ask the questions to yourself without the Data tables, And then Answer within 50 words, write the answer in the gpt_answer_withoutdata field. then compare the answer with the data table to see if the answer is consistent with the data table. If the answer is not consistent approximately with the data table, you need to reconsider the question.

6. Don't use other entities and relationships(out of {entity},{related_relationships_text_source},{related_relationships_text_target}) information to generate the questions.

7. Should include the root node in the question. Should not include the [middle_node] and [leaf nodes] in the question.

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


Please must generate {question_count}  questions and the answers in the following json format, which includes the question, gpt_answer_withoutdata, answer, consistency, [middle_node], leaf nodes, and root nodes. 
Please MUST  generate {question_count}  questions 

Return {question_count}  jsons in a list. Make sure the json is correct.

"""


EXAMPLE_USE_BACKUP = """
<JSON example>
{
  "question": "What medication should be used to treat a patient who has symptoms of Facial Paralysis?", 
  "gpt_answer_withoutdata": "...",
  "answer": "...",
  "consistency": true,
  "root_nodes": "Facial Paralysis",
  "middle_node": "Leprosy",
  "leaf_nodes": ["Rifampicin", "Dapsone", "Thalidomide"],
  "chain_of_thoughts": [
    "The disease have symptom of Facial Paralysis is Leprosy.",
    "The treatment drugs of Leprosy are Rifampicin, Dapsone, Thalidomide."
  ],
  "Template Relationship based on chain_of_thoughts": [
    "The disease have symptom of {root_node} is {middle_node}.",
    "The treatment drugs of {middle_node} are {leaf_node}."
  ],
  "Template Relationship between root and middle node": [
    "The disease have symptom of {root_node} is {middle_node}."
  ],
  "Template Relationship between middle and leaf node": [
    "The treatment drugs of {middle_node} are {leaf_node}."
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
}

"""
EXAMPLE_USE ="""

Return in Json list like this:

<JSON example>
[
  {
    "question": "What is the patronage of the most famous attractions in the capital of China?", 
    "gpt_answer_withoutdata": "Beijing's top attractions, such as the Forbidden City, the Great Wall, and the Temple of Heaven, draw millions annually. The Forbidden City alone sees over 14 million visitors each year, while sections of the Great Wall near Beijing attract similar numbers, showcasing their global appeal and cultural significance.",
    "answer": "The patronage of the most famous attractions in the capital of China is 100,000.",
    "consistency": true,
    "root_nodes": "China",
    "middle_node": "Beijing",
    "leaf_nodes": ["Great Wall", "Forbidden City", "Summer Palace"],
    "chain_of_thoughts": [
      "The capital of China is Beijing.",
      "Most famous attractions of Beijing is the Forbidden City.",
      "The patronage of the Forbidden City is 100,000."
    ],
    "Template Relationship based on chain_of_thoughts": [
      "The capital of {root_node} is {middle_node}.",
      "Most famous attractions of {middle_node} is the {leaf_node}.",
      "The patronage of the {leaf_node} is {answer}."
    ],
    "Template Relationship between root and middle node": [
      "The capital of {root_node} is {middle_node}."
    ],
    "Template Relationship between middle and leaf node": [
      "Most famous attractions of {middle_node} is the {leaf_node}."
    ],
    "Template Relationship between root and leaf node": [
      "{leaf_node} is located in the capital of {root_node}."
    ],
    "knowledge_graph": [
      ["China", "Beijing", "capital"],
      ["Beijing", "Forbidden City", "Most famous attractions"],
      ["Beijing", "100,000", "patronage"]
    ]
  },
  {
    "question": "What medication should be used to treat a patient who has symptoms of Facial Paralysis?", 
    "gpt_answer_withoutdata": "...",
    "answer": "...",
    "consistency": true,
    "root_nodes": "Facial Paralysis",
    "middle_node": "Leprosy",
    "leaf_nodes": ["Rifampicin", "Dapsone", "Thalidomide"],
    "chain_of_thoughts": [
      "The disease have symptom of Facial Paralysis is Leprosy.",
      "The treatment drugs of Leprosy are Rifampicin, Dapsone, Thalidomide."
    ],
    "Template Relationship based on chain_of_thoughts": [
      "The disease have symptom of {root_node} is {middle_node}.",
      "The treatment drugs of {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and middle node": [
      "The disease have symptom of {root_node} is {middle_node}."
    ],
    "Template Relationship between middle and leaf node": [
      "The treatment drugs of {middle_node} are {leaf_node}."
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
    "question": "How to mitigate the attack of the software Carbanak?", 
    "gpt_answer_withoutdata": "...",
    "answer": "...",
    "consistency": true,
    "root_nodes": "Carbanak",
    "middle_node": "OS Credential Dumping",
    "leaf_nodes": ["Encrypt Sensitive Information", "Behavior Prevention on Endpoint", "Password Policies"],
    "chain_of_thoughts": [
      "The attack technique of Carbanak is OS Credential Dumping.",
      "The mitigation technique of OS Credential Dumping are Encrypt Sensitive Information, Behavior Prevention on Endpoint, Password Policies."
    ],
    "Template Relationship based on chain_of_thoughts": [
      "The attack technique of {root_node} is {middle_node}.",
      "The mitigation technique of {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and middle node": [
      "The attack technique of {root_node} is {middle_node}."
    ],
    "Template Relationship between middle and leaf node": [
      "The mitigation technique of {middle_node} are {leaf_node}."
    ],
    "Template Relationship between root and leaf node": [
      "The mitigation technique of {root_node} are {leaf_node}."
    ],
    "knowledge_graph": [
      ["Carbanak", "OS Credential Dumping", "attack technique"],
      ["OS Credential Dumping", "Encrypt Sensitive Information", "mitigation technique"],
      ["OS Credential Dumping", "Behavior Prevention on Endpoint", "mitigation technique"],
      ["OS Credential Dumping", "Password Policies", "mitigation technique"]
    ]
  }
]

"""

CHANGE_RELATIONS_ORDER = """
Prompt:

You are given a list of pairs in the format [A, B], which will always include the [SAME ENTITY]. Your task is to reorganize these pairs such that the more general or abstract concept comes first, followed by the more specific concept. Here are the rules to follow:

Swap the order of [A, B] to [B, A] if A is more specific and B is more general. B may be larger than A in terms of scope or applicability.
Keep the original order [A, B] if both elements are of the same level of specificity or already correctly ordered.
For notable entities or specific items within a broader category, ensure the broader category comes first, swapping if necessary.
Examples:

[BEIJING, CHINA] should become [CHINA, BEIJING].
[SHANGHAI, CHINA] should become [CHINA, SHANGHAI].
[CHINA, EAST ASIA] should become [EAST ASIA, CHINA].
[BEIJING, BEIJING UNIVERSITY] should remain [BEIJING, BEIJING UNIVERSITY].
[TIANANMEN, BEIJING] should become [BEIJING, TIANANMEN].
[iPhone, Apple] should become [Apple, iPhone].
[Diseases, Symptoms] should become [Symptoms, Diseases].
[Body Location, Diseases] should become [Diseases, Body Location].
[Treatment Drugs, Diseases] should become [Diseases, Treatment Drugs].
[Treatment, Diseases] should become [Diseases, Treatment].
[attack technique, Software] should become [Software, attack technique].
[mitigation technique, attack technique] should become [attack technique, mitigation technique].

[math, mathematics department] should remain [math, mathematics department].
Apply these rules consistently to transform the list of pairs, ensuring that the more general concept (B) precedes the specific concept (A) unless both are of equal specificity or already correctly ordered.

Please return in the following JSON format:

if [SAME ENTITY] is the source(first of the list), put it in as_source, if [SAME ENTITY] is the target(second of the list), put it in as_target.  Make sure the order of each relationship is the correctly ordered one instead of the originial one.

<JSON example>
{
    "as_source": [["SAME ENTITY", "B"], ["SAME ENTITY", "C"], ["SAME ENTITY", "D"]],
    "as_target": [["E", "SAME ENTITY"], ["F", "SAME ENTITY"], ["G", "SAME ENTITY"]]
}

"""
class LocalQuestionGen_byentity_oneedge(BaseQuestionGen):
    """Search orchestration for global search mode."""

    def __init__(
        self,
        llm: BaseLLM,

        context_builder: LocalContextBuilder,
        entities=[],
        relationships=[],
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str = QUESTION_SYSTEM_PROMPT,
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
        llama_model: bool = False,
        pre_root_question_gen:bool  = False,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params,
        )
        self.entities=entities
        self.relationships=relationships
        self.system_prompt = system_prompt
        self.callbacks = callbacks
        self.entity_dict = {enti.title: enti for enti in entities}
        self.pre_root_question_gen = pre_root_question_gen
        self.llama_model = llama_model
        if llama_model:
            print("Load model from local")
            
            # Load model directly
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from transformers import pipeline
            
            #"meta-llama/Llama-3.1-70B-Instruct"
            # tokenizer = AutoTokenizer.from_pretrained(llama_model)
            # tokenizer.pad_token = tokenizer.eos_token
            # model = AutoModelForCausalLM.from_pretrained(llama_model)
            model,tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True,
            )
            tokenizer.pad_token = tokenizer.eos_token
            FastLanguageModel.for_inference(model)
            # Use a pipeline as a high-level helper


            self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,device_map="auto")




    def find_entity_by_title(self, root_node):
        return self.entity_dict.get(root_node)
    
    def ask_llama(self, system_prompt, user_prompt,temp=0.2):
        try_times = 0
        try:
            try_times += 1
            messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
           
            
            content = self.pipe(messages, max_length=10000, do_sample=True, temperature=temp)
            
            content = content[0]["generated_text"][-1]["content"]
            content_json = content.split('OUTPUT_START', 1)[-1].rsplit('OUTPUT_END', 1)[0]
            content_json = json.loads(content_json)
            if content_json is not None:
                return content_json
            else:
                return self.ask_llama(system_prompt, user_prompt,temp)
            
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
            return self.ask_llama(system_prompt, user_prompt)  # 递归调用以重试请求
        
        except Exception as e:
            print(f"Error: {e}")
            if try_times > 3:
                print("Increase the temperature")
                temp += 0.1
                return self.ask_llama(system_prompt, user_prompt,temp)
            print("Error RETRY")
            return self.ask_llama(system_prompt, user_prompt)
        
        
    def ask_gpt(self, system_prompt, user_prompt,temp=0.2):
        try_times = 0
        try:
            try_times += 1
            recent_time = time.time()
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
            return self.ask_gpt(system_prompt, user_prompt)  # 递归调用以重试请求
        
        except Exception as e:
            print(f"Error: {e}")
            if try_times > 3:
                print("Increase the temperature")
                temp += 0.1
                return self.ask_gpt(system_prompt, user_prompt,temp)
            print("Error RETRY")
            return self.ask_gpt(system_prompt, user_prompt)
    
    def ask_llm(self, system_prompt, user_prompt,temp=0.2):
        if self.llama_model:
            temp = 0.1
            return self.ask_llama(system_prompt, user_prompt,temp)
        else:
            return self.ask_gpt(system_prompt, user_prompt,temp)
    def process_target_base(self, as_target, ent_with_rel_name, related_relationships_text_source, context_data,  question_count, as_source_list,  **kwargs):
        per_text_target = "[Root Entity,middle_node]: " + str(as_target)
        
        question_history = [f"Find all the related text units for {ent_with_rel_name}. and the text units of entities in relationships of {related_relationships_text_source} and {per_text_target}, and the relationships of {related_relationships_text_source} and {per_text_target}. IMPORTANT: Do not lost entity in relationship {per_text_target}, and all the information about {ent_with_rel_name}"]
        
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)

        if context_data is None:
            recent_time = time.time()
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )
            print(f"\nTime for context builder: {time.time()-recent_time}")
        else:
            context_records = {"context_data": context_data}
        # print(f"Context_data: {context_data}")
        try:
            system_prompt = self.system_prompt.format(
                question_count=question_count
            )
            user_prompt = USER_PROMPT.format(context_data=context_data,
                                            question_count=question_count,
                                            entity=ent_with_rel_name,
                                            related_relationships_text_source=related_relationships_text_source,
                                            related_relationships_text_target=per_text_target) + EXAMPLE_USE

            # completion = client.chat.completions.create(
            #     model="gpt-4o-2024-08-06",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt},
            #     ],
            #     temperature=0.2,
            # )

            # content = completion.choices[0].message.content
            # content = content.split('```json\n', 1)[-1].rsplit('\n```', 1)[0]
            recent_time = time.time()
            content_json = self.ask_llm(system_prompt, user_prompt)
            pending_questions = {}
            if isinstance(content_json, dict):
                content_json = [content_json]
            if len(content_json) > 0:
                pending_questions["questions"] = content_json
            else:
                return None
            print(f"Time for ask_llm: {time.time()-recent_time}")

            pending_questions["middle_node"] = ent_with_rel_name
            pending_questions["as_source"] = as_source_list
            pending_questions["as_target"] = [as_target]



        except Exception:
            log.exception("Exception in generating question")
            return None
        return pending_questions

    def process_target_base_two_root(self, as_target_list, ent_with_rel_name, related_relationships_text_source, context_data,
                            question_count, as_source_list, **kwargs):
        per_text_target = "[Root Entity,middle_node]: " + str(as_target_list)

        question_history = [
            f"Find all the related text units for {ent_with_rel_name}. and the text units of entities in relationships of {related_relationships_text_source} and {per_text_target}, and the relationships of {related_relationships_text_source} and {per_text_target}. IMPORTANT: Do not lost entity in relationship {per_text_target}, and all the information about {ent_with_rel_name}"]

        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)

        if context_data is None:
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )
        else:
            context_records = {"context_data": context_data}
        try:
            system_prompt = self.system_prompt.format(
                question_count=question_count
            )
            user_prompt = USER_PROMPT_MULTI_ROOT.format(context_data=context_data,
                                             question_count=question_count,
                                             entity=ent_with_rel_name,
                                             related_relationships_text_source=related_relationships_text_source,
                                             related_relationships_text_target=per_text_target) + EXAMPLE_USE_MULTI_ROOT

            # completion = client.chat.completions.create(
            #     model="gpt-4o-2024-08-06",
            #     messages=[
            #         {"role": "system", "content": system_prompt},
            #         {"role": "user", "content": user_prompt},
            #     ],
            #     temperature=0.2,
            # )

            # content = completion.choices[0].message.content
            # content = content.split('```json\n', 1)[-1].rsplit('\n```', 1)[0]
            # content_json = json.loads(content)
            content_json = self.ask_llm(system_prompt, user_prompt)
            pending_questions = {}
            if len(content_json) > 0:
                pending_questions["questions"] = content_json
            else:
                return None

            pending_questions["middle_node"] = ent_with_rel_name
            pending_questions["as_source"] = as_source_list
            pending_questions["as_target"] = as_target_list



        except Exception:
            log.exception("Exception in generating question")
            return None
        return pending_questions

    def change_relations_order(self, as_relationships_list, ent_with_rel_name,  **kwargs):
        user_prompt = "The following relationships are given: " + str(as_relationships_list) + f" The [SAME ENTITY]  is {ent_with_rel_name}"
        return_json = self.ask_llm(CHANGE_RELATIONS_ORDER, user_prompt, temp=0.1)
        
            
        as_source_list = return_json["as_source"]
        as_target_list = return_json["as_target"]
        
        for as_source in as_source_list:
            try:
                if as_source[0] != ent_with_rel_name:
                    as_source_list.remove(as_source)
                    as_target_list.append(as_source)
                    print(f"\nRemove {as_source} from as_source_list")
            except:
                continue
        for as_target in as_target_list:
            try:
                if as_target[1] != ent_with_rel_name:
                    as_target_list.remove(as_target)
                    as_source_list.append(as_target)
                    print(f"\nRemove {as_target} from as_target_list")
            except:
                continue
            
        return as_source_list, as_target_list

    def process_target(self, as_target, ent_with_rel_name, related_relationships_text_source, context_data, question_count, as_source_list, single_questions, multi_questions, **kwarg):
        print(f"\nProcessing {as_target}")
        recent_time = time.time()
        if self.pre_root_question_gen:
            pre_node_pending_questions = self.pre_root_question(as_target, question_count,  **kwarg)
        else:
            pre_node_pending_questions = []
        if isinstance(pre_node_pending_questions, dict):
            pre_node_pending_questions = [pre_node_pending_questions]
        pending_questions = self.process_target_base(as_target, ent_with_rel_name, related_relationships_text_source, context_data, question_count, as_source_list, **kwarg)
        if pending_questions is not None:
            pending_questions["pre_node_pending_questions"] = pre_node_pending_questions
            if len(pending_questions["questions"]) == 1:
                single_questions.append(pending_questions)
                print("****single_questions len ", len(single_questions))
            else:
                multi_questions.append(pending_questions)
                print("*****multi_questions len ", len(multi_questions))
        print(f"\nprocess_target time: {time.time() - recent_time}")
                
    def process_target_two_root(self, as_target_list, ent_with_rel_name, related_relationships_text_source, context_data, question_count, as_source_list, single_questions, multi_questions, **kwarg):
        pending_questions = self.process_target_base_two_root(as_target_list, ent_with_rel_name, related_relationships_text_source, context_data, question_count, as_source_list, **kwarg)
        if pending_questions is not None:
            if len(pending_questions["questions"]) == 1:
                single_questions.append(pending_questions)
            else:
                multi_questions.append(pending_questions)

        
    def pre_root_question(self,middle_as_target,question_count,**kwargs):
        pre_node_pending_questions = []
        context_data = None
        #middle_as_target_list [Root Node,middle_node]
        root_node = middle_as_target[0].upper()
        ent_with_rel_name = root_node
        middle_node = middle_as_target[1].upper()
        all_relationships = [rel for rel in self.relationships if rel.source == root_node or rel.target == root_node]
    
        as_relationships_list = []
        for rel in all_relationships:
            as_relationships_list.append([rel.source, rel.target])
            
        

        as_source_list,as_target_list = self.change_relations_order(as_relationships_list, ent_with_rel_name,  **kwargs)        
        as_source_list = [middle_as_target]
        if len(as_target_list) > 0:
            for as_target in tqdm(as_target_list, desc="Processing prenode targets", leave=False):
                # pre_root_node = as_target[0].upper()
                related_relationships_text_source = "[middle_node,Leaf Entity]: " + str([root_node, middle_node])
                pre_node_pending_questions.append(self.process_target_base(as_target, ent_with_rel_name, related_relationships_text_source, context_data,  question_count, as_source_list, **kwargs))
        else:
            print(f"\nNo root node as target for [Root] root_node,  [Root,Middle] {root_node} -> {middle_node}")
        return pre_node_pending_questions
        
        
  

    def process_entity(self, ent_with_rel, context_data,  question_count, single_questions, multi_questions, multi_root_node=False,**kwargs):
        ent_with_rel_name = ent_with_rel["entity"].title
        print(f"\nProcessing {ent_with_rel_name}")
        recent_time = time.time()
        as_relationships_list = []
        for rel in ent_with_rel["all_relationships"]:
            as_relationships_list.append([rel.source, rel.target])
        

            
        as_source_list,as_target_list = self.change_relations_order(as_relationships_list, ent_with_rel_name,  **kwargs)
        if len(as_target_list) == 0 or len(as_source_list) == 0:
            print(f"\nNo as_target_list or as_source_list for {ent_with_rel_name}")
            return        

        print(f"\n ***change_relations_order time: {time.time() - recent_time}")
        # self.pre_root_question(as_target_list,question_count,pre_root_single_questions,pre_root_multi_questions, **kwargs)
        
        related_relationships_text_source = "[middle_node,Leaf Entity]: " +str(as_source_list)
        if multi_root_node and len(as_target_list) > 1:
            self.process_target_two_root(as_target_list, ent_with_rel_name, related_relationships_text_source, context_data,  question_count, as_source_list, single_questions,multi_questions, **kwargs)
        else:
            for as_target in tqdm(as_target_list, desc="Processing targets", leave=False):
                self.process_target(as_target, ent_with_rel_name, related_relationships_text_source, context_data,  question_count, as_source_list, single_questions, multi_questions, **kwargs)
            # max_threads = 1  # 设置线程数量

            # with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            #     futures = [
            #         executor.submit(self.process_target, as_target, ent_with_rel_name, related_relationships_text_source, context_data,  question_count, as_source_list, single_questions, multi_questions, **kwargs)
            #         for as_target in as_target_list
            #     ]
            #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing targets", leave=False):
            #         future.result()
              

    async def agenerate(
        self,
        question_history: list[str],
        context_data: str | None,
        question_count: int,
        entity_count: int = -1,
        need_to_keep_entity_names: list[str] = [],
        multi_root_node: bool = False,
        **kwargs,
    ) -> tuple[list, list]:
        """
        Generate a question based on the question history and context data.

        If context data is not provided, it will be generated by the local context builder
        """
        start_time = time.time()
        multi_questions = []
        single_questions = []   
        useful_entities = []
       
        for ent in self.entities:
            all_relationships = [rel for rel in self.relationships if rel.source == ent.title or rel.target == ent.title]
            if len(all_relationships) > 1 :
                useful_entities.append({"entity": ent, "all_relationships": all_relationships})
        print("=======Qualified entities: ", len(useful_entities))
        
        if len(need_to_keep_entity_names) != 0:
            print(f"keep entities: {need_to_keep_entity_names}")
            keep_entities = []
            for ent_with_rel in useful_entities:
                if ent_with_rel["entity"].title.lower() in need_to_keep_entity_names:
                    keep_entities.append(ent_with_rel)
            useful_entities = keep_entities
        elif entity_count != -1:
            print(f"random sample {entity_count} entities")
            useful_entities = random.sample(useful_entities, entity_count)
        else:
            print("keep all entities")

        for ent_with_rel in tqdm(useful_entities,desc="Processing entities"):
            self.process_entity(ent_with_rel, context_data,  question_count, single_questions, multi_questions, multi_root_node,**kwargs)
        # max_threads = 1  # 设置线程数量

        # with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        #     futures = [
        #         executor.submit(self.process_entity, ent_with_rel, context_data,  question_count, single_questions, multi_questions, multi_root_node,**kwargs)
        #         for ent_with_rel in useful_entities
        #     ]
        #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing entities"):
        #         future.result()

        return single_questions, multi_questions
        

    def generate(
        self,
        question_history: list[str],
        context_data: str | None,
        question_count: int,
        **kwargs,
    ) -> QuestionResult:
        """
        Generate a question based on the question history and context data.

        If context data is not provided, it will be generated by the local context builder
        """
        start_time = time.time()
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            # construct current query and conversation history
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)

        if context_data is None:
            # generate context data based on the question history
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )  # type: ignore
        else:
            context_records = {"context_data": context_data}
        log.info(
            "GENERATE QUESTION: %s. QUESTION HISTORY: %s", start_time, question_text
        )
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data, question_count=question_count
            )
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]

            response = self.llm.generate(
                messages=question_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return QuestionResult(
                response=response.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records,
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in generating questions")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )


if __name__ == "__main__":
    import os

    import pandas as pd
    import tiktoken

    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates,
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.input.loaders.dfs import (
        store_entity_semantic_embeddings,
    )
    from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    from graphrag.query.llm.oai.embedding import OpenAIEmbedding
    from graphrag.query.llm.oai.typing import OpenaiApiType
    from graphrag.query.question_gen.local_gen import LocalQuestionGen
    from graphrag.query.structured_search.local_search.mixed_context import (
        LocalSearchMixedContext,
    )
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    base_path = "/home/ljc/data/graphrag/alltest/dataset3_poison_met"
    INPUT_DIR = base_path + "/output/20240914-133546/artifacts"
    LANCEDB_URI = f"{INPUT_DIR}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    print(f"Entity count: {len(entity_df)}")
    entity_df.head()

    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    print(f"Relationship count: {len(relationship_df)}")
    relationship_df.head()


    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    print(f"Report records: {len(report_df)}")
    # print(reports)

    report_df.head()

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    print(f"Text unit records: {len(text_unit_df)}")
    text_unit_df.head()


    api_key = os.environ["OPENAI_API_KEY"]
    print(api_key)
    llm_model = 'gpt-4o-2024-08-06'
    embedding_model = 'text-embedding-3-small'

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        # if you did not run covariates during indexing, set this to None
        # covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    # text_unit_prop: proportion of context window dedicated to related text units
    # community_prop: proportion of context window dedicated to community reports.
    # The remaining proportion is dedicated to entities and relationships. Sum of text_unit_prop and community_prop should be <= 1
    # conversation_history_max_turns: maximum number of turns to include in the conversation history.
    # conversation_history_user_turns_only: if True, only include user queries in the conversation history.
    # top_k_mapped_entities: number of related entities to retrieve from the entity description embedding store.
    # top_k_relationships: control the number of out-of-network relationships to pull into the context window.
    # include_entity_rank: if True, include the entity rank in the entity table in the context window. Default entity rank = node degree.
    # include_relationship_weight: if True, include the relationship weight in the context window.
    # include_community_rank: if True, include the community rank in the context window.
    # return_candidate_context: if True, return a set of dataframes containing all candidate entity/relationship/covariate records that
    # could be relevant. Note that not all of these records will be included in the context window. The "in_context" column in these
    # dataframes indicates whether the record is included in the context window.
    # max_tokens: maximum number of tokens to use for the context window.


    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    question_generator = LocalQuestionGen_byentity(
        llm=llm,
        entities=entities,relationships=relationships,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )


    # question_history = [
    #     "What is the patronage of the most famous attractions in the capital of China?",
    #     "What is the patronage of the most famous attractions in the culture center city of China?",
    # ]

    question_path = os.path.join(base_path, 'question_v2.json')

    async def main():
        candidate_questions = await question_generator.agenerate(
        question_history=[], context_data=None, question_count=5)
        # print(candidate_questions)
        with open(question_path, 'w') as f:
            json.dump(candidate_questions, f, indent=4)
        
    # 运行异步函数
    import asyncio
    asyncio.run(main())