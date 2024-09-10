import os
from pathlib import Path  
import networkx as nx

from openai import OpenAI
import json
from tqdm import tqdm
client = OpenAI()


if __name__ == "__main__":
    base_path = '/home/ljc/data/graphrag/alltest/basedataset/test1'
    adv_prompt_path = os.path.join(base_path, 'adv_prompt_advance.json')
    all_jsons = json.loads(Path(adv_prompt_path).read_text(encoding='utf-8'))

    # 收集所有的 adv_text
    indirect_adv_texts = []
    direct_adv_texts = []
    for question in all_jsons:
        for indirect_adv_text in question["indirect_adv_texts"]:
            indirect_adv_texts.append(indirect_adv_text)
        for direct_adv_text in question["direct_adv_texts"]:
            direct_adv_texts.append(direct_adv_text)
    
    output_path_indirect = Path(base_path) / 'adv_texts_indirect.txt'
    output_path_indirect.write_text('\n\n'.join(indirect_adv_texts), encoding='utf-8')
    
    output_path_direct = Path(base_path) / 'adv_texts_direct.txt'
    output_path_direct.write_text('\n\n'.join(direct_adv_texts), encoding='utf-8')