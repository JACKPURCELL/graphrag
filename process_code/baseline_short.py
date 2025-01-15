

import json
import os
import shutil
import subprocess
from pathlib import Path
import random
from build_answer_base_subgraph import process_corpus_file

def ensure_minimum_word_count_and_save(direct_adv_texts, new_base_path, file_name, min_word_count=200):
    """
    Ensures each text in direct_adv_texts has at least 200 words, and the combined text
    has at least min_word_count words by repeating the content if necessary, and saves it to the specified file.

    :param direct_adv_texts: List of strings to be combined and saved.
    :param new_base_path: Base path where the file will be saved.
    :param file_name: Name of the file to save the content.
    :param min_word_count: Minimum number of words required in the file.
    """
    # Ensure each text has at least 200 words
    processed_texts = []
    for text in direct_adv_texts:
        if isinstance(text, dict):
            try:
                text = text['text']
            except:
                continue
        try:
            words = text.split()
            while len(words) < min_word_count:
                words += text.split()
            processed_texts.append(' '.join(words))
        except:
            continue

    # Join the texts with two newlines and calculate the word count
    combined_text = '\n\n'.join(processed_texts)
    

    # Write the resulting text to the output file
    output_path_direct = Path(os.path.join(new_base_path, file_name))
    output_path_direct.write_text(combined_text, encoding='utf-8')
    
def run_command(new_path):
    # 获取当前工作目录
    original_dir = new_path
    
    # 目标目录
    target_dir = '/home/ljc/data/graphrag/'
    
    try:
        # 切换到目标目录
        os.chdir(target_dir)
        
        # 构建命令
        command = [
            'python', '-m', 'graphrag.index', '--root', original_dir
        ]
        
        # 执行命令
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 输出结果
        print('Standard Output:', result.stdout)
        print('Standard Error:', result.stderr)
        
        # 检查返回码
        if result.returncode == 0:
            print('命令执行成功')
        else:
            print('命令执行失败')
    
    finally:
        # 切换回原始目录
        os.chdir(original_dir)


base_paths = ['/home/ljc/data/graphrag/alltest/1212_baseline/location_1207_tobeuse_only1_baseline_llama',
             '/home/ljc/data/graphrag/alltest/1212_baseline/cyber_v3_tobeuse_only1_baseline_llama',
             '/home/ljc/data/graphrag/alltest/1212_baseline/medi_v3_1207_tobeuse_only1_baseline_llama','/home/ljc/data/graphrag/alltest/1212_baseline/location_1207_tobeuse_only1_baseline',
             '/home/ljc/data/graphrag/alltest/1212_baseline/cyber_v3_tobeuse_only1_baseline',
             '/home/ljc/data/graphrag/alltest/1212_baseline/medi_v3_1207_tobeuse_only1_baseline']

selects = [1, 2, 3, 4]
for select in selects:
    for base_path in base_paths:
        new_base_path = base_path + "_selects_" + str(select)
        shutil.copytree(base_path, new_base_path)
        try:
            os.remove(os.path.join(new_base_path,  'input/adv_texts_direct_base.txt'))
        except:
            pass
        try:
        
            os.remove(os.path.join(new_base_path, 'question_base_corpus_1121.json'))
        except:
            pass
        try:
        
            os.remove(os.path.join(new_base_path, 'question_base_corpus_1121.log'))
        except:
            pass
        try:
        
            shutil.rmtree(os.path.join(new_base_path, 'output'))
        except:
            pass

        corpus_path = os.path.join(new_base_path, 'question_base_corpus.json')

        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
            
        need_to_write = []

        for item in corpus:
            
            need_to_write.extend(item['direct_adv_texts'][:select])
            

        ensure_minimum_word_count_and_save(need_to_write, new_base_path, 'input/adv_texts_direct_base.txt',min_word_count=1)
        run_command(new_base_path)

        process_corpus_file(new_base_path, corpus_path)   