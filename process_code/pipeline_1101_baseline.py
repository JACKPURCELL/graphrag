
from build_corpus_baseline_subgraph import process_questions_base
from build_answer_base_subgraph import process_corpus_file
import shutil
import os

import subprocess
import os
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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





    

clean_paths = [
    "/home/ljc/data/graphrag/alltest/0411/cyber_v3_tobeuse_ori",
    ]

for clean_path in clean_paths:
    new_base_path = clean_path+"_only1_baseline_c"
    # process_questions_base(clean_path,new_base_path,llama_model=False)
    run_command(new_base_path)
    corpus_file = new_base_path + '/question_base_corpus.json'
    process_corpus_file(new_base_path, corpus_file)         


# clean_path = '/home/ljc/data/graphrag/alltest/new_baseline_limittoken/medi_v3_base1207'
# new_base_path = '/home/ljc/data/graphrag/alltest/new_baseline_limittoken/medi_v3_base1207'
# process_questions_base(clean_path,new_base_path,llama_model=False)
# run_command(new_base_path)
# corpus_file = new_base_path + '/question_base_corpus.json'
# process_corpus_file(new_base_path, corpus_file) 


# clean_path = '/home/ljc/data/graphrag/alltest/new_baseline_limittoken/cyber_dataset_v2_only1_base1207'
# new_base_path = '/home/ljc/data/graphrag/alltest/new_baseline_limittoken/cyber_dataset_v2_only1_base1207'
# process_questions_base(clean_path,new_base_path,llama_model=False)
# run_command(new_base_path)
# corpus_file = new_base_path + '/question_base_corpus.json'
# process_corpus_file(new_base_path, corpus_file) 