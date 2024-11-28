from build_questions_v3_1023 import generate_questions
from build_corpus_1023_target import process_questions_v2,rewrite_txt_v2,rewrite_txt_v2_only_writeone
from build_answer_v4_1023 import process_corpus_file
import shutil
import os

import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

# import time
# time.sleep(28800)
clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1"
new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1_target"
process_questions_v2(clean_path, new_base_path, black_box=False,attack_middlewithleaf=False,llama_model=False,remove_2=False,remove_1=False)
rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)

run_command(new_base_path)
corpus_file = new_base_path + '/test0_corpus.json'
process_corpus_file(new_base_path, corpus_file)

# clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2"
# new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1"
# process_questions_v2(clean_path, new_base_path, black_box=False,attack_middlewithleaf=False,llama_model=False,remove_2=False,remove_1=False)
# rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)

# run_command(new_base_path)
# corpus_file = new_base_path + '/test0_corpus.json'
# process_corpus_file(new_base_path, corpus_file)


# clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2"
# new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1_black"
# process_questions_v2(clean_path, new_base_path, black_box=True,attack_middlewithleaf=False,llama_model=False,remove_2=False,remove_1=False)
# rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
# run_command(new_base_path)
# corpus_file = new_base_path + '/test0_corpus.json'
# process_corpus_file(new_base_path, corpus_file)   
    
# clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2"
# new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1_llama"
# process_questions_v2(clean_path, new_base_path, black_box=False,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)
# rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
# run_command(new_base_path)
# corpus_file = new_base_path + '/test0_corpus.json'
# process_corpus_file(new_base_path, corpus_file)   

# clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2"
# new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1_llama_black"
# process_questions_v2(clean_path, new_base_path, black_box=True,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)
# rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
# run_command(new_base_path)
# corpus_file = new_base_path + '/test0_corpus.json'
# process_corpus_file(new_base_path, corpus_file)   