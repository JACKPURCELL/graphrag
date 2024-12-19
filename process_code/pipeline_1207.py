from build_questions_v3_1023 import generate_questions
from build_corpus_1207 import process_questions_v2,rewrite_txt_v2_only_writeone
from build_answer_v4_1023 import process_corpus_file
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
    "/home/ljc/data/graphrag/alltest/new_1212/location_1207_tobeuse",
    # "/home/ljc/data/graphrag/alltest/new_1212/medi_v3_1207_tobeuse",
    # "/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse"
    ]

# for clean_path in clean_paths:
#     new_base_path = clean_path+"_only1_llama"
#     process_questions_v2(clean_path, new_base_path, black_box=False,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)
#     rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
#     run_command(new_base_path)
#     corpus_file = new_base_path + '/test0_corpus.json'
#     process_corpus_file(new_base_path, corpus_file)

for clean_path in clean_paths:
    new_base_path = clean_path+"_llama_black_orimodel"
    process_questions_v2(clean_path, new_base_path, black_box=True,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)
    rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=100,num_keep_indirect=50)
    run_command(new_base_path)
    corpus_file = new_base_path + '/test0_corpus.json'
    process_corpus_file(new_base_path, corpus_file)
    
# for clean_path in clean_paths:
#     new_base_path = clean_path+"_only1_black"
#     process_questions_v2(clean_path, new_base_path, black_box=True,attack_middlewithleaf=False,llama_model=False,remove_2=False,remove_1=False)
#     rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
#     run_command(new_base_path)
#     corpus_file = new_base_path + '/test0_corpus.json'
#     process_corpus_file(new_base_path, corpus_file)
    
# export CUDA_VISIBLE_DEVICES=2 && python /home/ljc/data/graphrag/process_code/pipeline_1207.py

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


# directs = [1,3,5]
# for direct in directs:
#     clean_path = "/home/ljc/data/graphrag/alltest/ablation/cyber_dataset_v2_only1"
#     new_base_path = "/home/ljc/data/graphrag/alltest/ablation/cyber_dataset_v2_only1_direct_"+str(direct)
#     try:
#         shutil.copytree(clean_path, new_base_path)
#         print(f"Copy clean output to {new_base_path}")
#         shutil.rmtree(os.path.join(new_base_path, 'output'))
#         shutil.rmtree(os.path.join(new_base_path, 'cache'))
#         os.remove(os.path.join(new_base_path, 'results_log.txt'))
#         os.remove(os.path.join(new_base_path, 'question_with_answer_v4_retest.json'))
#         print(f"Remove output and cache folders in {new_base_path}")
#     except: 
#         pass 

#     rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=direct,num_keep_indirect=5)
#     run_command(new_base_path)
#     corpus_file = new_base_path + '/test0_corpus.json'
#     process_corpus_file(new_base_path, corpus_file)   
    
# enhances = [1,5]
# for enhance in enhances:
#     clean_path = "/home/ljc/data/graphrag/alltest/ablation_temp/dataset4_v3_white_t2_multi_single_keep1"
#     new_base_path = "/home/ljc/data/graphrag/alltest/ablation_temp/dataset4_v3_white_t2_multi_single_keep1_enhance_"+str(enhance)
#     try:
#         shutil.copytree(clean_path, new_base_path)
#         print(f"Copy clean output to {new_base_path}")
#         shutil.rmtree(os.path.join(new_base_path, 'output'))
#         shutil.rmtree(os.path.join(new_base_path, 'cache'))
#         os.remove(os.path.join(new_base_path, 'results_log.txt'))
#         os.remove(os.path.join(new_base_path, 'question_with_answer_v4_retest.json'))
#         print(f"Remove output and cache folders in {new_base_path}")
#     except: 
#         pass 
#     rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=enhance)
#     run_command(new_base_path)
#     corpus_file = new_base_path + '/test0_corpus.json'
#     process_corpus_file(new_base_path, corpus_file)   
    
# repliactions = [3,5,10]
# for repliaction in repliactions:
#     clean_path = "/home/ljc/data/graphrag/alltest/ablation/cyber_dataset_v2_only1"
#     new_base_path = "/home/ljc/data/graphrag/alltest/ablation/cyber_dataset_v2_only1_repliaction_"+str(repliaction)
#     try:
#         shutil.copytree(clean_path, new_base_path)
#         print(f"Copy clean output to {new_base_path}")
#         shutil.rmtree(os.path.join(new_base_path, 'output'))
#         shutil.rmtree(os.path.join(new_base_path, 'cache'))
#         os.remove(os.path.join(new_base_path, 'results_log.txt'))
#         os.remove(os.path.join(new_base_path, 'question_with_answer_v4_retest.json'))
#         print(f"Remove output and cache folders in {new_base_path}")
#     except: 
#         pass 
#     rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=repliaction,num_keep_indirect=5)
#     run_command(new_base_path)
#     corpus_file = new_base_path + '/test0_corpus.json'
#     process_corpus_file(new_base_path, corpus_file)   
    
    
# clean_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2"
# new_base_path = "/home/ljc/data/graphrag/alltest/exp_final/cyber_dataset_v2_only1_llama_black"
# process_questions_v2(clean_path, new_base_path, black_box=True,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)
# rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)
# run_command(new_base_path)
# corpus_file = new_base_path + '/test0_corpus.json'
# process_corpus_file(new_base_path, corpus_file)   