from build_questions_v3_1023 import generate_questions
from build_corpus_1207 import process_questions_v2,rewrite_txt_v2_only_writeone,rewrite_txt_v2
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
    "/home/ljc/data/graphrag/alltest/1212_rm/location_1207_tobeuse_only1_t2_shuffle",
    "/home/ljc/data/graphrag/alltest/1212_rm/medi_v3_1207_tobeuse_only1_t2_shuffle"
    ]



for clean_path in clean_paths:
    new_base_path = clean_path+"_multi_middle"
    try:
        shutil.copytree(clean_path, new_base_path)
        print(f"Copy clean output to {new_base_path}")
        shutil.rmtree(os.path.join(new_base_path, 'output'))
        shutil.rmtree(os.path.join(new_base_path, 'cache'))
        os.remove(os.path.join(new_base_path, 'results_log_t2.txt'))
        os.remove(os.path.join(new_base_path, 'question_with_answer_v4_retest_t2.json'))
        print(f"Remove output and cache folders in {new_base_path}")
    except: 
        pass 

    rewrite_txt_v2(new_base_path,repeat_count=1,shuffle=True)
    run_command(new_base_path)
    corpus_file = new_base_path + '/test0_corpus.json'
    process_corpus_file(new_base_path, corpus_file)   
    
# enhances = [0,1,3]
# for enhance in enhances:
    
#     new_base_path = "/home/ljc/data/graphrag/alltest/ablation_new_1212/location_1207_tobeuse_only1_t2_enhance_"+str(enhance)
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

#     new_base_path = "/home/ljc/data/graphrag/alltest/ablation_new_1212/location_1207_tobeuse_only1_t2_repli_"+str(repliaction)
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