from build_questions_v3_1023 import generate_questions
from build_corpus_1023_target import process_questions_v2,rewrite_txt_v2,rewrite_txt_v2_only_writeone
from build_answer_v4_1023 import process_corpus_file
import shutil
import os

import subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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



def full_funtion(clean_path):
    new_base_path = clean_path + "_llama"
    try:
        shutil.copytree(clean_path, new_base_path)
        print(f"Copy clean output to {new_base_path}")
        shutil.rmtree(os.path.join(new_base_path, 'input/adv_texts_direct_test0.txt'))
        shutil.rmtree(os.path.join(new_base_path, 'input/adv_texts_enhanced_test0.txt'))
        shutil.rmtree(os.path.join(new_base_path, 'input/adv_texts_indirect_test0.txt'))
        shutil.rmtree(os.path.join(new_base_path, 'output'))
        shutil.rmtree(os.path.join(new_base_path, 'cache'))
        os.remove(os.path.join(new_base_path, 'results_log_t2.txt'))
        os.remove(os.path.join(new_base_path, 'question_with_answer_v4_retest_t2.json'))
        print(f"Remove output and cache folders in {new_base_path}")
    except:
        pass    
    process_questions_v2(clean_path, new_base_path, black_box=False,attack_middlewithleaf=False,llama_model=True,remove_2=False,remove_1=False)

    rewrite_txt_v2_only_writeone(new_base_path,repeat_count=1,num_keep_direct=10,num_keep_indirect=5)

    run_command(new_base_path)
    corpus_file = new_base_path + '/test0_corpus.json'
    process_corpus_file(new_base_path, corpus_file)

# ,'/home/ljc/data/graphrag/alltest/target/dataset4_v3_white_t2_multi_single_keep1_target','/home/ljc/data/graphrag/alltest/target/medi_v2_multi_only1_target'
if __name__ == '__main__':
    clean_paths = ['/home/ljc/data/graphrag/alltest/target/medi_v2_multi_only1_target']
    for clean_path in clean_paths:
        print(f"Processing {clean_path}")
        full_funtion(clean_path)
        print("Done")
        
        # export CUDA_VISIBLE_DEVICES=3 && python /home/ljc/data/graphrag/process_code/pipeline_1101_target.py