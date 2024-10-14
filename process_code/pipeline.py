from build_questions_v3 import generate_questions
from build_corpus_subgraph_fake import process_questions_v2,rewrite_txt_v2
from build_answer_v4_fornewq_subgraphv2 import process_corpus_file
import shutil
import os


base_path = "/home/ljc/data/graphrag/alltest/location_dataset/dataset_4_revised_subgraph_t1_ten_tofake"
# 调用函数并传递 base_path 参数
print("=======Start generate_questions")
# generate_questions(base_path,question_count=5, entity_count=4)

new_path = base_path + '_subgraph_t1'

    
    
print("=======Start build corpus")
# process_questions_v2(base_path,new_path)
# rewrite_txt_v2(new_path)


print("=======Start build NEW RAG")

import subprocess
import os

def run_command():
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

# 运行命令
run_command()




print("=======Start build answer")

corpus_file = new_path + '/test0_corpus.json'
process_corpus_file(new_path, corpus_file)