from .build_questions_v3 import generate_questions
from .build_corpus_v3_1_fornewq import process_questions_v3
from .build_answer_v3_fornewq import process_corpus_file
import shutil
import os


base_path = "/home/ljc/data/graphrag/alltest/location_dataset/dataset4_newq"
# 调用函数并传递 base_path 参数
print("=======Start generate_questions")
generate_questions(base_path,question_count=10, entity_count=20)

new_path = base_path + '_v31'
# 确保目标目录不存在，否则 copytree 会抛出异常
if not os.path.exists(new_path):
    # 复制目录
    shutil.copytree(base_path, new_path)
    print(f"目录已成功复制到 {new_path}")
else:
    print(f"目标目录 {new_path} 已存在")
    
    
print("=======Start build corpus")
process_questions_v3(new_path)



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

base_path = "alltest/location_dataset/dataset4_newq_v2"
corpus_file = new_path + '/question_v3_1_fornewq_corpus.json'
process_corpus_file(base_path, corpus_file)