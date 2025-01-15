import os
import re

base_paths = ['/home/ljc/data/graphrag/alltest/1212_baseline/location_1207_tobeuse_only1_baseline_llama',
             '/home/ljc/data/graphrag/alltest/1212_baseline/cyber_v3_tobeuse_only1_baseline_llama',
             '/home/ljc/data/graphrag/alltest/1212_baseline/medi_v3_1207_tobeuse_only1_baseline_llama','/home/ljc/data/graphrag/alltest/1212_baseline/location_1207_tobeuse_only1_baseline',
             '/home/ljc/data/graphrag/alltest/1212_baseline/cyber_v3_tobeuse_only1_baseline',
             '/home/ljc/data/graphrag/alltest/1212_baseline/medi_v3_1207_tobeuse_only1_baseline']

selects = [1, 2, 3, 4]
for base_path in base_paths:
    success_rates = []
    total_tokenss = []
    for select in selects:
        new_base_path = base_path + "_selects_" + str(select)
        log_path = os.path.join(new_base_path, 'question_base_corpus_1121.log')
        with open(log_path, 'r') as file:
            log_content = file.read()

        # 使用正则表达式提取成功率和总 token 数
        success_rate_match = re.search(r'Success rate: (\d+\.\d+)%', log_content)
        total_tokens_match = re.search(r'Total tokens in corpus: (\d+)', log_content)

        if success_rate_match and total_tokens_match:
            success_rate = float(success_rate_match.group(1))
            total_tokens = int(total_tokens_match.group(1))
            success_rates.append(success_rate)
            total_tokenss.append(total_tokens)
        else:
            print('未找到成功率或总 token 数')
    print(base_path)
    
    print(success_rates)
    print(total_tokenss)

    