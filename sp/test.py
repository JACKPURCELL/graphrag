import json
from collections import defaultdict
from pathlib import Path
import re

def process_paragraph_support_indices(file_path, output_dir):
    # Create a dictionary to store counts of paragraph_support_idx occurrences
    support_idx_count = defaultdict(int)
    support_idx_data = defaultdict(list)  # 用于存储每个 support index 对应的 data
    output_jsons = []
    # Open the JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        # Iterate over each line in the file
        for line in f:
            # Parse the JSON object
            data = json.loads(line.strip())
            question_id = data.get("id")
            if not question_id.startswith("2hop"):
                continue
            # Iterate over each question decomposition
            for question in data.get("question_decomposition", []):
                # Get the paragraph_support_idx
                support_idx = question.get("id")
                
                # Increment the count for this index
                if support_idx is not None:
                    support_idx_count[support_idx] += 1
                    support_idx_data[support_idx].append(data)

    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write each support index's data to a separate JSON file
    for support_idx, questions in support_idx_data.items():
        if support_idx_count[support_idx] < 2:
            print(f"Skipping support index {support_idx} with count {support_idx_count[support_idx]}.")
            continue
        
        # 使用列表推导式过滤掉符合条件的 question
        filtered_questions = []
        for question in questions:
            # 正则表达式
            pattern = r"(\w+)__(\d+)_(\d+)"
            # 使用正则表达式匹配并提取
            match = re.match(pattern, question.get("id", ""))
            if match:
                part1, part2, part3 = match.groups()
                if support_idx == int(part3):
                    support_idx_count[support_idx] -= 1
                    continue  # 跳过当前 question（即删除）
            filtered_questions.append(question)
        if support_idx_count[support_idx] < 2:
            print(f"Skipping support index {support_idx} with count {support_idx_count[support_idx]}.")
            continue 
        question_set = {
            "shared_index": support_idx,
            "questions": filtered_questions
        }
        output_jsons.append(question_set)
    print(f"Total question sets created: {len(output_jsons)}")
    with open(Path(output_dir) / "question_sets.json", 'w', encoding='utf-8') as out_file:
        json.dump(output_jsons, out_file, ensure_ascii=False, indent=4)

# Example usage
file_path = '/home/ljc/data/graphrag/sp/musique_ans_v1.0_dev.jsonl'
output_dir = '/home/ljc/data/graphrag/sp/output_question_sets'
process_paragraph_support_indices(file_path, output_dir)