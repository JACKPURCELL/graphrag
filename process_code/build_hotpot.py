import json
from pathlib import Path  













if __name__ == "__main__":

    path = '/home/ljc/data/graphrag/dataset/hotpot_train_v1.1.json'
    
    with open(path, 'r') as file:
        full_dataset = json.load(file)
    

    save_entries = []
    
    
    for entry in full_dataset:
        if entry["level"] == "medium":
            save_entries.append(entry)
        
        if len(save_entries) == 20:
            break
    short_path = '/home/ljc/data/graphrag/dataset/hotpot_train_short.json'
    with open(short_path, 'w') as file:
        json.dump(save_entries, file, indent=4)

    short_path = '/home/ljc/data/graphrag/dataset/hotpot_train_short.json'
    
    with open(short_path, 'r') as file:
        full_dataset = json.load(file)
    
    all_paragraphs = []
    
    for entry in full_dataset:
        for context in entry["context"]:
            paragraphs = context[1]
            all_paragraphs.append(''.join(paragraphs))
    
    output_path = Path('/home/ljc/data/graphrag/dataset/hotpot10_train.txt')
    output_path.write_text('\n\n\n'.join(all_paragraphs), encoding='utf-8')
    

# if __name__ == "__main__":

#     path = '/home/ljc/data/graphrag/dataset/hotpot_dev_fullwiki_v1.json'
    
#     with open(path, 'r') as file:
#         full_dataset = json.load(file)
    
#     # 只读取前 10 个
#     first_10_entries = full_dataset[:10]
    
#     short_path = '/home/ljc/data/graphrag/dataset/hotpot_dev_fullwiki_v1_short.json'
#     with open(short_path, 'w') as file:
#         json.dump(first_10_entries, file, indent=4)