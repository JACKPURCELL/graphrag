import json

# with open('/home/ljc/data/graphrag/alltest/exp_final/dataset4_v3_white_t2_multi_single_keep1/question_with_answer_v4_retest.json', 'r') as f:
#     single_jsons = json.load(f)
    
    
with open('/home/ljc/data/graphrag/alltest/exp_final/medi_v2_multi_full/question_with_answer_v4_retest.json', 'r') as f:
    full_jsons = json.load(f)
    
neeed_remove_q = []    
for full in full_jsons:
    if full is None:
        continue
    # if single['question'] == full['question']:
    full_bool = full["found_leaf"] or full["found_middle"]
    # print(single_bool,full_bool)
    if not full_bool:
        neeed_remove_q.append(full['question'])
   
print(len(neeed_remove_q))            
with open('/home/ljc/data/graphrag/alltest/exp_final/medi_v2_multi_full/remove_q.json', 'w') as f:
    json.dump(neeed_remove_q,f,ensure_ascii=False,indent=4)