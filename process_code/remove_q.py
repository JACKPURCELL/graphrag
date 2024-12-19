import json

# with open('/home/ljc/data/graphrag/alltest/exp_final/dataset4_v3_white_t2_multi_single_keep1/question_with_answer_v4_retest.json', 'r') as f:
#     single_jsons = json.load(f)
    
    
with open('/home/ljc/data/graphrag/alltest/new_corpus_1207/cyber_v3_tobeuse_only1_t2/question_with_answer_v4_retest_t2.json', 'r') as f:
    full_jsons = json.load(f)
neeed_remove_q = []    
count = 0
cross= 0
for full in full_jsons:
    if full is None:
        continue
    if not full["found_leaf"]:
        cross += 1
    

    
    # 假设 full["leaf_nodes"] 是一个列表，full["answer_after_attack"] 是一个字符串
        try:
            # 将 full["answer_after_attack"] 转换为小写
            answer_lower = full["answer_after_attack"].lower()
            
            # 检查 full["leaf_nodes"] 中每个元素是否都不在 answer_lower 中
            all_not_in = all(node.lower() not in answer_lower for node in full["leaf_nodes"])
            
            if all_not_in:
                # print("full['leaf_nodes'] 中的每个元素都不在 full['answer_after_attack'] 中")
                neeed_remove_q.append(full['question'])
                count += 1
                
            else:
                print("full['leaf_nodes'] 中至少有一个元素在 full['answer_after_attack'] 中")
        except KeyError as e:
            print(f"缺少键: {e}")
        except AttributeError as e:
            print(f"类型错误: {e}")
     
    
# neeed_remove_q = []    
# for full in full_jsons:
#     if full is None:
#         continue
#     # if single['question'] == full['question']:
#     full_bool = full["found_leaf"] or full["found_middle"]
#     # print(single_bool,full_bool)
#     if not full_bool:
#         neeed_remove_q.append(full['question'])
   
print(len(neeed_remove_q))  
print(count)
print(cross)          
with open('/home/ljc/data/graphrag/alltest/new_corpus_1207/cyber_v3_tobeuse_only1_t2/remove_q.json', 'w') as f:
    json.dump(neeed_remove_q,f,ensure_ascii=False,indent=4)