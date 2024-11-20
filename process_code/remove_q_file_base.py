import json


import os
with open('/home/ljc/data/graphrag/alltest/exp_final/dataset4_v3_white_t2_multi/remove_q.json', 'r') as f:
    single_jsons = json.load(f)
    
remove_dict = {}    
for single in single_jsons:
    remove_dict[single] = 1
    
remove_path = '/home/ljc/data/graphrag/alltest/exp_final/dataset4_v3_baseline'

question_multi_v3_path = os.path.join(remove_path,'question_multi_v3.json')
question_with_answer_v4_retest_path = os.path.join(remove_path,'question_with_answer_base.json')
test0_corpus_path = os.path.join(remove_path,'question_base_corpus.json')

with open(question_multi_v3_path, 'r') as f:
    question_multi_v3 = json.load(f)
for set in question_multi_v3:
    new_questions = []
    for q in set["questions"]:
        if q["question"] in remove_dict:
            continue
        new_questions.append(q)
    for q in set["questions"]:
        if q["question"] in remove_dict:
            print(q)
    set["questions"] = new_questions
    
    if set["pre_node_pending_questions"] != []:
        alll_ste = []
        for question_set in set["pre_node_pending_questions"]:
            new_pre_node_pending_questions = []
            for question in question_set["questions"]:
                if question["question"] not in remove_dict:
                    new_pre_node_pending_questions.append(question)
         
            question_set["questions"] = new_pre_node_pending_questions
            if len(question_set["questions"]) > 0:
                alll_ste.append(question_set)
        if len(alll_ste) == 0:
            set["pre_node_pending_questions"] = []
                
with open(question_multi_v3_path, 'w') as f:
    json.dump(question_multi_v3,f,ensure_ascii=False,indent=4)
    
with open(question_with_answer_v4_retest_path, 'r') as f:
    question_with_answer_v4_retest = json.load(f)
    
new_questions = []

total_succ_both = 0
total_succ_leaf_only = 0
total_succ_middle_only = 0
total_fail = 0
total_normal = 0

total_succ_pre_node = 0
total_pre_node = 0


        
for q in question_with_answer_v4_retest:
    if q["question"] not in remove_dict:
        new_questions.append(q)

        succ = q["found"]
        total_normal += 1
        if succ:
            total_succ_both += 1
        else:
            total_fail += 1
            
      


print(f"Total successful both: {total_succ_both}/{total_normal}")


print(f"FAILED: {total_fail}/{total_normal}")


log_file_path = os.path.join(remove_path, 'results_log_removeq.txt')
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    log_file.write(f"Total successful both: {total_succ_both}/{total_normal}\n")

    log_file.write(f"FAILED: {total_fail}/{total_normal}\n")

            
with open(question_with_answer_v4_retest_path, 'w') as f:
    json.dump(new_questions,f,ensure_ascii=False,indent=4)

with open(test0_corpus_path, 'r') as f:
    test0_corpus = json.load(f)
    
new_test0_corpus = []
for q in test0_corpus:
    if q is None:
        continue
    if q["question"] not in remove_dict:
        new_test0_corpus.append(q)
        
with open(test0_corpus_path, 'w') as f:
    json.dump(new_test0_corpus,f,ensure_ascii=False,indent=4)