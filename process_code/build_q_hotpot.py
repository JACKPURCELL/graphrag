import json

with open("/home/ljc/data/graphrag/alltest/new_corpus_1207/hotpot_test_fullwiki_v1.json", "r") as f:
    datas = json.load(f)
    
    
print(datas[0].keys())