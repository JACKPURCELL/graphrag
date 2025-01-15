import tiktoken

def count_tokens_in_file(file_path, encoding_name='cl100k_base'):
    # Load the tokenizer
    encoding = tiktoken.get_encoding(encoding_name)
    
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Encode the text to tokens
    tokens = encoding.encode(text)
    
    # Return the number of tokens
    return len(tokens)

# Example usage
# file_paths = ['/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_llama_llama_black/input/adv_texts_direct_base.txt','/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_llama_llama_black/input/adv_texts_enhanced_test0.txt','/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_llama_llama_black/input/adv_texts_indirect_test0.txt']  # Replace
# 

# base_paths = [
    
#                     "/home/ljc/data/graphrag/alltest/ablation_new_1212/location_1207_tobeuse_only1_t2_direct_3",
#                 "/home/ljc/data/graphrag/alltest/ablation_new_1212/medi_v3_1207_tobeuse_only1_t2_shuffle_direct_3",
#                 "/home/ljc/data/graphrag/alltest/ablation_new_1212/cyber_v3_tobeuse_only1_t3_shuffle_direct_3",
                
#                 "/home/ljc/data/graphrag/alltest/new_1212/location_1207_tobeuse_only1_black",
#                 "/home/ljc/data/graphrag/alltest/new_1212/medi_v3_1207_tobeuse_only1_black",
#                 "/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_black"
#                 ]
# file_paths=[
#             "/input/adv_texts_direct_test0.txt",
            
#             "/input/adv_texts_enhanced_test0.txt",
#             "/input/adv_texts_indirect_test0.txt",
            
       
#             ]

base_paths = [
        "/home/ljc/data/graphrag/alltest/new_1212/location_1207_tobeuse_only1_baseline",
    "/home/ljc/data/graphrag/alltest/new_1212/medi_v3_1207_tobeuse_only1_baseline",
    "/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_baseline",
    
    "/home/ljc/data/graphrag/alltest/new_1212/location_1207_tobeuse_only1_baseline_llama",
    "/home/ljc/data/graphrag/alltest/new_1212/medi_v3_1207_tobeuse_only1_baseline_llama",
    "/home/ljc/data/graphrag/alltest/new_1212/cyber_v3_tobeuse_only1_baseline_llama"

                ]
file_paths=[
            
            "/input/adv_texts_direct_base.txt"
       
            ]

for base_path in base_paths:
    for file_path in file_paths:
        token_count = count_tokens_in_file(base_path+file_path)
        print(f"{token_count}")
