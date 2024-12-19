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
file_path = '/home/ljc/data/graphrag/alltest/exp_final_keep/dataset4_v3_1102_blackbox_t1_keep1/input/adv_texts_direct_test0.txt'  # Replace with your file path
token_count = count_tokens_in_file(file_path)
print(f"The number of tokens in the file is: {token_count}")
