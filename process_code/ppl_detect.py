import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math

def calculate_perplexity(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Get the model's output
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    # Calculate the loss
    loss = outputs.loss.item()
    
    # Calculate perplexity
    perplexity = math.exp(loss)
    return perplexity

def process_text_file(file_path, model, tokenizer):
    perplexities = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                ppl = calculate_perplexity(line, model, tokenizer)
                perplexities.append(ppl)
    return perplexities

def main():
    # Define the directory containing the text files
    input_dir = '/home/ljc/data/graphrag/alltest/exp_final/dataset4_v3_white_t2_multi_single_keep1/input'
    
    # Define the malicious text files
    malicious_files = {
        'adv_texts_direct_test0.txt',
        'adv_texts_enhanced_test0.txt',
        'adv_texts_indirect_test0.txt'
    }
    
    # Load the pre-trained model and tokenizer
    model_name = 'gpt2'  # You can choose a different model if needed
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    
    clean_ppl = []
    malicious_ppl = []
    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if os.path.isfile(file_path):
            # Determine if the file is malicious or clean
            if filename in malicious_files:
                text_type = 'malicious'
            else:
                text_type = 'clean'
            
            # Calculate perplexities for each sentence in the file
            perplexities = process_text_file(file_path, model, tokenizer)
            if text_type == 'clean':
                clean_ppl.extend(perplexities)
            else:
                malicious_ppl.extend(perplexities)


