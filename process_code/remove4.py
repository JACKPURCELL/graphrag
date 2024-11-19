import re

def remove_third_sentence(paragraph):
    # Split the paragraph into sentences using regex to handle different punctuation marks
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    
    # Check if there are at least three sentences
    if len(sentences) >= 3:
        # Remove the third sentence
        sentences.pop(2)
    else:
        print('Paragraph has less than three sentences:', paragraph)
    # Join the sentences back into a paragraph
    return ' '.join(sentences)

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split the content into paragraphs
    paragraphs = content.split('\n\n')
    
    # Process each paragraph
    modified_paragraphs = [remove_third_sentence(paragraph) for paragraph in paragraphs]
    
    # Join the modified paragraphs back into a single string
    modified_content = '\n\n'.join(modified_paragraphs)
    
    # Write the modified content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(modified_content)

# Example usage
input_file = '/home/ljc/data/graphrag/alltest/exp_final/medi_v2_multi_only1_rm4/input/adv_texts_direct_test0.txt'  # Replace with your input file path
output_file = '/home/ljc/data/graphrag/alltest/exp_final/medi_v2_multi_only1_rm4/input/adv_texts_direct_test0_remove4.txt'  # Replace with your desired output file path
process_file(input_file, output_file)
