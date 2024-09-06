from datasets import load_dataset

# Load the hotpotqa dataset
dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki")

# Access the validation set
dev_set = dataset['validation']

# Open a text file to write the contexts
with open("hotpotqa_contexts.txt", "w", encoding="utf-8") as file:
    # Iterate over each example in the dev set
    for example in dev_set:
        # Extract the sentences from the context
        sentences = example["context"]["sentences"]
        
        # Write each sentence to the file
        for sentence_group in sentences:
            for sentence in sentence_group:
                file.write(sentence + "\n")

print("Contexts have been written to hotpotqa_contexts.txt")
