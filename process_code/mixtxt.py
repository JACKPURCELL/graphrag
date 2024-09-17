import os

# Define the folder path
folder_path = '/home/ljc/data/graphrag/alltest/dataset3/input'

# Initialize an empty list to hold the contents of all .txt files
combined_content = []

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Only process .txt files
        file_path = os.path.join(folder_path, filename)
        # Read the content of each file and append it to the combined_content list
        with open(file_path, 'r') as file:
            combined_content.append(file.read())

# Combine all contents into a single string
combined_text = "\n".join(combined_content)

# Save the combined content into a new file
output_file_path = os.path.join("/home/ljc/data/graphrag/process_code/", 'combined_output.txt')
with open(output_file_path, 'w') as output_file:
    output_file.write(combined_text)


