import os
from pathlib import Path  
import networkx as nx

from openai import OpenAI
import json
from tqdm import tqdm
client = OpenAI()



base_prompt = """
Prompt:

Using the graph provided, which consists of nodes and edges representing relationships derived from the original text, generate open-ended Multi-Source Reasoning questions. Each question should explore connections across two or three related nodes in the graph to ensure meaningful context and complexity. The questions must be based on the graph and the original text.

Requirements:

Each question should require an open-ended answer, focusing on specific details from the graph.
The relationships involved in the questions should utilize 2 or 3 nodes to ensure meaningful depth.
Avoid questions that directly compare or mention the differences between two entities.
The questions should be phrased naturally, avoiding technical terms like "entity" or "entities."
Each question should have 15 semantically similar questions, which rephrase the question without changing its meaning.
Each question should also have 15 related questions, which explore the same or adjacent relationships in the graph.
Answers must be directly derivable from the graph.
For each question, provide an explanation describing the specific relationships and nodes involved, ensuring that the logic behind the question is based on multiple connections in the graph.
Explanation Example: For a question like "What attractions can be found in the capital of China?", the explanation might be:

"This question explores the relationship 'China-Beijing' and lists key attractions connected to Beijing as sub-nodes, such as 'Forbidden City,' 'Temple of Heaven,' and 'Summer Palace.' It incorporates a clear hierarchy of relationships to derive a meaningful answer."
Example Structure:

<JSON OBJECT>
[
  {
    "question": "What attractions can be found in the capital of China?",
    "answer": "Forbidden City, Temple of Heaven, Summer Palace, Ming Tombs, Great Wall of China",
    "semantically_similar_questions": [
      "Which famous landmarks are located in the capital of China?",
      "What are the major tourist attractions in Beijing?",
      "What sites should one visit in Beijing, China?",
      "Can you list the top cultural landmarks in China's capital city?",
      "What are some must-see historical sites in Beijing?",
      "Which famous attractions can be found in Beijing?",
      "What are the top destinations for visitors in Beijing?",
      "What historical landmarks are in the capital of China?",
      "What well-known tourist sites are in the capital city of China?",
      "What popular attractions are in Beijing?",
      "What cultural landmarks are in the capital of China?",
      "Which historical landmarks are located in Beijing?",
      "Which places are the most popular tourist attractions in Beijing?",
      "What are the key places of interest in Beijing?",
      "Can you name the most famous landmarks in the capital of China?"
    ],
    "related_questions": [
      "What are the most visited tourist sites in China?",
      "Which places should I visit when traveling to Beijing?",
      "What are the key historical landmarks in China?",
      "What are the best tourist destinations in China?",
      "What are the most popular cultural attractions in China?",
      "What are some famous tourist spots in China?",
      "Where should tourists go when visiting Beijing?",
      "What are the top cultural and historical destinations in Beijing?",
      "What are the most iconic landmarks in China?",
      "Which places should be on my itinerary when visiting China?",
      "Which attractions in Beijing are popular among tourists?",
      "What are the must-see places in Beijing?",
      "What are the most important historical landmarks in Beijing?",
      "What are the main cultural heritage sites in Beijing?",
      "Where are the top tourist destinations in Beijing located?"
    ],
    "explanation": "This question explores the relationship 'China-Beijing,' focusing on key attractions that are sub-nodes under 'Beijing,' such as 'Forbidden City' and 'Summer Palace,' utilizing two relationships (China-Beijing and Beijing-attractions) for a comprehensive answer."
  }
]
Instructions:

Generate five distinct open-ended questions that utilize the relationships between the nodes in the graph. Ensure that each question incorporates two or three relationships for depth, but avoid questions that ask for comparisons or distinctions between entities.

For each question:

Provide 15 semantically similar questions.
Provide 15 related questions based on similar relationships.
Include a clear, logical answer derived from the graph.
Provide a brief explanation of the relationships used in the question, identifying which nodes and edges were considered to ensure depth and complexity.
Ensure that all JSON objects are in a list and follow the format provided.




    """ 



def get_graphml(base_path='output'):
    # Step 1: Find the folder with the latest modification time
    output_path= base_path + '/output'
    folders = [os.path.join(output_path, d) for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))]
    latest_folder = max(folders, key=os.path.getmtime)

    # Step 2: Find the artifacts/merged_graph.graphml file in that folder
    graphml_file_path = os.path.join(latest_folder, 'artifacts', 'merged_graph.graphml')


    # Read the graph from the GraphML file
    graph = nx.read_graphml(graphml_file_path)

    # Extract nodes and edges
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    # Convert nodes to a string
    nodes_str = ', '.join(nodes)

    # Convert edges to a string
    edges_str = '\n  - '.join([f"({u}, {v})" for u, v in edges])

    graph_description = f"Graph Description:\n- Nodes: {nodes_str}\n- Edges:\n  - {edges_str}"

    return graph_description


def get_oritext(base_path):
    # Step 1: Find the folder with the latest modification time
    folder_path= base_path + '/input'
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
    combined_text = "\n\n\n".join(combined_content)
    combined_text = "Original Text:\n" + combined_text
    return combined_text
    # # Save the combined content into a new file
    # output_file_path = os.path.join(base_path, 'combined_output.txt')
    # with open(output_file_path, 'w') as output_file:
    #     output_file.write(combined_text)
    
if __name__ == "__main__":
    base_path='/home/ljc/data/graphrag/alltest/dataset3'
    prompt_path = os.path.join(base_path, 'prompt.json')
    ori_prompt = base_prompt + get_oritext(base_path) + get_graphml(base_path) 
    
    
    # Number of times to generate questions
    num_iterations = 10  # Change this to how many times you want to generate questions

    # List to store all question sets
    all_questions = []
    prompt = ori_prompt

    for i in tqdm(range(num_iterations)):
        try:

            completion = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = completion.choices[0].message.content
            if content is not None:
                question_json = json.loads(content)["questions"]
                all_questions.extend(question_json)
            else:
                print('No response from OpenAI')
                
            combined_previous_questions = ' '.join(item["question"] for item in all_questions)
            prompt = ori_prompt + "The previous quesions which you generated are as follow. Don't generate the same quesions again. "+ combined_previous_questions
        except:
            print('Error occurred. Skipping this iteration.')
            continue
    # Write all the generated questions to a file
    prompt_path = Path(prompt_path)
    prompt_path.write_text(json.dumps(all_questions, ensure_ascii=False, indent=4), encoding='utf-8')
    print(f"All questions generated successfully and saved to {prompt_path}")

