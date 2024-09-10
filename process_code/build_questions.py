import os
from pathlib import Path  
import networkx as nx

from openai import OpenAI
import json
client = OpenAI()



base_prompt = """
    Prompt:

    Using the graph provided, which consists of nodes and edges representing relationships derived from the text, generate some Multi-Source Reasoning questions. Each question should consider two, or three relationships to ensure depth and complexity. 

    The questions should be formatted as follows:

    Yes/No Questions: These questions should be answerable with a simple "Yes" or "No." Include a brief explanation to support your answer and specify the relevant entities involved. The explanation can include your knowledge, don't mention graph information in explanation.
    Example:

    [{
    "question": "Are MUSÉE MATISSE and MUSÉE RODIN in the same place? Answer with a yes or no. And then explain why.",
    "answer": "No",
    "explanation": "MUSÉE MATISSE is in Nice. MUSÉE RODIN is in Paris.",
    "entities": [[Nice, MUSÉE MATISSE], [Paris, MUSÉE RODIN]]
    },
    {
    "question": "Are CHARLES DE GAULLE in the capital  of France? Answer with a yes or no. And then explain why.",
    "answer": "Yes",
    "explanation": "CHARLES DE GAULLE is in Paris. Paris is the capital of France. ",
    "entities": [[France,Nice],[Nice,CHARLES DE GAULLE],[France,Paris]]
    }
    ]
    Simple Questions: These questions should require straightforward answers, focusing on specific details from the graph. Include relevant entities without requiring an explanation.
    Example:


    [
    {
    "question": "List some famous place or sightseeings of the capital of France?",
    "answer": "MUSÉE D'ORSAY, MUSÉE MARMOTTAN,MUSÉE DE L'ORANGERIE,POMPIDOU CENTRE,MUSÉE RODIN,MUSÉE PICASSO",
    "explanation": None,
    "entities": [[France,Paris], [PARIS, MUSÉE D'ORSAY],[PARIS, MUSÉE MARMOTTAN MONET],[PARIS, MUSÉE DE L'ORANGERIE],[PARIS, POMPIDOU CENTRE],[PARIS, MUSÉE RODIN],[PARIS, MUSÉE PICASSO]]
    }
    ]
    Instructions:

    Create 5 Simple Questions and 5 Yes/No Questions that utilize the relationships between nodes effectively. 

    Each question should consider two, or three relationships to ensure depth and complexity. which means the len("entities") should be 2 or 3.
    Don't have the specific word "entities" "entity" in the question. The questions should be natural and coherent, which will be asked by humans.
    The questions can be based on your knowledge, but the entity must be from the graph.
    Ensure that the answers are derived logically from the graph.
    Present the questions and answers in JSON format as shown in the examples. All the json objects should be in a list. Do include the other keys in the json list except the one I give you.
    If the explanation is empty or no need to explanate, give None.
    

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

if __name__ == "__main__":
    base_path='/home/ljc/data/graphrag/alltest/basedataset/test1'
    prompt_path = os.path.join(base_path, 'prompt.json')
    prompt = base_prompt + get_graphml(base_path)
    completion = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        )
    
    content = completion.choices[0].message.content
    if content is not None:
        question_json = json.loads(content)
    else:
        question_json = {}
        print('No response from OpenAI')    
    
    
    prompt_path = Path(prompt_path)
    
    # Convert the JSON object to a string and write it to the file
    prompt_path.write_text(json.dumps(question_json, ensure_ascii=False, indent=4), encoding='utf-8')
    print(f"Questions generated successfully and saved to {prompt_path}")
