import os
import json
from neo4j import GraphDatabase
import xml.etree.ElementTree as ET
from openai import OpenAI
import transformers
import torch
from tqdm import tqdm
import yaml

# Constants
WORKING_DIR = "/home/ljc/data/graphrag/alltest/0411/cyber_v3_tobeuse/output/20250411-204229/artifacts"
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100

# Neo4j connection credentials https://console-preview.neo4j.io/projects
with open('process_code/neo4j.yaml', 'w') as file:
    config = yaml.safe_load(file)

NEO4J_URI= config.get('NEO4J_URI')
NEO4J_USERNAME= config.get('NEO4J_USERNAME')
NEO4J_PASSWORD=config.get('NEO4J_PASSWORD')
AURA_INSTANCEID=config.get('AURA_INSTANCEID')
AURA_INSTANCENAME=config.get('AURA_INSTANCENAME')

base_prompt = "You are a help assistant to extract relationship keyword from a given relationship description. Given a relationship discription, output one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details."

def gpt(description):
    client = OpenAI()
    completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": description}
            ]
        )
    return completion.choices[0].message.content

class Llama:
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        # Load the Llama model and tokenizer using Hugging Face's transformers pipeline
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto"  # Automatically place the model on GPU if available
        )
        self.base_prompt = base_prompt 

    def query(self, description):
        # Prepare the messages for the query
        messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": description}
        ]
        
        # Perform text generation using the pipeline
        outputs = self.pipeline(
            messages,  # Pass user input
            max_new_tokens=256,  # Limit on number of new tokens to generate
        )
        # Return the generated content
        return outputs[0]["generated_text"][-1]["content"]


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {
            "nodes": [],
            "edges": []
        }

        # Use empty string for the default namespace
        namespace = {'': 'http://graphml.graphdrawing.org/xmlns'}

        # Iterate through nodes
        for node in root.findall('.//{http://graphml.graphdrawing.org/xmlns}node'):
            node_data = {
                "id": node.get('id'),  # No need to strip quotes
                "entity_type": node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d0']").text if node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d0']") is not None else "",
                "description": node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d1']").text if node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d1']") is not None else "",
                "source_id": node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d2']").text if node.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d2']") is not None else ""
            }
            data["nodes"].append(node_data)

        # llama3 = Llama()

        # Iterate through edges
        for edge in tqdm(root.findall('.//{http://graphml.graphdrawing.org/xmlns}edge')):
            edge_data = {
                "source": edge.get('source'),  # No need to strip quotes
                "target": edge.get('target'),  # No need to strip quotes
                "weight": float(edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d3']").text) if edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d3']") is not None else 0.0,
                "description": edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d4']").text if edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d4']") is not None else "",
                # "keywords": edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d5']").text if edge.find("./{http://graphml.graphdrawing.org/xmlns}data[@key='d5']") is not None else ""
            }
            #edge_data["keywords"] = llama3.query(edge_data["description"])
            edge_data["keywords"] = gpt(edge_data["description"])
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None

def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})

def main():
    # Paths
    xml_file = os.path.join(WORKING_DIR, 'merged_graph.graphml')
    json_file = os.path.join(WORKING_DIR, 'graph_data.json')

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get('nodes', [])
    edges = json_data.get('edges', [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id  
    REMOVE e:Entity  
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Insert nodes in batches
            session.execute_write(process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES)

            # Insert edges in batches
            session.execute_write(process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES)

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        driver.close()

def main2(): 
    # Paths
    xml_file = os.path.join(WORKING_DIR, 'merged_graph.graphml')
    json_file = os.path.join(WORKING_DIR, 'graph_data.json')

    # Convert XML to JSON
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        
    if json_data is None:
        return

    # Load nodes and edges
    nodes = json_data.get('nodes', [])
    edges = json_data.get('edges', [])

    # Neo4j queries
    create_nodes_query = """
    UNWIND $nodes AS node
    MERGE (e:Entity {id: node.id})
    SET e.entity_type = node.entity_type,
        e.description = node.description,
        e.source_id = node.source_id,
        e.displayName = node.id  
    REMOVE e:Entity  
    WITH e, node
    CALL apoc.create.addLabels(e, [node.entity_type]) YIELD node AS labeledNode
    RETURN count(*)
    """

    create_edges_query = """
    UNWIND $edges AS edge
    MATCH (source {id: edge.source})
    MATCH (target {id: edge.target})
    WITH source, target, edge,
         CASE
            WHEN edge.keywords CONTAINS 'lead' THEN 'lead'
            WHEN edge.keywords CONTAINS 'participate' THEN 'participate'
            WHEN edge.keywords CONTAINS 'uses' THEN 'uses'
            WHEN edge.keywords CONTAINS 'located' THEN 'located'
            WHEN edge.keywords CONTAINS 'occurs' THEN 'occurs'
           ELSE REPLACE(SPLIT(edge.keywords, ',')[0], '\"', '')
         END AS relType
    CALL apoc.create.relationship(source, relType, {
      weight: edge.weight,
      description: edge.description,
      keywords: edge.keywords,
      source_id: edge.source_id
    }, target) YIELD rel
    RETURN count(*)
    """

    set_displayname_and_labels_query = """
    MATCH (n)
    SET n.displayName = n.id
    WITH n
    CALL apoc.create.setLabels(n, [n.entity_type]) YIELD node
    RETURN count(*)
    """

    clear_database_query = """
    MATCH (n)
    DETACH DELETE n
    """

    # Create a Neo4j driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        # Execute queries in batches
        with driver.session() as session:
            # Clear the existing data in the database
            session.run(clear_database_query)

            # Insert nodes in batches
            session.execute_write(process_in_batches, create_nodes_query, nodes, BATCH_SIZE_NODES)

            # Insert edges in batches
            session.execute_write(process_in_batches, create_edges_query, edges, BATCH_SIZE_EDGES)

            # Set displayName and labels
            session.run(set_displayname_and_labels_query)

    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        driver.close()

if __name__ == "__main__":
    xml_file = os.path.join(WORKING_DIR, 'merged_graph.graphml')
    json_file = os.path.join(WORKING_DIR, 'graph_data.json')
    # _ = convert_xml_to_json(xml_file, json_file)
    main2()

