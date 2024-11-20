import os
from pathlib import Path

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete

#######################################
# install environment
# cd LightRAG
# pip install -e .
#######################################


WORKING_DIR = "LightRAG/working_dir/test1"

if __name__ == "__main__":
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # create graph and insert text
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        # llm_model_func=gpt_4o_complete
    )
    folder_path = Path("LightRAG/working_dir/test1/input/")
    texts = [file.read_text(encoding="utf-8") for file in folder_path.glob("*.txt")]
    rag.insert(texts)


    # Perform naive search
    print(
        rag.query("Which city is the most populous urban area in the most populous country in East Asia?", param=QueryParam(mode="naive"))
    )

    # Perform local search
    print(
        rag.query("Which city is the most populous urban area in the most populous country in East Asia?", param=QueryParam(mode="local"))
    )

    # Perform global search
    print(
        rag.query("Which city is the most populous urban area in the most populous country in East Asia?", param=QueryParam(mode="global"))
    )

    # Perform hybrid search
    print(
        rag.query("Which city is the most populous urban area in the most populous country in East Asia?", param=QueryParam(mode="hybrid"))
    )
