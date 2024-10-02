# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Local question generation."""

import json
import logging
import random
import time
from typing import Any
from tqdm import tqdm
import tiktoken

from graphrag.query.context_builder.builders import LocalContextBuilder
from graphrag.query.context_builder.conversation_history import (
    ConversationHistory,
)
from graphrag.query.llm.base import BaseLLM, BaseLLMCallback
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.question_gen.base import BaseQuestionGen, QuestionResult
# from graphrag.query.question_gen_by_entity.system_prompt import QUESTION_SYSTEM_PROMPT
from openai import OpenAI

client = OpenAI()
log = logging.getLogger(__name__)

QUESTION_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant







"""

USER_PROMPT = """
Please must generate {question_count} or As many as possible questions 

---Data tables---

{context_data}

---Goal---

Let's focus on the selected entity {entity}. Consider the following:

Relationships where {entity} is the starting point, with subsequent nodes referred to as "leaf nodes",[selected entity,leaf]:
{related_relationships_text_source}

Relationships where {entity} is the endpoint, with preceding nodes referred to as "root nodes",[root,selected entity]:
{related_relationships_text_target}

1. Formulate questions along the path of root node -> selected entity -> leaf node. The questions should only include the root nodes without mentioning the selected entity. Try to hide the selected entity with another representation using the root nodes in the question.

2. The candidate questions should reflect the important or urgent information or themes within the data tables.

3. These questions should be answerable using the provided data tables but should not explicitly reference any specific data fields or tables in the question text.

4. The answers to these questions should be the leaf nodes itself or some content in leaf nodes' description or the text unit related to leaf nodes. 

5. At the same time you need to ask the questions to yourself without the Data tables, And then Answer within 50 words, write the answer in the gpt_answer_withoutdata field. then compare the answer with the data table to see if the answer is consistent with the data table. If the answer is not consistent approximately with the data table, you need to reconsider the question.

6. Don't use other entities and relationships(out of {entity},{related_relationships_text_source},{related_relationships_text_target}) information to generate the questions.


Please must generate {question_count} or As many as possible questions and the answers in the following json format, which includes the question, gpt_answer_withoutdata, answer, consistency, selected entity, leaf nodes, and root nodes. 

Return {question_count} or As many as possible jsons in a list.
Just output json, don't saw any other information.
"""

EXAMPLE_USE ="""
<JSON example>
{
"question": "What is the patronage of the most famous attractions in the capital of China?", 
"gpt_answer_withoutdata":"Beijing's top attractions, such as the Forbidden City, the Great Wall, and the Temple of Heaven, draw millions annually. The Forbidden City alone sees over 14 million visitors each year, while sections of the Great Wall near Beijing attract similar numbers, showcasing their global appeal and cultural significance.",
"answer": "The patronage of the most famous attractions in the capital of China is 100,000.",
"consistency": true,
"selected_entity": "BEIJING",
"leaf_nodes": ["GREAT WALL", "FORBIDDEN CITY", "SUMMER PALACE"],
"root_nodes": ["CHINA", "CAPITAL", "TOURISM"]
}
"""
class LocalQuestionGen_byentity_oneedge(BaseQuestionGen):
    """Search orchestration for global search mode."""

    def __init__(
        self,
        llm: BaseLLM,

        context_builder: LocalContextBuilder,
        entities=[],
        relationships=[],
        token_encoder: tiktoken.Encoding | None = None,
        system_prompt: str = QUESTION_SYSTEM_PROMPT,
        callbacks: list[BaseLLMCallback] | None = None,
        llm_params: dict[str, Any] | None = None,
        context_builder_params: dict[str, Any] | None = None,
    ):
        super().__init__(
            llm=llm,
            context_builder=context_builder,
            token_encoder=token_encoder,
            llm_params=llm_params,
            context_builder_params=context_builder_params,
        )
        self.entities=entities
        self.relationships=relationships
        self.system_prompt = system_prompt
        self.callbacks = callbacks

    async def agenerate(
        self,
        question_history: list[str],
        context_data: str | None,
        question_count: int,
        entity_count: int,
        **kwargs,
    ) -> tuple[list, list]:
        """
        Generate a question based on the question history and context data.

        If context data is not provided, it will be generated by the local context builder
        """
        start_time = time.time()
        multi_questions = []
        single_questions = []   
        useful_entities = []
        for ent in self.entities:
            related_relationships_source = [rel for rel in self.relationships if rel.source == ent.title ]
            related_relationships_target = [rel for rel in self.relationships if rel.target == ent.title ]
            if len(related_relationships_source) > 1 and len(related_relationships_target) > 0:
                useful_entities.append({"entity": ent, "related_relationships_source": related_relationships_source, "related_relationships_target": related_relationships_target})
        print("=======Qualified entities: ", len(useful_entities))
        if entity_count != -1:
            useful_entities = random.sample(useful_entities, entity_count)
        for ent_with_rel in tqdm(useful_entities):
            ent_with_rel_name = ent_with_rel["entity"].title
            as_source_list = []
            related_relationships_text_source = "[Selected Entity,Leaf Entity]: "
            for rel in ent_with_rel["related_relationships_source"]:
                related_relationships_text_source += f"[{rel.source}, {rel.target}],"
                as_source_list.append([rel.source, rel.target])
            
            as_target_list = []    
            # related_relationships_text_target = "[Root Entity,Selected Entity]: "    
            for rel in ent_with_rel["related_relationships_target"]:
                # related_relationships_text_target += f"[{rel.source}, {rel.target}],"
                as_target_list.append([rel.source, rel.target])
            for as_target in as_target_list:
                per_text_target = f"[Root Entity,Selected Entity]: [{as_target[0]}, {as_target[1]}]"    
                
                question_history = [f"Find all the related text units for {ent_with_rel_name}. and the text units of entities in relationships of {related_relationships_text_source} and {per_text_target}, and the relationships of {related_relationships_text_source} and {per_text_target}.     IMPORTANT: Do not lost entity in relationship {per_text_target}, and all the information about {ent_with_rel_name}"]
                
                if len(question_history) == 0:
                    question_text = ""
                    conversation_history = None
                else:
                    # construct current query and conversation history
                    question_text = question_history[-1]
                    history = [
                        {"role": "user", "content": query} for query in question_history[:-1]
                    ]
                    conversation_history = ConversationHistory.from_list(history)

                if context_data is None:
                    # generate context data based on the question history
                    context_data, context_records = self.context_builder.build_context(
                        query=question_text,
                        conversation_history=conversation_history,
                        **kwargs,
                        **self.context_builder_params,
                    )  # type: ignore
                else:
                    context_records = {"context_data": context_data}
                # log.info("GENERATE QUESTION: %s. LAST QUESTION: %s", start_time, question_text)
                
                    
                try:
                    system_prompt = self.system_prompt.format(
                         question_count=question_count
                    )
                    user_prompt = USER_PROMPT.format(context_data=context_data,
                                                     question_count=question_count,
                                                     entity=ent_with_rel_name,
                                                     related_relationships_text_source=related_relationships_text_source,
                                                     related_relationships_text_target=per_text_target) + EXAMPLE_USE

                  
                    completion = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                     messages=[
                            {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        ],
                     temperature=0.3,
                    )
                    
                    content = completion.choices[0].message.content
                    content = content.split('```json\n', 1)[-1].rsplit('\n```', 1)[0]
                    pending_questions={}
                    content_json = json.loads(content)
                    if len(content_json) > 0:
                        pending_questions["questions"] = json.loads(content)
                    else:
                        continue
                    
                    try:
                        if len(pending_questions["questions"]) > 1:
                            pending_questions["selected_entity"] = ent_with_rel_name
                            pending_questions["as_source"] = as_source_list
                            pending_questions["as_target"] = as_target_list
                            multi_questions.append(pending_questions)
                        else:
                            pending_questions["selected_entity"] = ent_with_rel_name
                            pending_questions["as_source"] = as_source_list
                            pending_questions["as_target"] = as_target_list
                            single_questions.append(pending_questions)
                    except Exception:
                        raise ValueError(f"Error in parsing response: {completion.choices[0].message.content}")
                
                except Exception:
                    log.exception("Exception in generating question")
        return single_questions, multi_questions
        

    def generate(
        self,
        question_history: list[str],
        context_data: str | None,
        question_count: int,
        **kwargs,
    ) -> QuestionResult:
        """
        Generate a question based on the question history and context data.

        If context data is not provided, it will be generated by the local context builder
        """
        start_time = time.time()
        if len(question_history) == 0:
            question_text = ""
            conversation_history = None
        else:
            # construct current query and conversation history
            question_text = question_history[-1]
            history = [
                {"role": "user", "content": query} for query in question_history[:-1]
            ]
            conversation_history = ConversationHistory.from_list(history)

        if context_data is None:
            # generate context data based on the question history
            context_data, context_records = self.context_builder.build_context(
                query=question_text,
                conversation_history=conversation_history,
                **kwargs,
                **self.context_builder_params,
            )  # type: ignore
        else:
            context_records = {"context_data": context_data}
        log.info(
            "GENERATE QUESTION: %s. QUESTION HISTORY: %s", start_time, question_text
        )
        system_prompt = ""
        try:
            system_prompt = self.system_prompt.format(
                context_data=context_data, question_count=question_count
            )
            question_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ]

            response = self.llm.generate(
                messages=question_messages,
                streaming=True,
                callbacks=self.callbacks,
                **self.llm_params,
            )

            return QuestionResult(
                response=response.split("\n"),
                context_data={
                    "question_context": question_text,
                    **context_records,
                },
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )

        except Exception:
            log.exception("Exception in generating questions")
            return QuestionResult(
                response=[],
                context_data=context_records,
                completion_time=time.time() - start_time,
                llm_calls=1,
                prompt_tokens=num_tokens(system_prompt, self.token_encoder),
            )


if __name__ == "__main__":
    import os

    import pandas as pd
    import tiktoken

    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.indexer_adapters import (
        read_indexer_covariates,
        read_indexer_entities,
        read_indexer_relationships,
        read_indexer_reports,
        read_indexer_text_units,
    )
    from graphrag.query.input.loaders.dfs import (
        store_entity_semantic_embeddings,
    )
    from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    from graphrag.query.llm.oai.embedding import OpenAIEmbedding
    from graphrag.query.llm.oai.typing import OpenaiApiType
    from graphrag.query.question_gen.local_gen import LocalQuestionGen
    from graphrag.query.structured_search.local_search.mixed_context import (
        LocalSearchMixedContext,
    )
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    base_path = "/home/ljc/data/graphrag/alltest/dataset3_poison_met"
    INPUT_DIR = base_path + "/output/20240914-133546/artifacts"
    LANCEDB_URI = f"{INPUT_DIR}/lancedb"

    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    COVARIATE_TABLE = "create_final_covariates"
    TEXT_UNIT_TABLE = "create_final_text_units"
    COMMUNITY_LEVEL = 2

    # read nodes table to get community and degree data
    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    # load description embeddings to an in-memory lancedb vectorstore
    # to connect to a remote db, specify url and port values.
    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings",
    )
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store
    )

    print(f"Entity count: {len(entity_df)}")
    entity_df.head()

    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    print(f"Relationship count: {len(relationship_df)}")
    relationship_df.head()


    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    print(f"Report records: {len(report_df)}")
    # print(reports)

    report_df.head()

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    print(f"Text unit records: {len(text_unit_df)}")
    text_unit_df.head()


    api_key = os.environ["OPENAI_API_KEY"]
    llm_model = 'gpt-4o-2024-08-06'
    embedding_model = 'text-embedding-3-small'

    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,  # OpenaiApiType.OpenAI or OpenaiApiType.AzureOpenAI
        max_retries=20,
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        # if you did not run covariates during indexing, set this to None
        # covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,  # if the vectorstore uses entity title as ids, set this to EntityVectorStoreKey.TITLE
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    # text_unit_prop: proportion of context window dedicated to related text units
    # community_prop: proportion of context window dedicated to community reports.
    # The remaining proportion is dedicated to entities and relationships. Sum of text_unit_prop and community_prop should be <= 1
    # conversation_history_max_turns: maximum number of turns to include in the conversation history.
    # conversation_history_user_turns_only: if True, only include user queries in the conversation history.
    # top_k_mapped_entities: number of related entities to retrieve from the entity description embedding store.
    # top_k_relationships: control the number of out-of-network relationships to pull into the context window.
    # include_entity_rank: if True, include the entity rank in the entity table in the context window. Default entity rank = node degree.
    # include_relationship_weight: if True, include the relationship weight in the context window.
    # include_community_rank: if True, include the community rank in the context window.
    # return_candidate_context: if True, return a set of dataframes containing all candidate entity/relationship/covariate records that
    # could be relevant. Note that not all of these records will be included in the context window. The "in_context" column in these
    # dataframes indicates whether the record is included in the context window.
    # max_tokens: maximum number of tokens to use for the context window.


    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,  # set this to EntityVectorStoreKey.TITLE if the vectorstore uses entity title as ids
        "max_tokens": 12_000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 5000)
    }

    llm_params = {
        "max_tokens": 2000,  # change this based on the token limit you have on your model (if you are using a model with 8k limit, a good setting could be 1000=1500)
        "temperature": 0.0,
    }

    question_generator = LocalQuestionGen_byentity(
        llm=llm,
        entities=entities,relationships=relationships,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )


    # question_history = [
    #     "What is the patronage of the most famous attractions in the capital of China?",
    #     "What is the patronage of the most famous attractions in the culture center city of China?",
    # ]

    question_path = os.path.join(base_path, 'question_v2.json')

    async def main():
        candidate_questions = await question_generator.agenerate(
        question_history=[], context_data=None, question_count=5)
        # print(candidate_questions)
        with open(question_path, 'w') as f:
            json.dump(candidate_questions, f, indent=4)
        
    # 运行异步函数
    import asyncio
    asyncio.run(main())