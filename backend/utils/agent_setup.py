import os
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, Tool, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.utils import query_weaviate_function
from langchain import hub
import weaviate

# Connect to Weaviate client
weaviate_client = weaviate.connect_to_local()

# Global variables for retriever and review_chain
retriever = None
review_chain = None

# Function to process uploaded file
def process_uploaded_file(content: str):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(content)

    embeddings = OpenAIEmbeddings()
    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embeddings,
        client=weaviate_client,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
    )
    retriever = docsearch.as_retriever()
    return retriever

# Function to update the global retriever
def update_retriever(content: str):
    global retriever
    retriever = process_uploaded_file(content)
    update_review_chain()

# Function to update the review chain with the new retriever
def update_review_chain():
    global review_chain, retriever

    review_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )

# Define tools and agent setup
review_template_str = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
And ask the user to provide the necessary information and document if you don't know the answer or you don't know.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer. .

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"],
        template=review_template_str,
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],
        template="{question}",
    )
)
messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

tools = [
    Tool(
        name="Documents",
        func=lambda input: review_chain.invoke(input) if review_chain else "No document uploaded yet.",
        description="""Useful when you need to answer questions
    about uploaded documents. Not useful for answering questions
    about specific For instance,
    if the question is "What does the document say about project deadlines?",
    the input should be "What does the document say about project deadlines?"
    """,
    ),
    Tool(
        name="JobWait",
        func=query_weaviate_function,
        description="""Use when asked about current wait times to get a job.
    This tool can only get the current wait time for a job application and does
    not have any information about aggregate or historical wait times. This tool returns wait times in
    days. Do not pass the word "job" as input, only the job title itself. For instance, if the question is
    "what type of jobs specific skills required for jobs in software?", the input should be 
    "Jobs in software".
    """,
    )
]

agent_prompt = hub.pull("hwchase17/openai-functions-agent")
agent_chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
)

agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    # return_intermediate_steps=True,
    # verbose=True,
)
