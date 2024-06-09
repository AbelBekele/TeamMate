# agent_executor.py
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain_openai import ChatOpenAI
from langchain import hub
from utils.job_wait_time import get_current_wait_time
from utils.review_chain import ReviewChain

def create_agent_executor(retriever):
    review_chain = ReviewChain(retriever)

    tools = [
        Tool(
            name="Documents",
            func=lambda input: review_chain.invoke(input.get('context', ''), input.get('question', '')),
            description="""Useful when you need to answer questions
            about uploaded documents. Not useful for answering questions
            about specific For instance,
            if the question is "What does the document say about project deadlines?",
            the input should be "What does the document say about project deadlines?"
            """,
        ),
        Tool(
            name="JobWait",
            func=get_current_wait_time,
            description="""Use when asked about current wait times to get a job.
            This tool can only get the current wait time for a job application and does
            not have any information about aggregate or historical wait times. This tool returns wait times in
            days. Do not pass the word "job" as input, only the job title itself. For instance, if the question is
            "How long will I wait for a Software Engineer position?", the input should be 
            "What is the wait time for a Software Engineer position".
            """,
        ),
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
        return_intermediate_steps=True,
        verbose=True,
    )

    return agent_executor
