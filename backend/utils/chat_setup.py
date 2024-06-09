# chat_setup.py
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_review_chain(docsearch):
    review_template_str = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

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

    review_chain = review_prompt_template | chat_model | StrOutputParser()

    review_chain = (
        {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )
    return review_chain
