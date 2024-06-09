from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_weaviate.vectorstores import WeaviateVectorStore
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore

weaviate_client = weaviate.connect_to_local()


def process_document(file_contents: bytes, client, chunk_size=1000, chunk_overlap=0):
    embeddings = OpenAIEmbeddings()

    # Decode the byte contents to a string
    document_text = file_contents.decode("windows-1252")

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(document_text)

    docsearch = WeaviateVectorStore.from_texts(
        texts,
        embeddings,
        client=weaviate_client,
        metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
    )
    return docsearch
