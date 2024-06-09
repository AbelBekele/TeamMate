from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

class DocumentLoader:
    def __init__(self, embeddings):
        self.client = weaviate.connect_to_local()
        self.embeddings = embeddings

    def load_documents(self, file_path: str):
        loader = PyMuPDFLoader(file_path)
        data = loader.load()

        # Read data from the file and put them into a variable called text
        text = ''
        for document in data:
            page_text = document.page_content
            if page_text:
                try:
                    text += page_text
                except UnicodeDecodeError:
                    # If there is an issue with decoding, you can log it or handle it as needed
                    pass

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(text)
        docsearch = WeaviateVectorStore.from_texts(
            texts,
            self.embeddings,
            client=self.client,
            metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]
        )
        return docsearch

def connect_weaviate(url: str, api_key: str):
    client = weaviate.Client(
        url=url,
        additional_headers={"X-OpenAI-API-Key": api_key}
    )
    return client
