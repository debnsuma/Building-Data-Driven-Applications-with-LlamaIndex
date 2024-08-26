from llama_index.core import ( 
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.core.settings import Settings
from llama_index.llms.bedrock import Bedrock
from llama_index.embeddings.bedrock import BedrockEmbedding

# ------------------------------------------------------------------------
# LlamaIndex - Amazon Bedrock

llm = Bedrock(model = "amazon.titan-text-express-v1")
embed_model = BedrockEmbedding(model = "amazon.titan-embed-g1-text-02")

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader('files').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()


# Perform a query on the documents
response = query_engine.query("summarize each document in a few sentences")
print(response)