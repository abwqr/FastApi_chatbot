from langchain import PromptTemplate
from langchain.llms import Replicate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import EmbaasEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
import pinecone
import os

index_name = 'langchain-retrieval-augmentation-fast'

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') or "debe432d-e014-4ead-93bc-1f6afab5e773"
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT') or 'us-west1-gcp-free'

# CLARIFAI_PAT = "5fd8207f420b42f3b5c5504cfa9a4a09"
embaas_api_key = "752E02C21F4FC1896EDF24469852E65A06B0347FE1119C129B1FBB6F21E5ED72"
os.environ["EMBAAS_API_KEY"] = embaas_api_key
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["REPLICATE_API_TOKEN"] = "r8_BgLm14N0ZpV41kK5mYs4IKr2ON8Et0W4f6LIz"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)
if index_name not in pinecone.list_indexes():
  # we create a new index
  pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=768,
  )

loader = TextLoader("doc.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print("Embeddings:")
embeddings = EmbaasEmbeddings(
    model="instructor-large"
  )

llm = Replicate(
  # model="replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
  # model="daanelson/flan-t5-large:ce962b3f6792a57074a601d3979db5839697add2e4e02696b3ced4c022d4767f",
  model="daanelson/flan-t5-base:9233575cbf22d2a6a55ee822f121da3188c123b24f2cfc61584f5b88d6c25da8",

  input={"temperature": 0.75, "max_length": 500, "top_p": 1},
)

index = pinecone.Index(index_name)
print("Vector:")
db = Pinecone.from_documents(texts, embeddings, index_name=index_name)
print("Chain:")

prompt_template = """
      FAST NUCES is a university in Islamabad, Lahore, Karachi, Peshawar, Faislabad. Several
      degrees are offered there, ranging from BS in computer science, BBA, accounting and finance, and many more.

      Context: {context}

      Question: {question}
      Answer:
"""
prompt = PromptTemplate(
    template = prompt_template,
    input_variables=["context", "question"]
)
memory = ConversationBufferMemory(memory_key="chat_history",
                                  # k=5,
                                  # return_messages=True
                                )
qa = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)
# print("Loaded:")

# print("Running:")

# query = "What are the requirements for BS computer science?"
# result = qa(query)
# print(result)
# qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
# qa = RetrievalQA.from_chain_type(
#     llm,
#     retriever=db.as_retriever())