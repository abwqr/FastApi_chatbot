from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form, Depends
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from langchain.llms import Replicate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
import pickle
from langchain.vectorstores import FAISS
import pickle
import os

import models
from database import SessionLocal, engine
models.Base.metadata.create_all(bind=engine)
app = FastAPI()
templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Query(BaseModel):
    id: int
    question: str = Field(min_length=1, max_length = 200)
    answer: str = Field(min_length=1, max_length = 400)
    
QUERY = []

chat_history=[]

@app.get("/")
def form_post(request: Request):
    answer = "" 
    return templates.TemplateResponse('chatbot.html', context={'request': request, 'answer': answer})

@app.post("/")
def form_post(request: Request, id: str = Form(...), question: str = Form(...), db: Session=Depends(get_db)):
    answer = ""

    #model result
    
    query_model = models.Query()
    query_model.id = id
    query_model.question = question
    os.environ["REPLICATE_API_TOKEN"] = "r8_5iLKa00rjDFkHuNFBfc2oJSnAdmdQf04ZTAnC"
    loader = TextLoader("doc.txt")
    documents = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    llm = Replicate( 
        model="replicate/flan-t5-xl:7a216605843d87f5426a10d2cc6940485a232336ed04d655ef86b91e020e9210",
        input={"temperature": 0.75, "max_length": 500, "top_p": 1},
    )
    
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    
    result = qa({"question":question, "chat_history":chat_history})
    chat_history = [(question, result["answer"])]
    print(result['answer'])
    answer = result['answer']
    query_model.answer = answer
   
    result = qa({"question":question, "chat_history":chat_history})
    chat_history = [(question, result["answer"])]
    print(result['answer'])
    answer = result['answer']
    query_model.answer = answer
    db.add(query_model)
    db.commit()
    
    

    return templates.TemplateResponse('chatbot.html', context={'request': request, 'answer': answer})

