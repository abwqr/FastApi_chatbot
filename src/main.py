from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form, Depends, Query
from pydantic import BaseModel
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from chain import qa
import os

import models
from database import SessionLocal, engine
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class User(BaseModel):
    userId = int
    userName = str
    degree = str
    phoneNum  = int
    class Config:
      orm_mode = True
      
class Chat(BaseModel):
    queryId = int
    userId = int
    question = str
    answer = str

    class Config:
      orm_mode = True
      
    
# QUERY = []

# chat_history=[]


@app.post("/")
def login_post(request: Request, db: Session=Depends(get_db), userId: int = Form(...)):
    user_model = db.query(models.User).filter(models.User.userId == userId).first()
    if user_model is None:
        return RedirectResponse(url=f"/signup?user_id={userId}", status_code=303) 
    
    return RedirectResponse(url=f"/chatbot?user_id={userId}", status_code=303) 
            
        # return templates.TemplateResponse('chatbot.html', context={'request': request})

    # return templates.TemplateResponse('greeting.html', context={'request': request})
        # return False
    # return True
    
@app.get("/")
def login_post(request: Request, db: Session=Depends(get_db)):
    # answer = "" 
    return templates.TemplateResponse('login.html', context={'request': request})
    # return templates.TemplateResponse('login.html', context={'request': request})


@app.get("/chatbot")
def form_post(request: Request, user_id: int = Query(...), db: Session=Depends(get_db)):
    answer = "" 
    # return templates.TemplateResponse('chatbot_2.html', context={'request': request})

    updated_chat = db.query(models.Chat).filter(models.Chat.userId == user_id).all()

    return templates.TemplateResponse('chatbot_2.html', context={'request': request, "chat":updated_chat})


@app.post("/chatbot")
def form_post(request: Request, db: Session=Depends(get_db),user_id: int = Query(...), question: str = Form(...)):
    answer = ""

    #model result
    
    chat = models.Chat()
    
    chat.userId = user_id
    chat.question = question
    result = qa(question)
    chat.answer = result['result']
    db.add(chat)
    db.commit()
    updated_chat = db.query(models.Chat).filter(models.Chat.userId == user_id).all()


    return templates.TemplateResponse('chatbot_2.html', context={'request': request, "chat":updated_chat
    })


@app.get("/signup")
def form_post(request: Request, db: Session=Depends(get_db)):

    return templates.TemplateResponse('signup.html', context={'request': request})

@app.post("/signup")
def form_post(request: Request, db: Session=Depends(get_db), user_id: int = Query(...), userName: str = Form(...), degree: str = Form(...), phoneNum: str = Form(...)):
    answer = ""

    #model result
    
    user = models.User()
    user.userId = user_id
    user.userName = userName
    user.degree = degree 
    user.phoneNum = phoneNum

    # user.

    db.add(user)
    db.commit()
    return RedirectResponse(url=f"/chatbot?user_id={user_id}", status_code=303) 

    # return templates.TemplateResponse('signup.html', context={'request': request})

    # result = qa({"query":question})
    # answer = result['result']

    """
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
    """
    


