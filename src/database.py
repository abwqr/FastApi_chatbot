from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "mysql://localhost:3306/chatbot_db"

# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"
engine = create_engine("mysql+mysqlconnector://root:abdullah@localhost:3306/chatbot_db")
# engine = create_engine(DATABASE_URL)
# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL
# )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
 

