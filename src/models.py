from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
   __tablename__ = 'usertable'
   userId = Column(Integer, primary_key=True, index=True)
   userName = Column(String(255))
   degree = Column(String(255))
   phoneNum  = Column(Integer)
   # queries = relationship("Query", back_populates="user")


class Chat(Base):
   __tablename__ = 'querytable'
   queryId = Column(Integer, primary_key=True, index=True, autoincrement=True)
   userId = Column(Integer, ForeignKey("usertable.userId"))
   question = Column(String(255))
   answer = Column(String(255))
   # user = relationship("User", back_populates="queries")
