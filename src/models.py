from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class Query(Base):
   __tablename__ = 'query'
   id = Column(Integer, primary_key=True, index=True)
   question = Column(String(200))
   answer = Column(String(400))
   