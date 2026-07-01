from pydantic import BaseModel
from typing import Optional


class CreateWorkspaceBody(BaseModel):
    name: str
    description: Optional[str] = None
    
class WorkspaceAnalyzeBody(BaseModel):
    document_id: str
    session_id: str
    question: str

class TaskBody(BaseModel):
    task_type: str
    document_ids: list[str]
    session_id: str
    params: dict = {}