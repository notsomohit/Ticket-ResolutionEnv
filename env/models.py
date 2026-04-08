from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class Ticket(BaseModel):
    id: str
    text: str
    customer_tier: Literal["free", "premium", "enterprise"]
    priority: int = 1  # 1: Low, 2: Medium, 3: High
    sla_steps: int = 10 # Steps allowed to resolve
    sentiment: str = "neutral"
    
    expected_category: str
    expected_assignee: str
    expected_resolution: str
    
    predicted_category: Optional[str] = None
    predicted_assignee: Optional[str] = None
    predicted_resolution: Optional[str] = None
    status: str = "open"
    steps_taken: int = 0

class Observation(BaseModel):
    ticket_id: Optional[str]
    ticket_text: Optional[str]
    customer_tier: Optional[str]
    priority: int
    sla_remaining: int
    sentiment: str
    current_category: Optional[str]
    current_assignee: Optional[str]
    current_resolution: Optional[str]
    queue_length: int
    last_message: str

class Action(BaseModel):
    action_type: Literal["set_category", "set_assignee", "set_resolution", "close_ticket"]
    value: str

class State(BaseModel):
    tickets: List[Ticket]
    current_ticket_index: int
    total_reward: float
    steps: int
    done: bool
    last_action: Optional[Action] = None
    repeat_count: int = 0
    tier2_slots_remaining: int = 2 # Constraint: limited senior agents
