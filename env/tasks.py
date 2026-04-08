from typing import Dict, Any, List
from .models import State, Ticket

def get_easy_task() -> Dict[str, Any]:
    """
    EASY TASK: 1 simple billing ticket.
    Goal: Identify billing issue and assign to billing team.
    """
    return {
        "tickets": [
            {
                "id": "T001",
                "text": "I was double charged for my monthly subscription. Please refund the extra amount.",
                "customer_tier": "free",
                "priority": 1,
                "sla_steps": 15,
                "sentiment": "neutral",
                "expected_category": "billing",
                "expected_assignee": "billing_team",
                "expected_resolution": "issue_refund"
            }
        ]
    }

def get_medium_task() -> Dict[str, Any]:
    """
    MEDIUM TASK: 2 tickets, one high priority.
    Goal: Handle a tech issue for a premium user correctly.
    """
    return {
        "tickets": [
            {
                "id": "T002",
                "text": "Critical: My dashboard is not loading on Chrome, getting a 500 error. I have a presentation in 10 minutes!",
                "customer_tier": "premium",
                "priority": 3,
                "sla_steps": 8,
                "sentiment": "angry",
                "expected_category": "tech_support",
                "expected_assignee": "tier2_support",
                "expected_resolution": "escalate"
            },
            {
                "id": "T003",
                "text": "How do I change my profile picture?",
                "customer_tier": "free",
                "priority": 1,
                "sla_steps": 20,
                "sentiment": "happy",
                "expected_category": "account",
                "expected_assignee": "tier1_support",
                "expected_resolution": "provide_faq"
            }
        ]
    }

def get_hard_task() -> Dict[str, Any]:
    """
    HARD TASK: 3 complex tickets with conflicting signals and constraints.
    - T4: High-value enterprise user, angry, but issue is a simple password reset (don't over-escalate).
    - T5: Subtle bug report for enterprise API, requires tier 2.
    - T6: Billing dispute that looks like a technical error.
    """
    return {
        "tickets": [
            {
                "id": "T004",
                "text": "I AM LOCKED OUT! FIX THIS NOW! This is unacceptable for an Enterprise customer!",
                "customer_tier": "enterprise",
                "priority": 3,
                "sla_steps": 5, # Very tight SLA
                "sentiment": "angry",
                "expected_category": "account",
                "expected_assignee": "tier1_support", # Despite being enterprise/angry, it's a simple lockout.
                "expected_resolution": "send_reset_link"
            },
            {
                "id": "T005",
                "text": "Our automated systems are seeing intermittent 429s despite being 50% under our provisioned throughput.",
                "customer_tier": "enterprise",
                "priority": 2,
                "sla_steps": 12,
                "sentiment": "neutral",
                "expected_category": "tech_support",
                "expected_assignee": "tier2_support",
                "expected_resolution": "escalate"
            },
            {
                "id": "T006",
                "text": "I received a 'Transaction Failed' message but my bank shows the money was taken.",
                "customer_tier": "premium",
                "priority": 2,
                "sla_steps": 10,
                "sentiment": "neutral",
                "expected_category": "billing",
                "expected_assignee": "billing_team",
                "expected_resolution": "issue_refund"
            }
        ]
    }

def get_task(task_name: str) -> Dict[str, Any]:
    if task_name == "easy":
        return get_easy_task()
    elif task_name == "medium":
        return get_medium_task()
    elif task_name == "hard":
        return get_hard_task()
    else:
        raise ValueError(f"Unknown task {task_name}")

def grade_task(state_obj: State) -> float:
    """
    UPGRADED GRADER:
    - 40% Accuracy of fields
    - 30% SLA compliance (weighted by priority)
    - 30% Efficiency (penalty for excessive steps)
    """
    if not state_obj or not state_obj.tickets:
        return 0.0
    
    total_score = 0.0
    for t in state_obj.tickets:
        # 1. Field Accuracy (Max 1.0 per ticket)
        accuracy = 0.0
        if t.predicted_category == t.expected_category: accuracy += 0.33
        if t.predicted_assignee == t.expected_assignee: accuracy += 0.33
        if t.predicted_resolution == t.expected_resolution: accuracy += 0.34
        
        # 2. SLA Compliance (Max 1.0 per ticket)
        sla_score = 1.0 if t.steps_taken <= t.sla_steps else max(0, 1.0 - (t.steps_taken - t.sla_steps) * 0.2)
        
        # 3. Efficiency (Max 1.0 per ticket)
        # Optimal path is 4 steps: 3 set_ + 1 close.
        efficiency = 1.0 if t.steps_taken <= 4 else max(0, 1.0 - (t.steps_taken - 4) * 0.1)
        
        # Combine
        ticket_score = (accuracy * 0.4) + (sla_score * 0.3) + (efficiency * 0.3)
        total_score += ticket_score
        
    final_score = total_score / len(state_obj.tickets)
    return min(1.0, max(0.0, round(final_score, 2)))
