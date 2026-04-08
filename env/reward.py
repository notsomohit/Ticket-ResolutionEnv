from .models import Ticket

def calculate_step_reward(ticket: Ticket, is_closing: bool = False) -> float:
    """
    UPGRADED REWARD LOGIC:
    - Continuous small penalty (-0.05) per step to encourage speed.
    - Large rewards (+1.0 each) for correct fields only upon closing.
    - SLA Bonus: +0.5 if resolved well within SLA.
    - Escalation Penalty: -0.2 if Tier 2 is used for Priority 1 tasks.
    """
    reward = -0.05 # Step penalty (efficiency)
    
    if is_closing:
        # Accuracy rewards
        if ticket.predicted_category == ticket.expected_category:
            reward += 1.0
        else:
            reward -= 0.5 # Penalty for wrong category
            
        if ticket.predicted_assignee == ticket.expected_assignee:
            reward += 1.0
        else:
            reward -= 0.5
            
        if ticket.predicted_resolution == ticket.expected_resolution:
            reward += 1.0
        else:
            reward -= 0.5
            
        # SLA compliance reward
        if ticket.steps_taken <= ticket.sla_steps:
            reward += 0.5
        else:
            reward -= 1.0 # Missed SLA penalty
            
        # Optimization check: Over-escalation
        if ticket.predicted_assignee == "tier2_support" and ticket.priority == 1:
            reward -= 0.5 # Avoid wasting senior resources on simple tasks
            
    return round(reward, 2)
