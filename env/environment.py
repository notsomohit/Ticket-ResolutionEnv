from typing import Tuple, Dict, Any
from .models import State, Action, Observation, Ticket
from .tasks import get_task
from .reward import calculate_step_reward

class TicketResolutionEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.task_data = get_task(task_name)
        self.state_obj = None
        self.max_steps = 50

    def reset(self) -> Observation:
        self.state_obj = State(
            tickets=[Ticket(**t) for t in self.task_data["tickets"]],
            current_ticket_index=0,
            total_reward=0.0,
            steps=0,
            done=False,
            tier2_slots_remaining=2 # Constraint: Only 2 Tier2 assignments allowed per session
        )
        return self._get_obs("Environment reset. High-priority tickets may require faster resolution.")

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state_obj.done:
            return self._get_obs("Episode already done."), 0.0, True, {"error": "Episode done"}

        self.state_obj.steps += 1
        if self.state_obj.steps > self.max_steps:
            self.state_obj.done = True
            return self._get_obs("Max steps reached."), -5.0, True, {"error": "Max steps reached"}

        if self.state_obj.current_ticket_index >= len(self.state_obj.tickets):
            self.state_obj.done = True
            return self._get_obs("All tickets processed."), 0.0, True, {}

        ticket = self.state_obj.tickets[self.state_obj.current_ticket_index]
        ticket.steps_taken += 1
        
        # --- Anti-Exploit / Looping Check ---
        reward = 0.0
        if self.state_obj.last_action and self.state_obj.last_action.model_dump() == action.model_dump():
            self.state_obj.repeat_count += 1
            reward -= (0.5 * self.state_obj.repeat_count) # Increasing penalty for repetition
        else:
            self.state_obj.repeat_count = 0
        self.state_obj.last_action = action

        msg = ""
        error = None

        valid_categories = ["billing", "tech_support", "account", "other"]
        valid_assignees = ["tier1_support", "tier2_support", "billing_team", "account_team"]
        valid_resolutions = ["issue_refund", "send_reset_link", "escalate", "provide_faq"]

        if action.action_type == "set_category":
            if action.value in valid_categories:
                ticket.predicted_category = action.value
                reward += calculate_step_reward(ticket, is_closing=False)
                msg = f"Category set to {action.value}."
            else:
                reward -= 1.0
                msg = f"Invalid category {action.value}."
                error = "Invalid category"

        elif action.action_type == "set_assignee":
            if action.value in valid_assignees:
                # --- Constraint System: Tier 2 Pool ---
                if action.value == "tier2_support":
                    if self.state_obj.tier2_slots_remaining > 0:
                        self.state_obj.tier2_slots_remaining -= 1
                        ticket.predicted_assignee = action.value
                        reward += calculate_step_reward(ticket, is_closing=False)
                        msg = f"Assignee set to tier2_support. Slots left: {self.state_obj.tier2_slots_remaining}"
                    else:
                        reward -= 2.0 # Heavy penalty for ignoring constraints
                        msg = "NO TIER 2 SLOTS REMAINING. Task must be handled by Tier 1 or Billing."
                        error = "Resource exhausted"
                else:
                    ticket.predicted_assignee = action.value
                    reward += calculate_step_reward(ticket, is_closing=False)
                    msg = f"Assignee set to {action.value}."
            else:
                reward -= 1.0
                msg = f"Invalid assignee {action.value}."
                error = "Invalid assignee"

        elif action.action_type == "set_resolution":
            if action.value in valid_resolutions:
                ticket.predicted_resolution = action.value
                reward += calculate_step_reward(ticket, is_closing=False)
                msg = f"Resolution set to {action.value}."
            else:
                reward -= 1.0
                msg = f"Invalid resolution {action.value}."
                error = "Invalid resolution"

        elif action.action_type == "close_ticket":
            if not ticket.predicted_category or not ticket.predicted_assignee or not ticket.predicted_resolution:
                reward -= 2.0
                msg = "CRITICAL ERROR: Attempted to close ticket with missing fields."
                error = "Missing fields"
            else:
                ticket.status = "closed"
                reward += calculate_step_reward(ticket, is_closing=True)
                msg = f"Ticket {ticket.id} closed. SLA Steps: {ticket.steps_taken}/{ticket.sla_steps}."
                self.state_obj.current_ticket_index += 1
                
                if self.state_obj.current_ticket_index >= len(self.state_obj.tickets):
                    self.state_obj.done = True
                    msg += " Episode Finished."
        else:
            reward -= 5.0
            msg = "MALFORMED ACTION TYPE."
            error = "Malformed action"

        self.state_obj.total_reward += reward
        obs = self._get_obs(msg)
        return obs, round(reward, 2), self.state_obj.done, {"error": error}

    def _get_obs(self, msg: str) -> Observation:
        if self.state_obj.current_ticket_index < len(self.state_obj.tickets):
            t = self.state_obj.tickets[self.state_obj.current_ticket_index]
            return Observation(
                ticket_id=t.id,
                ticket_text=t.text,
                customer_tier=t.customer_tier,
                priority=t.priority,
                sla_remaining=max(0, t.sla_steps - t.steps_taken),
                sentiment=t.sentiment,
                current_category=t.predicted_category,
                current_assignee=t.predicted_assignee,
                current_resolution=t.predicted_resolution,
                queue_length=len(self.state_obj.tickets) - self.state_obj.current_ticket_index,
                last_message=msg
            )
        else:
            return Observation(
                ticket_id=None, ticket_text=None, customer_tier=None,
                priority=0, sla_remaining=0, sentiment="none",
                current_category=None, current_assignee=None, current_resolution=None,
                queue_length=0, last_message=msg
            )

    def state(self) -> State:
        return self.state_obj
