import os
import json
from env import TicketResolutionEnv, Action, grade_task
from openai import OpenAI

def run_inference():
    task_name = os.getenv("TASK_NAME", "easy")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("HF_TOKEN")

    print(f"[START] task={task_name} env=support_ticket_env model={model_name}")
    
    env = TicketResolutionEnv(task_name=task_name)
    obs = env.reset()
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    system_prompt = """You are an expert customer support routing agent. 
You will be given support tickets one by one. For each ticket, you must determine:
1. Category ("billing", "tech_support", "account", "other")
2. Assignee ("tier1_support", "tier2_support", "billing_team", "account_team")
3. Resolution ("issue_refund", "send_reset_link", "escalate", "provide_faq")

Routing Rules:
- "billing" tickets -> "billing_team" and "issue_refund".
- "free" tier tech_support -> "tier1_support".
- "premium" and "enterprise" tech_support -> "tier2_support" and "escalate".
- "account" issues -> "tier1_support" and "send_reset_link" or "provide_faq".

You must output ONLY a valid JSON object representing your action.
Allowed action_type: "set_category", "set_assignee", "set_resolution", "close_ticket".
For "close_ticket", value should be empty string "".

Example:
{"action_type": "set_category", "value": "billing"}
"""
    
    step_num = 0
    rewards = []
    
    try:
        while not env.state_obj.done and step_num < env.max_steps:
            step_num += 1
            prompt = f"Observation: {obs.model_dump_json()}\nWhat is your next action? Respond in JSON format only."
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    seed=42
                )
                
                content = response.choices[0].message.content
                action_dict = json.loads(content)
                
                if "action_type" not in action_dict or "value" not in action_dict:
                    raise ValueError("Missing action_type or value in response")
                    
                action = Action(**action_dict)
                action_str = json.dumps(action_dict, separators=(',', ':'))
                
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                
                error_val = info.get("error")
                error_str = str(error_val) if error_val else "null"
                
                print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")
                
            except Exception as e:
                error_msg = str(e).replace(' ', '_').replace('\n', '_')
                print(f"[STEP] step={step_num} action={{\"error\":\"invalid_format\"}} reward=0.00 done=true error={error_msg}")
                env.state_obj.done = True
                break
    finally:
        score = grade_task(env.state())
        success = score > 0.99
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()
