import os
import json
from env import TicketResolutionEnv, Action, grade_task
from openai import OpenAI

def run_inference():
    # Mandatory [START] log
    task_name = os.getenv("TASK_NAME", "easy")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    print(f"[START] task={task_name} env=support_ticket_env model={model_name}")
    
    env = TicketResolutionEnv(task_name=task_name)
    
    # OpenAI client with environment variables
    client = OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "dummy"))
    )
    
    system_prompt = """You are an expert customer support routing agent. 
You must output ONLY a valid JSON object representing your action.
Allowed action_type: "set_category", "set_assignee", "set_resolution", "close_ticket".
For "close_ticket", value should be empty string "".

Rules:
- billing -> billing_team, issue_refund
- free tech -> tier1_support
- premium/enterprise tech -> tier2_support, escalate
- account -> tier1_support, send_reset_link/provide_faq"""
    
    step_num = 0
    rewards = []
    
    try:
        obs = env.reset()
        while not env.state_obj.done and step_num < env.max_steps:
            step_num += 1
            prompt = f"Observation: {obs.model_dump_json()}\nNext Action (JSON):"
            
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
                
                if "action_type" not in action_dict:
                    raise ValueError("Missing action_type")
                if "value" not in action_dict:
                    action_dict["value"] = ""
                    
                action = Action(**action_dict)
                action_str = json.dumps(action_dict, separators=(',', ':'))
                
                obs, reward, done, info = env.step(action)
                rewards.append(reward)
                
                error_str = str(info.get("error")) if info.get("error") else "null"
                print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_str}")
                
            except Exception as e:
                # COMPLIANT: log as step then break, no [DEBUG]
                print(f"[STEP] step={step_num} action={{\"error\":\"invalid\"}} reward=0.00 done=true error={str(e).replace(' ', '_')}")
                env.state_obj.done = True
                break
    except Exception:
        # COMPLIANT: No [DEBUG]
        pass
    finally:
        # Guarantee [END] is printed
        state = env.state()
        score = grade_task(state) if state else 0.0
        success = score > 0.99
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()
