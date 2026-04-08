from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import TicketResolutionEnv, Action

app = FastAPI()

# Global env instance for the space
env_instance = None

class ResetRequest(BaseModel):
    task: str = "easy"

@app.post("/reset")
def reset_env(req: ResetRequest):
    global env_instance
    try:
        env_instance = TicketResolutionEnv(task_name=req.task)
        obs = env_instance.reset()
        # obs is an Observation model, so we can return its dict/dump
        return {"observation": obs.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_env(action: Action):
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    
    try:
        obs, reward, done, info = env_instance.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/state")
def get_state():
    global env_instance
    if env_instance is None:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return {"state": env_instance.state().model_dump()}
