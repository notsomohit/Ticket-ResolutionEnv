from fastapi import FastAPI, HTTPException, Request
from env import TicketResolutionEnv, Action

app = FastAPI()

# Global environment instance
env_instance = None


@app.post("/reset")
async def reset_env(request: Request):
    global env_instance

    # Safely read body (handles empty body from judges)
    try:
        body = await request.json()
    except:
        body = {}

    task = body.get("task", "easy")

    try:
        env_instance = TicketResolutionEnv(task_name=task)
        obs = env_instance.reset()
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