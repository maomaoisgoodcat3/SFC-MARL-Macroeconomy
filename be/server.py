import os
import asyncio
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from be.env import MacroEnvironment
from be.logger import InstitutionalLogger

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("InstitutionalEconomist.Server")

app = FastAPI(title="Institutional AI Economist Control Server")

# Mount thu muc frontend tinh
app.mount("/css", StaticFiles(directory="fe/css"), name="css")
app.mount("/js", StaticFiles(directory="fe/js"), name="js")
if os.path.exists("fe/assets"):
    app.mount("/assets", StaticFiles(directory="fe/assets"), name="assets")

class SimulationController:
    """Dieu phoi vong lap thuc thi mo phong trong luong bat dong bo."""
    def __init__(self):
        self.env: MacroEnvironment = MacroEnvironment(num_employees=50, num_firms=5, max_steps=480)
        self.logger: InstitutionalLogger = InstitutionalLogger(flush_interval=50)
        self.is_running: bool = False
        self.speed_delay: float = 0.2
        self.current_step: int = 0
        self.active_connections: List[WebSocket] = []
        
        # Dang ky lang nghe Event Bus
        for event_type in self.env.event_bus._subscribers.keys():
            self.env.event_bus.subscribe(event_type, self.logger.log_event_for_ui)
        
        self.env.reset(seed=42)

    async def broadcast(self, message: Dict[str, Any]):
        dead_sockets = []
        payload = json.dumps(message)
        for ws in self.active_connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead_sockets.append(ws)
        for dead in dead_sockets:
            if dead in self.active_connections:
                self.active_connections.remove(dead)

    async def simulation_loop(self):
        while True:
            if self.is_running:
                self.current_step += 1
                
                # Heuristic inference cho che do demo truc quan
                actions = {}
                for agent_id, agent in self.env.agents.items():
                    raw_state = self.env.get_raw_environment_state()
                    agent_obs = agent.observe(raw_state)
                    agent_act = agent.decide(agent_obs)
                    actions[agent_id] = agent_act.values

                obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
                
                world_state = self.env.export_full_world_state()
                events = [
                    {
                        "type": e["type"],
                        "source": e["source"],
                        "target": e["target"],
                        "payload": e["payload"],
                        "timestep": e["timestep"]
                    }
                    for e in self.logger.ui_event_queue
                ]
                self.logger.flush_ui_events()

                payload = {
                    "type": "SIMULATION_TICK",
                    "timestep": self.current_step,
                    "macro": world_state["macro"],
                    "agents": world_state["agents"],
                    "events": events
                }
                await self.broadcast(payload)

                if terminateds.get("__all__", False) or self.current_step >= self.env.max_steps:
                    self.is_running = False
                    await self.broadcast({"type": "SIMULATION_ENDED", "timestep": self.current_step})

            await asyncio.sleep(self.speed_delay)

sim_controller = SimulationController()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(sim_controller.simulation_loop())

@app.get("/")
async def get_index():
    return FileResponse("fe/html/index.html")

class ControlRequest(BaseModel):
    action: str
    value: Any = None

@app.post("/api/control")
async def handle_control(req: ControlRequest):
    if req.action == "PLAY":
        sim_controller.is_running = True
    elif req.action == "PAUSE":
        sim_controller.is_running = False
    elif req.action == "STEP":
        sim_controller.is_running = False
        # Chay dung 1 step
        actions = {}
        for aid, agent in sim_controller.env.agents.items():
            obs = agent.observe(sim_controller.env.get_raw_environment_state())
            actions[aid] = agent.decide(obs).values
        sim_controller.env.step(actions)
        sim_controller.current_step += 1
        state = sim_controller.env.export_full_world_state()
        await sim_controller.broadcast({
            "type": "SIMULATION_TICK",
            "timestep": sim_controller.current_step,
            "macro": state["macro"],
            "agents": state["agents"],
            "events": []
        })
    elif req.action == "RESET":
        sim_controller.is_running = False
        sim_controller.current_step = 0
        sim_controller.env.reset(seed=42)
        state = sim_controller.env.export_full_world_state()
        await sim_controller.broadcast({
            "type": "SIMULATION_RESET",
            "timestep": 0,
            "macro": state["macro"],
            "agents": state["agents"],
            "events": []
        })
    elif req.action == "SET_SPEED":
        sim_controller.speed_delay = max(0.02, min(2.0, float(req.value)))
    return {"status": "SUCCESS", "current_running": sim_controller.is_running}

@app.get("/api/checkpoints")
async def list_checkpoints():
    cp_dir = "be/checkpoint/training"
    if not os.path.exists(cp_dir):
        return {"checkpoints": []}
    dirs = [d for d in os.listdir(cp_dir) if d.startswith("iter_")]
    dirs.sort(key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
    return {"checkpoints": dirs}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_controller.active_connections.append(websocket)
    try:
        # Gui ngay trang thai hien tai cho client vua ket noi
        state = sim_controller.env.export_full_world_state()
        await websocket.send_text(json.dumps({
            "type": "INIT_STATE",
            "timestep": sim_controller.current_step,
            "macro": state["macro"],
            "agents": state["agents"],
            "events": []
        }))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in sim_controller.active_connections:
            sim_controller.active_connections.remove(websocket)