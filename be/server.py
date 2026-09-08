import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from be.core.enums import LifeCycleStatus
from be.env import MacroEnvironment
from be.logger import InstitutionalLogger
from be.rllib_wrapper import (
    RLlibMacroEnv,
    policy_mapping_fn,
    EMPLOYEE_OBS_SPACE, EMPLOYEE_ACT_SPACE,
    FIRM_OBS_SPACE, FIRM_ACT_SPACE,
    GOVERNMENT_OBS_SPACE, GOVERNMENT_ACT_SPACE,
    BANK_OBS_SPACE, BANK_ACT_SPACE,
    SUPERVISOR_OBS_SPACE, SUPERVISOR_ACT_SPACE,
    ECONOMY_OBS_SPACE, ECONOMY_ACT_SPACE
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s")
logger = logging.getLogger("InstitutionalEconomist.Server")

app = FastAPI(title="Institutional AI Economist Control Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("fe/css", exist_ok=True)
os.makedirs("fe/js", exist_ok=True)
os.makedirs("fe/assets", exist_ok=True)

app.mount("/css", StaticFiles(directory="fe/css"), name="css")
app.mount("/js", StaticFiles(directory="fe/js"), name="js")
if os.path.exists("fe/assets"):
    app.mount("/assets", StaticFiles(directory="fe/assets"), name="assets")

class SimulationController:
    """Dieu phoi mo phong va suy luan trong so PPO Checkpoint."""
    def __init__(self):
        self.env: MacroEnvironment = MacroEnvironment(num_employees=50, num_firms=5, max_steps=480)
        self.logger: InstitutionalLogger = InstitutionalLogger(flush_interval=50)
        self.is_running: bool = False
        self.speed_delay: float = 0.2
        self.current_step: int = 0
        self.active_connections: List[WebSocket] = []
        
        # RLlib Inference Engine
        self.algo = None
        self.active_checkpoint: Optional[str] = None
        
        for event_type in self.env.event_bus._subscribers.keys():
            self.env.event_bus.subscribe(event_type, self.logger.log_event_for_ui)
        
        self.env.reset(seed=42)
        self._init_rllib_inference()

    def _init_rllib_inference(self):
        """Khoi tao worker suy luan RLlib 0-runner va nap checkpoint moi nhat."""
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False)
            policies = {
                "policy_employee": (None, EMPLOYEE_OBS_SPACE, EMPLOYEE_ACT_SPACE, {}),
                "policy_firm": (None, FIRM_OBS_SPACE, FIRM_ACT_SPACE, {}),
                "policy_government": (None, GOVERNMENT_OBS_SPACE, GOVERNMENT_ACT_SPACE, {}),
                "policy_bank": (None, BANK_OBS_SPACE, BANK_ACT_SPACE, {}),
                "policy_supervisor": (None, SUPERVISOR_OBS_SPACE, SUPERVISOR_ACT_SPACE, {}),
                "policy_economy": (None, ECONOMY_OBS_SPACE, ECONOMY_ACT_SPACE, {})
            }
            config = (
                PPOConfig()
                .environment(env=RLlibMacroEnv, env_config={"num_employees": 50, "num_firms": 5, "max_steps": 480})
                .framework("torch")
                .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
                .training(model={"fcnet_hiddens": [64, 64]})  # DONG QUYET DINH DE KHOP SHAPE CHECKPOINT
                .resources(num_gpus=0)
                .env_runners(num_env_runners=0)
            )
            self.algo = config.build_algo() if hasattr(config, "build_algo") else config.build()
            
            latest_cp = self._get_latest_checkpoint()
            if latest_cp:
                self.load_checkpoint(latest_cp)
        except Exception as exc:
            logger.warning(f"[SERVER] Cannot initialize Ray RLlib inference engine: {str(exc)}. Falling back to heuristic.")
            self.algo = None

    def _get_latest_checkpoint(self) -> Optional[str]:
        cp_dir = "be/checkpoint/training"
        if not os.path.exists(cp_dir):
            return None
        subdirs = [d for d in os.listdir(cp_dir) if os.path.isdir(os.path.join(cp_dir, d))]
        if not subdirs:
            return None
        # Sap xep theo thoi gian tao moi nhat
        subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(cp_dir, d)), reverse=True)
        return subdirs[0]

    def load_checkpoint(self, checkpoint_name: str) -> bool:
        if checkpoint_name == "heuristic" or not checkpoint_name:
            self.active_checkpoint = "heuristic"
            logger.info("[SERVER] Switched to Heuristic rule-based engine.")
            return True

        target_path = os.path.join("be/checkpoint/training", checkpoint_name)
        if not os.path.exists(target_path):
            logger.error(f"[SERVER] Checkpoint path not found: {target_path}")
            return False

        if self.algo is None:
            logger.warning("[SERVER] RLlib algorithm instance is not built.")
            return False

        try:
            self.algo.restore(target_path)
            self.active_checkpoint = checkpoint_name
            logger.info(f"[SERVER] Successfully loaded policy weights from: {target_path}")
            return True
        except Exception as exc:
            logger.error(f"[SERVER] Failed to restore checkpoint {checkpoint_name}: {str(exc)}")
            return False

    def get_actions(self) -> Dict[str, np.ndarray]:
        """Quyet dinh hanh dong: Uu tien dung mang No-ron neu checkpoint da nap, fallback sang Heuristic."""
        actions = {}
        raw_state = self.env.get_raw_environment_state()

        for agent_id, agent in self.env.agents.items():
            if agent.status != LifeCycleStatus.ACTIVE:
                continue

            agent_obs = agent.observe(raw_state)

            if self.algo is not None and self.active_checkpoint and self.active_checkpoint != "heuristic":
                try:
                    pid = policy_mapping_fn(agent_id)
                    act = self.algo.compute_single_action(
                        observation=agent_obs.vector,
                        policy_id=pid,
                        explore=False
                    )
                    if isinstance(act, tuple):
                        act = act[0]
                    actions[agent_id] = np.asarray(act, dtype=np.float32)
                except Exception:
                    actions[agent_id] = agent.decide(agent_obs).values
            else:
                actions[agent_id] = agent.decide(agent_obs).values

        return actions

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
                actions = self.get_actions()

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
        actions = sim_controller.get_actions()
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
    elif req.action == "LOAD_CHECKPOINT":
        success = sim_controller.load_checkpoint(str(req.value))
        return {"status": "SUCCESS" if success else "FAILED", "active_checkpoint": sim_controller.active_checkpoint}

    return {"status": "SUCCESS", "current_running": sim_controller.is_running}

@app.get("/api/checkpoints")
async def list_checkpoints():
    cp_dir = "be/checkpoint/training"
    available = []
    if os.path.exists(cp_dir):
        available = [d for d in os.listdir(cp_dir) if os.path.isdir(os.path.join(cp_dir, d))]
        available.sort(key=lambda d: os.path.getmtime(os.path.join(cp_dir, d)), reverse=True)
    return {
        "checkpoints": available,
        "active": sim_controller.active_checkpoint or "heuristic"
    }

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_controller.active_connections.append(websocket)
    try:
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