import os
import asyncio
import json
import logging
import secrets
from typing import Dict, Any, List, Optional
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException, Query, Depends
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

# XÁC THỰC API TỐI GIẢN (Giai đoạn 3 -- vá lỗ hổng "ai cũng điều khiển được
# simulation"): sinh một token phiên ngẫu nhiên nếu người dùng không tự đặt
# biến môi trường AI_ECONOMIST_API_TOKEN. Mọi request tới /api/* và kết nối
# /ws/stream đều phải mang đúng token này. Đây là mô hình xác thực phù hợp
# cho một sandbox nghiên cứu chạy local/LAN một người dùng -- không cần OAuth
# đầy đủ, nhưng đủ để chặn truy cập trái phép qua mạng.
API_TOKEN = os.environ.get("AI_ECONOMIST_API_TOKEN") or secrets.token_urlsafe(24)
if not os.environ.get("AI_ECONOMIST_API_TOKEN"):
    # Dong log nay CO CHU Y khong dung dau tieng Viet: console mac dinh cua
    # Windows (khong phai UTF-8) co the hien thi sai dau, khien dong quan
    # trong nhat (chua API token) bi kho doc/kho copy. Dong khung ro rang de
    # nguoi dung de nhan ra va copy chinh xac ma khong bi lan voi log khac.
    logger.warning(
        "\n"
        + "=" * 78 + "\n"
        + "[SECURITY] No AI_ECONOMIST_API_TOKEN set. Generated a random session token.\n"
        + "Paste this EXACT string into the 'API key' field on the dashboard (top-right):\n"
        + "\n"
        + f"    {API_TOKEN}\n"
        + "\n"
        + "(Or set it yourself before starting the server: "
        + "set AI_ECONOMIST_API_TOKEN=<your-token>  [cmd]  /  "
        + "$env:AI_ECONOMIST_API_TOKEN='<your-token>'  [PowerShell])\n"
        + "=" * 78
    )

def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

# CORS: KHÔNG dùng wildcard "*" kết hợp allow_credentials=True (kết hợp này bị
# chính đặc tả CORS/trình duyệt coi là cấu hình không an toàn). Chỉ cho phép
# các origin tường minh -- mặc định là host chạy chính server này; có thể mở
# rộng qua biến môi trường AI_ECONOMIST_CORS_ORIGINS (danh sách phân tách bởi
# dấu phẩy) khi cần phục vụ frontend từ một origin khác.
_default_origins = "http://localhost:8000,http://127.0.0.1:8000"
CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get("AI_ECONOMIST_CORS_ORIGINS", _default_origins).split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
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
        self.env: MacroEnvironment = MacroEnvironment(num_employees=50, num_firms=5, num_banks=1, max_steps=480)
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
        self.policy_overrides: Dict[str, float] = {}
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
        subdirs = [
            d for d in os.listdir(cp_dir)
            if os.path.isdir(os.path.join(cp_dir, d)) and d.startswith("iter_")
        ]
        if not subdirs:
            return None

        def parse_iter(name: str) -> int:
            # iter_20 -> 20, iter_47_interrupt -> 47
            parts = name.split("_")
            return int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else -1

        # Ưu tiên checkpoint định kỳ (không interrupt) để tránh load weights chưa flush đủ
        regular = [d for d in subdirs if "interrupt" not in d]
        pool = regular if regular else subdirs

        # Sort theo số iteration, lấy cao nhất
        pool.sort(key=parse_iter)
        return pool[-1]

    def load_checkpoint(self, checkpoint_name: str) -> bool:
        if checkpoint_name == "heuristic" or not checkpoint_name:
            self.active_checkpoint = "heuristic"
            logger.info("[SERVER] Switched to Heuristic rule-based engine.")
            return True

        cp_dir = "be/checkpoint/training"
        # BẢO MẬT: whitelist chính xác theo os.listdir() thay vì nối chuỗi trực
        # tiếp vào os.path.join(). checkpoint_name đến từ request của client --
        # nếu không kiểm tra, một giá trị là đường dẫn tuyệt đối (vd. "C:\\...")
        # sẽ khiến os.path.join() BỎ QUA cp_dir và cho phép self.algo.restore()
        # nạp bất kỳ đường dẫn nào trên máy (path traversal / arbitrary-file
        # restore -- Ray/RLlib checkpoint restore deserialize nội dung, tiềm ẩn
        # rủi ro thực thi mã nếu checkpoint đến từ nguồn không tin cậy).
        valid_names = set()
        if os.path.exists(cp_dir):
            valid_names = {d for d in os.listdir(cp_dir) if os.path.isdir(os.path.join(cp_dir, d))}
        if checkpoint_name not in valid_names:
            logger.error(f"[SERVER] Rejected checkpoint load request for unrecognized name: {checkpoint_name!r}")
            return False

        target_path = os.path.join(cp_dir, checkpoint_name)

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

    def apply_policy_shock(self, overrides: Dict[str, float]):
        """Luu tru can thiep va ghi log su kien POLICY_SHOCK."""
        for k in ["worker_tax", "firm_tax", "lending_rate", "living_cost"]:
            if k in overrides:
                self.policy_overrides[k] = float(overrides[k])

        # Dong bo truc tiep vao the gioi vat ly
        if "living_cost" in self.policy_overrides:
            self.env.eco.base_living_cost = self.policy_overrides["living_cost"]

        # Ghi vao hang doi log UI su kien cam quyen
        shock_event = {
            "type": "POLICY_SHOCK",
            "source": "RULER_CONSOLE",
            "target": "MACRO_SYSTEM",
            "payload": {
                "worker_tax": round(self.policy_overrides.get("worker_tax", self.env.gov.tax_rate_worker) * 100, 1),
                "firm_tax": round(self.policy_overrides.get("firm_tax", self.env.gov.tax_rate_firm) * 100, 1),
                "lending_rate": round(self.policy_overrides.get("lending_rate", self.env.bank.lending_rate) * 100, 2),
                "living_cost": round(self.policy_overrides.get("living_cost", self.env.eco.base_living_cost), 1)
            },
            "timestep": self.current_step
        }
        self.logger.ui_event_queue.append(shock_event)
        logger.info(f"[POLICY SHOCK ENFORCED] {self.policy_overrides}")

    def configure_population(self, num_employees: Optional[int], num_firms: Optional[int], num_banks: Optional[int]) -> Dict[str, int]:
        """Đổi quy mô dân số/doanh nghiệp/ngân hàng cho một simulation MỚI (áp
        dụng ngay một RESET). Biên hợp lý được kẹp (clip) để tránh cấu hình phi
        thực tế (0 người/0 firm) hoặc quá tải bộ nhớ cho một dashboard tương tác
        thời gian thực."""
        if num_employees is not None:
            self.env.num_employees = int(np.clip(int(num_employees), 5, 300))
        if num_firms is not None:
            self.env.num_firms = int(np.clip(int(num_firms), 1, 30))
        if num_banks is not None:
            self.env.num_banks = int(np.clip(int(num_banks), 1, 5))

        self.is_running = False
        self.current_step = 0
        self.policy_overrides.clear()
        self.env.reset(seed=42)

        return {
            "num_employees": self.env.num_employees,
            "num_firms": self.env.num_firms,
            "num_banks": self.env.num_banks,
        }

    def get_actions(self) -> Dict[str, np.ndarray]:
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

        # KHOA CUONG BUC: Ghi de len hanh dong cua AI bang gia tri nguoi dung thiet lap
        if self.policy_overrides:
            if "gov_1" in actions:
                gov_act = np.array(actions["gov_1"], dtype=np.float32, copy=True)
                if "worker_tax" in self.policy_overrides:
                    gov_act[0] = self.policy_overrides["worker_tax"]
                if "firm_tax" in self.policy_overrides:
                    gov_act[1] = self.policy_overrides["firm_tax"]
                actions["gov_1"] = gov_act

            # Cú sốc lãi suất áp dụng CHO TOÀN BỘ ngân hàng trong hệ thống
            # (giống một chỉ thị lãi suất kiểu ngân hàng trung ương), không chỉ
            # một bank_id cố định -- cần thiết vì số lượng/ID ngân hàng có thể
            # thay đổi khi người dùng cấu hình lại quy mô (xem CONFIGURE_POPULATION).
            if "lending_rate" in self.policy_overrides:
                for agent_id in list(actions.keys()):
                    if agent_id.startswith("bank_"):
                        bank_act = np.array(actions[agent_id], dtype=np.float32, copy=True)
                        bank_act[0] = self.policy_overrides["lending_rate"]
                        actions[agent_id] = bank_act

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
            try:
                if self.is_running:
                    self.current_step += 1
                    actions = self.get_actions()

                    obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

                    # Re-apply policy overrides sau mỗi env.step() để duy trì lock bền vững.
                    # Nếu không làm điều này, Economy agent sẽ multiply factor lên giá ngay
                    # tick tiếp theo, khiến override của người dùng mất tác dụng sau 1 bước.
                    if "living_cost" in self.policy_overrides:
                        self.env.eco.base_living_cost = self.policy_overrides["living_cost"]

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
            except Exception as exc:
                logger.error(f"[SERVER] Error in simulation tick {self.current_step}: {str(exc)}", exc_info=True)
                self.is_running = False

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

@app.post("/api/control", dependencies=[Depends(require_api_key)])
async def handle_control(req: ControlRequest):
    if req.action == "PLAY":
        sim_controller.is_running = True
    elif req.action == "PAUSE":
        sim_controller.is_running = False
    elif req.action == "STEP":
        sim_controller.is_running = False
        actions = sim_controller.get_actions()
        obs, rewards, terminateds, truncateds, infos = sim_controller.env.step(actions)
        sim_controller.current_step += 1
        # Re-apply override sau step (nhất quán với simulation_loop)
        if "living_cost" in sim_controller.policy_overrides:
            sim_controller.env.eco.base_living_cost = sim_controller.policy_overrides["living_cost"]
        state = sim_controller.env.export_full_world_state()

        events = [
            {
                "type": e["type"],
                "source": e["source"],
                "target": e["target"],
                "payload": e["payload"],
                "timestep": e["timestep"]
            }
            for e in sim_controller.logger.ui_event_queue
        ]
        sim_controller.logger.flush_ui_events()

        await sim_controller.broadcast({
            "type": "SIMULATION_TICK",
            "timestep": sim_controller.current_step,
            "macro": state["macro"],
            "agents": state["agents"],
            "events": events
        })
    elif req.action == "RESET":
        sim_controller.is_running = False
        sim_controller.current_step = 0
        sim_controller.policy_overrides.clear() # Xoa bo can thiep khi reset
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
    elif req.action == "SET_POLICY" and isinstance(req.value, dict):
        sim_controller.apply_policy_shock(req.value)
    elif req.action == "CONFIGURE_POPULATION" and isinstance(req.value, dict):
        cfg = sim_controller.configure_population(
            num_employees=req.value.get("num_employees"),
            num_firms=req.value.get("num_firms"),
            num_banks=req.value.get("num_banks"),
        )
        state = sim_controller.env.export_full_world_state()
        await sim_controller.broadcast({
            "type": "SIMULATION_RESET",
            "timestep": 0,
            "macro": state["macro"],
            "agents": state["agents"],
            "events": [],
            "population_config": cfg
        })
        return {"status": "SUCCESS", "current_running": sim_controller.is_running, "population_config": cfg}

    return {"status": "SUCCESS", "current_running": sim_controller.is_running}

@app.get("/api/checkpoints", dependencies=[Depends(require_api_key)])
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
async def websocket_endpoint(websocket: WebSocket, token: Optional[str] = Query(default=None)):
    # WebSocket API của trình duyệt không cho gửi custom header khi bắt tay, nên
    # token được truyền qua query string thay vì header X-API-Key như /api/*.
    if token is None or not secrets.compare_digest(token, API_TOKEN):
        await websocket.close(code=4401)
        return

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