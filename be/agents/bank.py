from typing import Dict, Any, List, Tuple
import numpy as np
from be.core.enums import LifeCycleStatus, AgentType
from be.core.types import Observation, Action, ValidationResult, TransitionResult
from be.agents.base_agent import BaseAgent


def compute_npl_ratio_pct(non_performing_loans: float, total_loans: float) -> float:
    """Ty le no xau (%), LUON trong [0, 100].

    LOI DA SUA: cong thuc cu NPL / max(tong du no, 1) cho ra 471-482% (vuot 100%) khi so vay
    xap xi 0 nhung no xau van dang "mo" trong cua so write-off (IFRS 9 staging) -- day la
    hien tuong cua mau so, khong phai no xau 480%. Dinh nghia chuan Financial Soundness
    Indicators cua IMF (IMF (2019), "Financial Soundness Indicators Compilation Guide") la
    NPL / TONG DU NO GOP (gross loans, GOM ca no xau); o day total_loans co the da loai phan
    no xau da ghi nhan mat, nen mau so = total_loans + NPL de tong du no gop >= NPL.
    """
    npl = max(0.0, float(non_performing_loans))
    gross_loans = max(0.0, float(total_loans)) + npl
    return 0.0 if gross_loans <= 0.0 else float(npl / gross_loans * 100.0)

class Bank(BaseAgent):
    """
    Tac tu Ngan hang (Bank).
    Quan ly thanh khoan, lai suat, tin dung va no xau toan he thong.
    Tuan thu nghiem ngat contract BaseAgent theo SAS v1.0.
    """
    def __init__(self, agent_id: str, npl_flow_penalty_coef: float = 0.06, npl_stock_penalty_coef: float = 50.0,
                 npl_writeoff_months: int = 6):
        super().__init__(agent_id)
        self.agent_type = AgentType.BANK

        # He so hieu chinh reward (calibration constants, xem calculate_reward)
        # -- co the cau hinh qua ScenarioConfig, KHONG doi dang ham reward.
        # Truyen o constructor (giong Government) vi Bank duoc tao lai moi
        # lan reset() (_create_world()).
        self.npl_flow_penalty_coef: float = float(npl_flow_penalty_coef)
        self.npl_stock_penalty_coef: float = float(npl_stock_penalty_coef)

        # So thang mot khoan no xau duoc phep "dang mo" tren so sach truoc khi
        # bi WRITE OFF (xem giai thich day du + trich dan tai apply_result).
        # HE SO CAU TRUC TU DO HIEU CHINH.
        self.npl_writeoff_months: int = int(npl_writeoff_months)
        # Danh sach (so tien, thang phat sinh) cho tung DOT no xau con "song"
        # tren so theo doi NPL -- can thiet de biet dot nao da qua han write-off.
        self.npl_vintages: List[Tuple[float, int]] = []

        # Co cau tai san va nguon von
        self.reserves: float = 0.0
        self.total_deposits: float = 0.0
        self.total_loans: float = 0.0
        self.non_performing_loans: float = 0.0

        # NO CUU TRO KHAN CAP (bailout debt) -- Bagehot, W. (1873), "Lombard
        # Street: A Description of the Money Market", Henry S. King & Co.
        # -- hoc thuyet kinh dien "lend freely, at a high rate, against good
        # collateral": nguoi cho vay cuoi cung (Kho bac, xem env.py::step()
        # nhanh "Lender of Last Resort") KHONG duoc cuu tro MIEN PHI, neu
        # khong se trung hoa hoan toan dong co quan tri rui ro cua Bank (moral
        # hazard) -- BAN VA DAU TIEN (v0.20) tai cap von thang, KHONG co lai
        # phat, bi phat hien la trich dan sai tinh than Bagehot (chi ap dung
        # nua ve "lend freely", bo qua nua ve "at a high rate"). Sua: khoan
        # cuu tro duoc ghi nhan la MOT KHOAN NO thuc su Bank phai tra Kho bac
        # kem lai suat PHAT (xem bailout_penalty_rate, Section 5B rule_engine.py).
        self.bailout_debt: float = 0.0
        # HE SO CAU TRUC TU DO HIEU CHINH: chon RO RANG cao hon tran
        # lending_rate hop le [0.01, 0.25]/nam (bien do action space Bank o
        # rllib_wrapper.py) de dung tinh than "high rate" cua Bagehot (lai
        # phat phai cao hon lai suat thi truong thong thuong, khong the vay
        # duoc muc nay tu bat ky nguon nao khac) -- khong suy ra ty le cu the
        # tu chinh Bagehot (1873).
        self.bailout_penalty_rate: float = 0.40

        # Chinh sach lai suat va an toan von.
        # QUY UOC: lending_rate/deposit_rate la LAI SUAT NAM (annual, simple/
        # linear), KHONG PHAI lai suat thang. rule_engine.py luon chia cho 12.0
        # truoc khi ap dung vao tung buoc thang (vd. Section 5:
        # "monthly_interest = firm.debt * (lending_rate / 12.0)"; Section 8B:
        # "gross_deposit_interest = old_deposit * (deposit_rate / 12.0)") --
        # KHONG duoc chia cho 12 mot lan nua o bat ky noi nao khac dung
        # lending_rate/deposit_rate, se lam lai suat thuc te nho hon 12 lan gia
        # tri du dinh. Voi bien action space [0.01, 0.25] (lending) va
        # [0.005, 0.15] (deposit) o rllib_wrapper.py, day da la khoang lai suat
        # nam hop ly cho thi truong tin dung emerging market (1%-25%/nam,
        # 0.5%-15%/nam) -- KHONG can quy doi them.
        self.lending_rate: float = 0.06
        self.deposit_rate: float = 0.02
        self.reserve_requirement_ratio: float = 0.10
        self.credit_expansion_factor: float = 1.0
        
        # Ket qua hoat dong ky truoc
        self.last_interest_income: float = 0.0
        self.last_interest_expense: float = 0.0
        self.last_default_loss: float = 0.0

    def initialize(self, 
                   initial_reserves: float = 500000.0,
                   initial_lending_rate: float = 0.06,
                   initial_deposit_rate: float = 0.02,
                   reserve_requirement_ratio: float = 0.10) -> None:
        self.reserves = float(initial_reserves)
        self.total_deposits = 0.0
        self.total_loans = 0.0
        self.non_performing_loans = 0.0
        self.npl_vintages = []
        self.bailout_debt = 0.0
        self.lending_rate = float(initial_lending_rate)
        self.deposit_rate = float(initial_deposit_rate)
        self.reserve_requirement_ratio = float(reserve_requirement_ratio)
        self.credit_expansion_factor = 1.0
        self.last_interest_income = 0.0
        self.last_interest_expense = 0.0
        self.last_default_loss = 0.0
        self.status = LifeCycleStatus.ACTIVE

    def observe(self, raw_environment_state: Dict[str, Any]) -> Observation:
        """
        Khong gian quan sat 10 chieu:
        [0]: Luong tien mat du tru tai ngan hang (Reserves scaled)
        [1]: Tong quy tien gui cua toan bo nen kinh te (Deposits scaled)
        [2]: Tong du no tin dung dang luu hanh (Loans scaled)
        [3]: Khoi luong no xau chua xu ly (NPL scaled)
        [4]: Ty le no xau tren tong du no (NPL Ratio)
        [5]: Ty le du tru thuc te tren tong tien gui (Current Reserve Ratio)
        [6]: Lai suat cho vay hien tai
        [7]: Lai suat tien gui hien tai
        [8]: Ty le lam phat thi truong
        [9]: Cau vay von cua Doanh nghiep toan thi truong
        """
        macro = raw_environment_state.get("macro_indicators", {})
        inflation = float(macro.get("inflation", 0.0))
        credit_demand = float(macro.get("total_credit_demand", 0.0)) * 0.001

        # LOI DA SUA (v0.26, phat hien qua audit chu dong theo yeu cau nguoi dung): truoc day
        # dung truc tiep self.non_performing_loans/self.total_loans -- DUNG CHINH cong thuc
        # da duoc tai lieu hoa la LOI o dau file nay (compute_npl_ratio_pct docstring: mau so
        # total_loans don thuan co the gan 0 trong khi non_performing_loans van con "mo" trong
        # cua so write-off, cho ra ty le >100%). Ham compute_npl_ratio_pct() da SUA dung (mau
        # so = gross_loans = total_loans + npl) nhung TRUOC DAY chi duoc goi o env.py/logger.py/
        # rllib_wrapper.py de HIEN THI -- rieng observe() (anh huong truc tiep obs space RLlib
        # nhin thay) van dung cong thuc CU chua sua, chi duoc "che" trieu chung bang np.clip(0,1)
        # ben duoi (bao hoa ve 1.0 dung luc quan trong nhat -- Bank khong con phan biet duoc
        # "NPL gap doi du no" voi "NPL gap 100 lan du no", ca hai deu clip ve 1.0). Sua: dung lai
        # dung mot ham compute_npl_ratio_pct() da co san, chia 100 de ve thang [0,1].
        npl_ratio = compute_npl_ratio_pct(self.non_performing_loans, self.total_loans) / 100.0
        reserve_ratio = (self.reserves / self.total_deposits) if self.total_deposits > 0.0 else 1.0

        obs_array = np.array([
            self.reserves * 0.0001,
            self.total_deposits * 0.0001,
            self.total_loans * 0.0001,
            self.non_performing_loans * 0.0001,
            float(np.clip(npl_ratio, 0.0, 1.0)),
            float(np.clip(reserve_ratio, 0.0, 2.0)),
            self.lending_rate,
            self.deposit_rate,
            inflation,
            credit_demand
        ], dtype=np.float32)

        return Observation(
            agent_id=self.agent_id,
            timestep=raw_environment_state.get("timestep", 0),
            vector=obs_array,
            metadata={"npl_ratio": npl_ratio, "reserves": self.reserves}
        )

    def decide(self, observation: Observation) -> Action:
        """
        Khong gian hanh dong 3 chieu:
        [0]: Muc tieu Lai suat cho vay (Lending Rate): [0.01, 0.25]
        [1]: Muc tieu Lai suat huy dong tien gui (Deposit Rate): [0.005, 0.15]
        [2]: He so mo rong / that chat tin dung (Credit Expansion Factor): [0.0, 1.0]
        """
        if "injected_action" in observation.metadata:
            raw_action = observation.metadata["injected_action"]
        else:
            # Chinh sach tien te on dinh mac dinh
            raw_action = np.array([0.06, 0.02, 0.8], dtype=np.float32)

        return Action(
            agent_id=self.agent_id,
            action_type="MONETARY_POLICY_AND_CREDIT",
            values=np.asarray(raw_action, dtype=np.float32)
        )

    def validate_action(self, action: Action) -> ValidationResult:
        vals = action.values
        if len(vals) < 3:
            return ValidationResult(
                is_valid=False,
                sanitized_values=np.array([0.06, 0.02, 0.5], dtype=np.float32),
                reason="Bank action requires 3 elements"
            )

        lending_rate = float(np.clip(vals[0], 0.01, 0.25))
        deposit_rate = float(np.clip(vals[1], 0.005, 0.15))
        
        # Nguyen tac song con cua Ngan hang: Lai suat cho vay luon phai cao hon Lai suat huy dong
        if lending_rate < deposit_rate + 0.01:
            lending_rate = deposit_rate + 0.01

        credit_factor = float(np.clip(vals[2], 0.0, 1.0))

        # Neu ty le du tru duoi nguong an toan, dong bang viec bom them tin dung
        required_cash = self.total_deposits * self.reserve_requirement_ratio
        if self.reserves < required_cash:
            credit_factor = 0.0

        sanitized = np.array([lending_rate, deposit_rate, credit_factor], dtype=np.float32)
        return ValidationResult(is_valid=True, sanitized_values=sanitized)

    def apply_result(self, transition_result: TransitionResult) -> None:
        delta = transition_result.state_delta
        
        # Bien dong luong tien gui va giai ngan tin dung thuc te do RuleEngine xac thuc
        self.total_deposits = float(max(0.0, self.total_deposits + delta.get("deposits_delta", 0.0)))
        self.total_loans = float(max(0.0, self.total_loans + delta.get("loans_delta", 0.0)))
        self.reserves += float(delta.get("reserves_delta", 0.0))
        
        # Xử lý nợ xấu từ các Doanh nghiệp phá sản trong kỳ. "new_defaults" ở đây
        # là phần KHÔNG THU HỒI ĐƯỢC (bad_debt) sau khi RuleEngine đã trừ đi phần
        # thu hồi bằng tiền mặt + thanh lý tài sản thế chấp (Merton, 1974) -- việc
        # xoá khoản vay đã tất toán khỏi tổng dư nợ (total_loans) được xử lý qua
        # kênh loans_delta chung ở trên (bao gồm cả phần thu hồi được lẫn phần mất
        # trắng), nên KHÔNG được trừ trùng total_loans ở đây lần nữa.
        new_defaults = float(delta.get("new_defaults", 0.0))
        current_month = int(delta.get("current_month", 0))

        if new_defaults > 0.0:
            self.npl_vintages.append((new_defaults, current_month))

        # GHI SỔ & KHÉP SỔ NỢ XẤU THEO THỜI GIAN (Write-off).
        #
        # Trước bản vá này, non_performing_loans là bộ đếm CHỈ TĂNG trong suốt
        # một episode -- không ngân hàng thật nào giữ nguyên bad debt trên sổ
        # sách vĩnh viễn. Tham chiếu: IFRS 9 (International Accounting
        # Standards Board, 2014), "IFRS 9 Financial Instruments" -- khái niệm
        # phân loại nợ "credit-impaired" (Stage 3) theo thời gian quá hạn; và
        # Basel Committee on Banking Supervision (2017), "Prudential treatment
        # of problem assets -- definitions of non-performing exposures and
        # forbearance" -- thông lệ ngân hàng thực tế luôn có một mốc thời gian
        # để chính thức WRITE OFF (khép sổ theo dõi) một khoản nợ xấu, dù thời
        # hạn cụ thể khác nhau theo ngân hàng/quốc gia. npl_writeoff_months=6
        # (mặc định) là HỆ SỐ CẤU TRÚC TỰ DO HIỆU CHỈNH -- chọn ở cận dưới
        # khoảng thực tế để tạo áp lực write-off rõ ràng trong phạm vi 1
        # episode (240 bước).
        writeoff_cutoff_month = current_month - self.npl_writeoff_months
        total_writeoff = sum(amt for amt, born in self.npl_vintages if born <= writeoff_cutoff_month)
        self.npl_vintages = [(amt, born) for amt, born in self.npl_vintages if born > writeoff_cutoff_month]

        self.non_performing_loans = float(max(0.0, self.non_performing_loans + new_defaults - total_writeoff))
        self.last_default_loss = new_defaults

        # QUAN TRỌNG: write-off CHỈ trừ khỏi non_performing_loans (metric theo
        # dõi/đầu vào reward) -- KHÔNG trừ reserves thêm lần nữa. Tổn thất tín
        # dụng THẬT SỰ đã được ghi nhận DUY NHẤT MỘT LẦN ngay tại thời điểm vỡ
        # nợ (dòng dưới đây), đúng tinh thần kế toán ngân hàng thật: một khoản
        # nợ đã trích lập dự phòng đầy đủ (đã trừ reserves) thì write-off chỉ
        # là thao tác khép sổ theo dõi, không tạo thêm tổn thất P&L. Trừ
        # reserves thêm lần nữa ở đây sẽ ghi nhận TRÙNG cùng một khoản lỗ, vi
        # phạm bảo toàn SFC (đã chạy lại be/tests/test_sfc_accounting.py để
        # xác nhận thiết kế này không phá vỡ đẳng thức bảo toàn).
        if new_defaults > 0.0:
            self.reserves -= new_defaults  # ngân hàng chịu lỗ tín dụng thực sự

        self.last_interest_income = float(delta.get("interest_income", 0.0))
        self.last_interest_expense = float(delta.get("interest_expense", 0.0))

        # No cuu tro (bailout_debt) -- xem chu thich day du tai __init__ va
        # rule_engine.py Section 5B. "bailout_debt_delta" duong khi vua duoc
        # cuu tro (env.py::step()), am khi tra bot goc (rule_engine.py Section
        # 5B, cung nhip voi Firm Section 5).
        self.bailout_debt = float(max(0.0, self.bailout_debt + delta.get("bailout_debt_delta", 0.0)))

        # Cap nhat tham so thuc thi
        self.lending_rate = float(delta.get("executed_lending_rate", self.lending_rate))
        self.deposit_rate = float(delta.get("executed_deposit_rate", self.deposit_rate))
        self.credit_expansion_factor = float(delta.get("executed_credit_factor", self.credit_expansion_factor))

        # Ranh gioi pha san ngan hang (Mat kha nang thanh toan tram trong)
        if self.reserves < -100000.0:
            self.terminate(reason="Bank run / Total liquidity failure")

    def calculate_reward(self, transition_result: TransitionResult) -> float:
        """
        Hàm mục tiêu tài chính của Ngân hàng:
        Reward = Biên lãi ròng (NIM) - Phạt Nợ xấu MỚI (dòng) - Phạt TỒN KHO
        Nợ xấu (tỷ lệ NPL/Tổng dư nợ) - Phạt Vi phạm Dự trữ bắt buộc

        Ghi chú hiệu chỉnh (quan sát thực nghiệm qua nhiều lần train dài
        hạn): trước đây chỉ phạt theo DÒNG (last_default_loss, phát sinh MỘT
        LẦN đúng tháng vỡ nợ) trong khi lãi vay của MỘT khoản vay được cộng
        dồn NHIỀU THÁNG liên tục (net_interest_margin) -- tạo bất đối xứng
        khuyến khích cho vay rủi ro cao (kỳ vọng lãi nhiều tháng > kỳ vọng lỗ
        một lần). Hệ quả quan sát được: NPL tăng tuyến tính không có dấu hiệu
        bão hoà qua hàng chục iteration training. Hai thay đổi (hệ số cấu
        trúc TỰ DO HIỆU CHỈNH, không phải công thức mới):
          1. Tăng hệ số phạt dòng 0.02 -> 0.06.
          2. Bổ sung phạt theo TỶ LỆ tồn kho NPL/tổng dư nợ mỗi bước, để
             ngân hàng chịu áp lực liên tục giảm nợ xấu tồn đọng chứ không
             chỉ tránh tạo thêm nợ xấu mới trong đúng tháng đó.
        """
        net_interest_margin = (self.last_interest_income - self.last_interest_expense) * 0.01
        npl_flow_penalty = self.last_default_loss * self.npl_flow_penalty_coef

        # LOI DA SUA (v0.26): cung loi nhu observe() o tren -- cong thuc CU
        # "non_performing_loans / max(total_loans, 1.0)" la DUNG CHINH cong thuc da duoc tai
        # lieu hoa la loi o docstring compute_npl_ratio_pct() dau file (co the vuot 100% khi
        # total_loans gan 0 nhung con NPL "mo" trong cua so write-off) -- truoc day chi duoc
        # sua cho ham hien thi, KHONG duoc ap dung o day noi no truc tiep anh huong gradient
        # PPO qua npl_stock_penalty. Kich ban de kich hoat nhat: firm pha san hang loat lam
        # total_loans sup nhanh hon toc do write-off 6 thang cua NPL (npl_writeoff_months).
        npl_ratio = compute_npl_ratio_pct(self.non_performing_loans, self.total_loans) / 100.0
        npl_stock_penalty = npl_ratio * self.npl_stock_penalty_coef

        # Phat neu du tru thuc te thap hon ty le bat buoc
        reserve_penalty = 0.0
        required_reserves = self.total_deposits * self.reserve_requirement_ratio
        if self.reserves < required_reserves:
            deficit = required_reserves - self.reserves
            reserve_penalty = deficit * 0.05

        reward = net_interest_margin - npl_flow_penalty - npl_stock_penalty - reserve_penalty
        return float(np.clip(reward, -100.0, 100.0))

    def export_state(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "status": self.status.name,
            "reserves": self.reserves,
            "total_deposits": self.total_deposits,
            "total_loans": self.total_loans,
            "non_performing_loans": self.non_performing_loans,
            "lending_rate": self.lending_rate,
            "deposit_rate": self.deposit_rate,
            "last_nim": self.last_interest_income - self.last_interest_expense,
            "bailout_debt": round(self.bailout_debt, 1)
        }

    def reset(self) -> None:
        super().reset()
        self.reserves = 0.0
        self.total_deposits = 0.0
        self.total_loans = 0.0
        self.bailout_debt = 0.0
        self.non_performing_loans = 0.0
        self.npl_vintages = []
        self.lending_rate = 0.06
        self.deposit_rate = 0.02
        self.credit_expansion_factor = 1.0
        self.last_interest_income = 0.0
        self.last_interest_expense = 0.0
        self.last_default_loss = 0.0

    def terminate(self, reason: str = "") -> None:
        super().terminate(reason)