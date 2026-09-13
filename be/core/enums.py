from enum import Enum, auto

class AgentType(Enum):
    GOVERNMENT = "government"
    ECONOMY = "economy"
    BANK = "bank"
    FIRM = "firm"
    EMPLOYEE = "employee"
    SUPERVISOR = "supervisor"

class LifeCycleStatus(Enum):
    INITIALIZED = auto()
    ACTIVE = auto()
    BANKRUPT = auto()
    DECEASED = auto()
    DEAD = DECEASED       # Alias dong bo tuyet doi voi rule_engine
    TERMINATED = auto()

class EventType(Enum):
    HIRE = "HIRE"
    FIRE = "FIRE"
    WAGE_PAID = "WAGE_PAID"
    GOODS_PURCHASED = "GOODS_PURCHASED"
    TAX_COLLECTED = "TAX_COLLECTED"
    TAX_EVADED = "TAX_EVADED"
    LOAN_DISBURSED = "LOAN_DISBURSED"
    LOAN_REPAID = "LOAN_REPAID"
    DEFAULT_OCCURRED = "DEFAULT_OCCURRED"
    AUDIT_CONDUCTED = "AUDIT_CONDUCTED"
    PENALTY_ENFORCED = "PENALTY_ENFORCED"
    PRICE_ADJUSTED = "PRICE_ADJUSTED"
    AGENT_DIED = "AGENT_DIED"
    AGENT_BANKRUPT = "AGENT_BANKRUPT"
    AGENT_BORN = "AGENT_BORN"          # Bo sung cho co che Demographic Turnover
    STATE_UPDATE = "STATE_UPDATE"