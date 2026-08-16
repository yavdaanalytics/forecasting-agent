from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


Z_BY_SERVICE_LEVEL = {
    0.90: 1.2815515655446004,
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}


@dataclass(frozen=True)
class SafetyStockResult:
    sku: str
    method: str
    lead_time_days: int
    service_level: float
    n_origins: int
    error_std: float | None
    error_bias: float | None
    z: float
    safety_stock: float | None
    by_service_level: Mapping[str, float] = field(default_factory=dict)
    reason: str = "ok"
    escalate: bool = False

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "method": self.method,
            "lead_time_days": self.lead_time_days,
            "service_level": self.service_level,
            "n_origins": self.n_origins,
            "error_std": self.error_std,
            "error_bias": self.error_bias,
            "z": self.z,
            "safety_stock": self.safety_stock,
            "by_service_level": dict(self.by_service_level),
            "reason": self.reason,
            "escalate": self.escalate,
        }
