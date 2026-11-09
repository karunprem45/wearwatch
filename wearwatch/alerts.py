"""Turn noisy per-reading failure probabilities into stable maintenance alerts.

Alerting on a single high reading floods the maintenance team with false alarms. This
tracker raises an alert only after k high readings in a row on the same machine, and
clears it only after k low readings in a row (hysteresis), so a flickering signal does
not toggle the alert on and off.
"""
from dataclasses import dataclass, field


@dataclass
class MachineState:
    high_streak: int = 0
    low_streak: int = 0
    alerting: bool = False


@dataclass
class AlertTracker:
    threshold: float
    k: int = 3
    machines: dict = field(default_factory=dict)
    raised: int = 0
    cleared: int = 0

    def update(self, machine_id, prob):
        """Feed one reading. Returns 'raise', 'clear' or None."""
        s = self.machines.setdefault(machine_id, MachineState())
        if prob >= self.threshold:
            s.high_streak += 1
            s.low_streak = 0
            if not s.alerting and s.high_streak >= self.k:
                s.alerting = True
                self.raised += 1
                return "raise"
        else:
            s.low_streak += 1
            s.high_streak = 0
            if s.alerting and s.low_streak >= self.k:
                s.alerting = False
                self.cleared += 1
                return "clear"
        return None

    def active(self):
        return sorted(m for m, s in self.machines.items() if s.alerting)
