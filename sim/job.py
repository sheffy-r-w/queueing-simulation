"""
job.py — Job dataclass for the queueing simulation.

A Job represents a single customer/task in the M/G/1 queue.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    """
    Represents a single job in the M/G/1 queue.

    Attributes set at arrival:
        job_id:         Unique integer identifier.
        arrival_time:   Time the job arrived to the system.
        size:           True service requirement (drawn from job size distribution).

    Attributes updated during simulation:
        attained:       Amount of service received so far (age).
        start_time:     Time the job first entered service (for tracking).
        departure_time: Time the job left the system (set on completion).

    Derived quantities (computed as properties):
        response_time:  departure_time - arrival_time  (valid after completion)
        sojourn_time:   alias for response_time
        remaining:      size - attained
    """

    job_id: int
    arrival_time: float
    size: float

    # Mutable state updated during simulation
    attained: float = 0.0
    start_time: Optional[float] = None
    departure_time: Optional[float] = None

    @property
    def remaining(self) -> float:
        """Remaining service requirement."""
        return self.size - self.attained

    @property
    def response_time(self) -> Optional[float]:
        """Total time in system (None if job not yet complete)."""
        if self.departure_time is None:
            return None
        return self.departure_time - self.arrival_time

    # Alias
    sojourn_time = response_time

    def __repr__(self) -> str:
        return (
            f"Job(id={self.job_id}, arr={self.arrival_time:.4f}, "
            f"size={self.size:.4f}, attained={self.attained:.4f})"
        )