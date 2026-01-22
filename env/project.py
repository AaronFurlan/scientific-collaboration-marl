from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .area import Area


class Project:
    """Represents a scientific project in the simulation."""

    def __init__(
        self,
        project_id: str,
        required_effort: int,
        prestige: float,
        time_window: int,
        peer_fit: np.ndarray,
        **kwargs,
    ):
        self.project_id = project_id
        self.required_effort = required_effort
        self.prestige = prestige
        self.time_window = time_window
        self.peer_fit = peer_fit
        # Project state
        self.current_effort = 0
        self.contributors: List[int] = []
        self.start_time = 0
        self.finished = False
        self.final_reward = None

        # Additional attributes
        self.kene: Optional[np.array] = kwargs.get("kene", None)
        self.citations: List[str] = kwargs.get("citations", [])
        self.cited_by: List[str] = kwargs.get("cited_by", [])
        self.generator_project_id: Optional[str] = kwargs.get(
            "generator_project_id", None
        )
        self.novelty: float = kwargs.get("novelty", 0.5)

        # Validation and quality metrics
        self.validation_noise = 0.0
        self.quality_score = 0.0
        self.effort_score = 0.0
        self.novelty_score = 0.0
        self.societal_value_score = 0.0

    @staticmethod
    def seed(seed: int):
        if seed is not None:
            np.random.seed(seed)

    def add_contributor(self, agent_id: int) -> bool:
        """Add an agent as a contributor to the project."""
        if agent_id not in self.contributors:
            self.contributors.append(agent_id)
            return True
        return False

    def add_effort(self, effort: float) -> None:
        """Add effort to the project."""
        self.current_effort += effort

    def get_completion_progress(self) -> float:
        """Get the completion progress as a percentage."""
        return min(1.0, self.current_effort / self.required_effort)

    def get_remaining_effort(self) -> float:
        """Get the remaining effort needed to complete the project."""
        return max(0, self.required_effort - self.current_effort)

    def get_time_remaining(self, current_timestep: int) -> int:
        """Get the remaining time before the project deadline."""
        return max(0, self.time_window - (current_timestep - self.start_time))

    def is_due(self, current_timestep: int) -> bool:
        """Check if the project is overdue."""
        return current_timestep >= (self.start_time + self.time_window)

    def calculate_quality(
        self, topic_area: Area, relative_density: int, noise_factor: float = 0.5
    ) -> float:
        """Calculate the final quality of the completed project."""

        # the higher the prestige the samller the noise
        self.validation_noise = np.random.normal(1, noise_factor * (1 - self.prestige))
        # Base quality based on effort and prestige
        self.effort_score = self.calculate_effort_score(self.validation_noise)
        self.novelty_score = self.calculate_novelty_score(relative_density)
        self.societal_value_score = self.calculate_societal_value_score(topic_area)
        self.quality_score = (
            1 / 3 * self.effort_score
            + 1 / 3 * self.novelty_score
            + 1 / 3 * self.societal_value_score
        )
        # Add validation noise
        return self.quality_score

    def calculate_effort_score(self, noise=1.0):
        return (
            1
            - (max(0, self.required_effort - self.current_effort) * noise)
            / self.required_effort
        )

    def calculate_novelty_score(self, relative_density):
        return relative_density

    def calculate_societal_value_score(self, topic_area: Area):
        return topic_area.value_at(*self.kene)

    def calculate_reward(self, quality, threshold=0.5, noise_factor=0.15) -> float:
        if quality > threshold * self.prestige:
            self.final_reward = self.prestige + np.random.normal(0, noise_factor)
        else:
            self.final_reward = 0
        return self.final_reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert the project to a dictionary representation."""
        return {
            "project_id": self.project_id,
            "required_effort": self.required_effort,
            "prestige": self.prestige,
            "time_window": self.time_window,
            "peer_fit": (
                self.peer_fit.tolist()
                if isinstance(self.peer_fit, np.ndarray)
                else self.peer_fit
            ),
            "novelty": self.novelty,
            "current_effort": self.current_effort,
            "contributors": self.contributors.copy(),
            "start_time": self.start_time,
            "finished": self.finished,
            "final_reward": self.final_reward,
            "kene": self.kene,
            "citations": self.citations.copy(),
            "cited_by": self.cited_by.copy(),
            "generator_project_id": self.generator_project_id,
            "validation_noise": self.validation_noise,
            "quality_score": self.quality_score,
            "novelty_score": self.novelty_score,
            "societal_value_score": self.societal_value_score,
            "effort_score": self.effort_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create a Project instance from a dictionary."""
        project = cls(
            project_id=data["id"],
            required_effort=data["required_effort"],
            prestige=data["prestige"],
            time_window=data["time_window"],
            peer_fit=(
                np.array(data["peer_fit"])
                if isinstance(data["peer_fit"], list)
                else data["peer_fit"]
            ),
        )

        # Restore state
        project.current_effort = data.get("current_effort", 0)
        project.contributors = data.get("contributors", [])
        project.start_time = data.get("start_time", 0)
        project.finished = data.get("finished", False)
        project.final_reward = data.get("final_reward", 0.0)
        project.kene = data.get("kene")
        project.citations = data.get("citations", [])
        project.cited_by = data.get("cited_by", [])
        project.generator_project_id = data.get("generator_project_id")
        project.validation_noise = data.get("validation_noise", 0.0)
        project.quality_score = data.get("quality_score", 0.0)
        project.novelty = data.get("novelty", 0.0)
        project.effort_score = data.get("effort_score", 0.0)
        project.novelty_score = data.get("novelty_score", 0.0)
        project.societal_value_score = data.get("societal_value_score", 0.0)

        return project

    def as_observation_dict(self) -> Dict:
        return {
            "required_effort": np.array(
                [self.required_effort], dtype=np.int32
            ),  # Box(0, 200, (1,), dtype=np.int32),
            "prestige": np.array(
                [self.prestige], dtype=np.float32
            ),  # Box(0, 1, (1,), dtype=np.float32),
            "novelty": np.array(
                [self.novelty], dtype=np.float32
            ),  # Box(0, 1, (1,), dtype=np.float32),
            "peer_fit": (
                self.peer_fit.tolist()
                if isinstance(self.peer_fit, np.ndarray)
                else self.peer_fit
            ),
            "contributors": self.contributors.copy(),
            "current_effort": np.array([self.current_effort], dtype=np.float32),
            "start_time": self.start_time,
            "time_window": self.time_window,
        }

    def __str__(self) -> str:
        return f"Project({self.project_id}, effort: {self.current_effort}/{self.required_effort}, contributors: {len(self.contributors)})"

    def __repr__(self) -> str:
        return self.__str__()
