from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class SimulationStats:
    """Tracks aggregate statistics across simulation steps.

    This helper collects totals and averages such as:
    - Total/new effort applied (overall and per agent)
    - Number of finished projects and averages over finished projects
    - Rewards distributed totals
    - Aggregations over observation values (ages, reputations, etc.)

    Usage:
        stats = SimulationStats()
        ...
        obs, rewards, terminations, truncations, infos = env.step(actions)
        stats.update(env, obs, rewards, terminations, truncations)
        print(stats.to_dict())
    """

    def __init__(self) -> None:
        # Time tracking
        self.total_steps: int = 0

        # Effort tracking
        self.total_effort_applied: float = 0.0
        self.total_effort_per_agent: Dict[str, float] = {}
        self._prev_agent_project_effort: Optional[List[Dict[Any, float]]] = None

        # Project completion tracking
        self.finished_projects_seen: Set[Any] = set()
        self.finished_projects_count: int = 0
        self.finished_projects_efforts: List[float] = []
        self.finished_projects_durations: List[float] = []
        self.finished_projects_team_sizes: List[int] = []

        # Success tracking
        self.successful_projects_count: int = 0
        self.unsuccessful_projects_count: int = 0
        self.successful_projects_efforts: List[float] = []
        self.unsuccessful_projects_efforts: List[float] = []

        # Individual project metrics for detailed logging
        self.finished_project_details: List[Dict[str, Any]] = []

        # Reward tracking
        self.total_rewards_distributed: float = 0.0

        # Observation aggregations (last snapshot)
        self.last_obs_aggregate: Dict[str, float] = {}

        self.peer_group_sizes: List[List[int]] = []  # Track peer group sizes over time
        self.success_rates_over_time: List[float] = []  # Track success rates over time

        # Enhanced metrics for visualizations
        self.papers_per_author: Dict[str, int] = {}  # agent_name -> paper_count
        self.authors_per_paper: List[int] = []  # team sizes for all papers
        self.prestige_category_stats: Dict[int, Dict[str, int]] = (
            {}
        )  # category -> {success: count, total: count}
        self.effort_ratios: List[float] = []  # actual_effort / required_effort ratios

    def reset(self) -> None:
        self.__init__()

    def update(
        self,
        env: Any,
        obs: Dict[str, Any],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
    ) -> None:
        """Update statistics based on the latest step outputs and env state."""
        self.total_steps += 1

        # Ensure per-agent effort dict initialized
        for agent in getattr(env, "agents", []):
            if agent not in self.total_effort_per_agent:
                self.total_effort_per_agent[agent] = 0.0
            if agent not in self.papers_per_author:
                self.papers_per_author[agent] = 0

        # 1) New effort applied this step (delta from last snapshot)
        # env.agent_project_effort is a list indexed by agent id: Dict[project_id, cumulative_effort]
        current_effort_snapshot: List[Dict[Any, float]] = copy.deepcopy(
            getattr(env, "agent_project_effort", [])
        )

        if self._prev_agent_project_effort is None:
            # First step: take snapshot; deltas are the absolute values (from 0)
            deltas_by_agent: List[float] = []
            for agent_idx, proj_map in enumerate(current_effort_snapshot):
                delta_sum = float(0.0)
                for _, cumulative_effort in proj_map.items():
                    delta_sum += _as_float(cumulative_effort)
                deltas_by_agent.append(delta_sum)
        else:
            deltas_by_agent = []
            for agent_idx, proj_map in enumerate(current_effort_snapshot):
                prev_map = (
                    self._prev_agent_project_effort[agent_idx]
                    if agent_idx < len(self._prev_agent_project_effort)
                    else {}
                )
                delta_sum = float(0.0)
                for proj_id, cumulative_effort in proj_map.items():
                    prev = _as_float(prev_map.get(proj_id, 0.0))
                    curr = _as_float(cumulative_effort)
                    delta = max(0.0, curr - prev)
                    delta_sum += delta
                deltas_by_agent.append(delta_sum)

        # Accumulate totals
        for agent_name in getattr(env, "agents", []):
            agent_idx = env.agent_to_id[agent_name]
            agent_delta = float(
                deltas_by_agent[agent_idx] if agent_idx < len(deltas_by_agent) else 0.0
            )
            self.total_effort_per_agent[agent_name] += agent_delta
            self.total_effort_applied += agent_delta

        # Update snapshot
        self._prev_agent_project_effort = current_effort_snapshot

        if hasattr(env, "peer_groups"):
            current_sizes = [len(group) for group in env.peer_groups]
            self.peer_group_sizes.append(current_sizes)

        # 2) Track newly finished projects and record their stats once
        for proj_id, proj in getattr(env, "projects", {}).items():
            if proj.finished and proj.project_id not in self.finished_projects_seen:
                self.finished_projects_seen.add(proj.project_id)
                self.finished_projects_count += 1
                proj = proj.to_dict()
                project_effort = _as_float(proj.get("current_effort", 0.0))
                self.finished_projects_efforts.append(project_effort)
                self.finished_projects_durations.append(
                    float(proj.get("time_window", 0))
                )
                self.finished_projects_team_sizes.append(
                    int(len(proj.get("contributors", [])))
                )

                # Track success vs failure based on rewards
                project_reward = _as_float(proj.get("final_reward", 0.0))
                if project_reward > 0:
                    self.successful_projects_count += 1
                    self.successful_projects_efforts.append(project_effort)
                else:
                    self.unsuccessful_projects_count += 1
                    self.unsuccessful_projects_efforts.append(project_effort)

                # Enhanced metrics tracking
                # self._track_enhanced_metrics(env, proj, proj_id, project_reward > 0)

                # Log detailed project metrics
                # self._log_finished_project(env, proj, proj_id)

        # 3) Rewards distributed this step (sum across agents)
        if rewards:
            self.total_rewards_distributed += float(
                sum(_as_float(v) for v in rewards.values())
            )

        # 4) Aggregate observation values (ages, reputations, running projects, open opportunities)
        self.last_obs_aggregate = self._compute_observation_aggregates(env, obs)

    def _track_enhanced_metrics(
        self, env: Any, proj: Dict[str, Any], proj_id: str, is_successful: bool
    ) -> None:
        """Track enhanced metrics for visualizations."""
        # Track authors per paper
        team_size = len(proj.get("contributors", []))
        self.authors_per_paper.append(team_size)

        # Track effort ratio (actual / required)
        required_effort = _as_float(proj.get("required_effort", 1.0))
        actual_effort = _as_float(proj.get("current_effort", 0.0))
        if required_effort > 0:
            effort_ratio = actual_effort / required_effort
            self.effort_ratios.append(effort_ratio)

        # Track prestige category stats
        # Extract project template index from project ID (format: "project_{len}-{template_idx}-{timestep}")
        try:
            parts = proj_id.split("-")
            if len(parts) >= 3:
                template_idx = int(parts[-2])  # Second to last part
                prestige_category = template_idx % 3  # 0: low, 1: medium, 2: high

                if prestige_category not in self.prestige_category_stats:
                    self.prestige_category_stats[prestige_category] = {
                        "success": 0,
                        "total": 0,
                    }

                self.prestige_category_stats[prestige_category]["total"] += 1
                if is_successful:
                    self.prestige_category_stats[prestige_category]["success"] += 1
        except (ValueError, IndexError):
            pass  # Skip if project ID format is unexpected

        # Track papers per author
        contributors = proj.get("contributors", [])
        for agent_idx in contributors:
            agent_name = f"agent_{agent_idx}"
            if agent_name in self.papers_per_author:
                self.papers_per_author[agent_name] += 1

    def _compute_observation_aggregates(
        self, env: Any, obs: Dict[str, Any]
    ) -> Dict[str, float]:
        ages: List[float] = []
        acc_rewards: List[float] = []
        peer_reps: List[float] = []
        running_projects_counts: List[int] = []
        project_team_sizes: Dict[str, int] = {}

        for agent in getattr(env, "agents", []):
            agent_obs_container = obs.get(agent, {})
            agent_obs = agent_obs_container.get("observation", agent_obs_container)

            age_arr = agent_obs.get("age")
            if age_arr is not None:
                ages.append(_as_float(_first(age_arr, 0.0)))

            acc_arr = agent_obs.get("accumulated_rewards")
            if acc_arr is not None:
                acc_rewards.append(_as_float(_first(acc_arr, 0.0)))

            peer_rep_arr = agent_obs.get("peer_reputation")
            if peer_rep_arr is not None:
                peer_reps.extend(
                    [_as_float(x) for x in list(np.array(peer_rep_arr).flatten())]
                )

            running = agent_obs.get("running_projects", {}) or {}
            running_projects_counts.append(len(running))

        # Open opportunities from env (single shared list)
        open_projects = getattr(env, "open_projects", [])
        open_required_efforts = []
        open_approx_rewards = []
        open_time_windows = []
        for p in open_projects:
            if isinstance(p, dict):
                if "required_effort" in p:
                    open_required_efforts.append(
                        _as_float(p.get("required_effort", 0.0))
                    )
                if "approx_reward" in p:
                    open_approx_rewards.append(_as_float(p.get("approx_reward", 0.0)))
                if "time_window" in p:
                    open_time_windows.append(_as_float(p.get("time_window", 0.0)))

        # Team sizes for each running or finished project (by id)
        for proj in getattr(env, "projects", {}).values():
            team_size = int(len(proj.to_dict().get("contributors", [])))
            project_team_sizes[proj.project_id] = team_size

        aggregates: Dict[str, float] = {
            "avg_age": _safe_mean(ages),
            "avg_accumulated_rewards": _safe_mean(acc_rewards),
            "avg_peer_reputation": _safe_mean(peer_reps),
            "avg_running_projects_per_agent": _safe_mean(
                [float(x) for x in running_projects_counts]
            ),
            "open_projects_count": float(len(open_projects)),
            "avg_open_required_effort": _safe_mean(open_required_efforts),
            "avg_open_approx_reward": _safe_mean(open_approx_rewards),
            "avg_open_time_window": _safe_mean(open_time_windows),
            "active_projects_count": float(
                sum(1 for p in getattr(env, "projects", {}).values() if not p.finished)
            ),
        }
        # Attach nested dict (kept separate from numeric fields)
        aggregates["project_team_sizes"] = project_team_sizes
        return aggregates

    def _log_finished_project(
        self, env: Any, proj: Dict[str, Any], proj_id: Any
    ) -> None:
        """Log detailed metrics for a finished project."""
        # Get project completion time
        completion_time = getattr(env, "timestep", 0)
        start_time = proj.get("start_time", 0)
        # Calculate project quality if possible
        project_quality = proj.get("quality", None)

        # Get contributor details
        contributors = proj.get("contributors", [])
        contributor_efforts = []
        for agent_idx in contributors:
            if hasattr(env, "agent_project_effort") and agent_idx < len(
                env.agent_project_effort
            ):
                agent_effort = env.agent_project_effort[agent_idx].get(proj_id, 0.0)
                contributor_efforts.append(_as_float(agent_effort))

        project_detail = {
            "project_id": str(proj_id),
            "completion_step": completion_time,
            "start_time": start_time,
            "duration": proj.get("time_window", 0),
            "required_effort": _as_float(proj.get("required_effort", 0.0)),
            "actual_effort": _as_float(proj.get("current_effort", 0.0)),
            "team_size": len(contributors),
            "contributors": [f"agent_{idx}" for idx in contributors],
            "contributor_efforts": contributor_efforts,
            "approx_reward": _as_float(proj.get("approx_reward", 0.0)),
            "time_window": _as_float(proj.get("time_window", 0.0)),
            "project_quality": (
                float(project_quality) if project_quality is not None else None
            ),
            "validator_skill": (
                float(env.validators[proj["validator"]]["skill"])
                if hasattr(env, "validators")
                and "validator" in proj
                and proj["validator"] < len(env.validators)
                else None
            ),
        }

        self.finished_project_details.append(project_detail)

    def to_dict(self) -> Dict[str, Any]:
        avg_effort_per_step = (
            (self.total_effort_applied / self.total_steps) if self.total_steps else 0.0
        )
        avg_effort_per_finished_project = _safe_mean(self.finished_projects_efforts)
        avg_finished_project_duration = _safe_mean(self.finished_projects_durations)
        avg_finished_project_team_size = _safe_mean(
            [float(x) for x in self.finished_projects_team_sizes]
        )
        avg_reward_per_step = (
            (self.total_rewards_distributed / self.total_steps)
            if self.total_steps
            else 0.0
        )

        # Success rate calculations
        success_rate = (
            (self.successful_projects_count / self.finished_projects_count)
            if self.finished_projects_count > 0
            else 0.0
        )
        avg_effort_successful = _safe_mean(self.successful_projects_efforts)
        avg_effort_unsuccessful = _safe_mean(self.unsuccessful_projects_efforts)

        return {
            "steps": self.total_steps,
            "total_effort": self.total_effort_applied,
            "avg_effort_per_step": avg_effort_per_step,
            "finished_projects": self.finished_projects_count,
            "successful_projects": self.successful_projects_count,
            "unsuccessful_projects": self.unsuccessful_projects_count,
            "success_rate": success_rate,
            "avg_effort_per_finished_project": avg_effort_per_finished_project,
            "avg_effort_successful_projects": avg_effort_successful,
            "avg_effort_unsuccessful_projects": avg_effort_unsuccessful,
            "avg_finished_project_duration": avg_finished_project_duration,
            "avg_finished_project_team_size": avg_finished_project_team_size,
            "total_rewards_distributed": self.total_rewards_distributed,
            "avg_reward_per_step": avg_reward_per_step,
            "per_agent_total_effort": dict(self.total_effort_per_agent),
            "observation_aggregates": dict(self.last_obs_aggregate),
            # Enhanced metrics for visualizations
            "papers_per_author": dict(self.papers_per_author),
            "authors_per_paper": self.authors_per_paper.copy(),
            "prestige_category_stats": dict(self.prestige_category_stats),
            "effort_ratios": self.effort_ratios.copy(),
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data specifically formatted for visualizations."""
        # Calculate acceptance rates per prestige category
        acceptance_rates = {}
        for category, stats in self.prestige_category_stats.items():
            if stats["total"] > 0:
                acceptance_rates[category] = stats["success"] / stats["total"]
            else:
                acceptance_rates[category] = 0.0

        # Calculate effort ratio statistics
        effort_ratio_stats = {}
        if self.effort_ratios:
            effort_ratio_stats = {
                "min": min(self.effort_ratios),
                "max": max(self.effort_ratios),
                "mean": sum(self.effort_ratios) / len(self.effort_ratios),
                "median": (
                    sorted(self.effort_ratios)[len(self.effort_ratios) // 2]
                    if len(self.effort_ratios) % 2 == 1
                    else (
                        sorted(self.effort_ratios)[len(self.effort_ratios) // 2 - 1]
                        + sorted(self.effort_ratios)[len(self.effort_ratios) // 2]
                    )
                    / 2
                ),
            }

        return {
            "papers_per_author": self.papers_per_author,
            "authors_per_paper": self.authors_per_paper,
            "prestige_category_stats": self.prestige_category_stats,
            "acceptance_rates": acceptance_rates,
            "effort_ratios": self.effort_ratios,
            "effort_ratio_stats": effort_ratio_stats,
        }

    def summary_line(self) -> str:
        d = self.to_dict()
        return (
            f"t={d['steps']} | finished={d['finished_projects']} | "
            f"success={d['successful_projects']}/{d['finished_projects']} "
            f"({d['success_rate']:.1%}) | "
            f"effort(total={d['total_effort']:.2f},/step={d['avg_effort_per_step']:.2f},/fin_proj={d['avg_effort_per_finished_project']:.2f}) | "
            f"rewards(total={d['total_rewards_distributed']:.2f},/step={d['avg_reward_per_step']:.2f}) | "
            f"open={int(d['observation_aggregates'].get('open_projects_count', 0))} "
            f"active={int(d['observation_aggregates'].get('active_projects_count', 0))}"
        )

    def get_project_details(self) -> List[Dict[str, Any]]:
        """Get detailed metrics for all finished projects."""
        return self.finished_project_details.copy()


def _as_float(value: Any) -> float:
    """Convert numpy scalars/arrays or Python numbers to float safely."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _as_float(value[0])
    try:
        arr = np.asarray(value)
        if arr.size == 1:
            return float(arr.item())
        return float(arr.sum())
    except Exception:
        try:
            return float(value)
        except Exception:
            return 0.0


def _first(value: Any, default: Any) -> Any:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return arr.flat[0]
    except Exception:
        return value if value is not None else default


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
