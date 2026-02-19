from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Iterable
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict as GymDict, MultiBinary
from copy import deepcopy

class RLLibSingleAgentWrapper(gym.Env):
    """
    Wrap a PettingZoo ParallelEnv into a single-agent Gymnasium Env.
    - env: a pettingzoo.ParallelEnv
    - controlled_agent: agent id (str) or callable(observations) -> agent id. If None, first active agent is used.
    - other_policies: dict(agent_id -> callable(flat_obs) -> action) for non-controlled agents.
      If an agent has no policy, its action is sampled from its action_space.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env,
        controlled_agent: Optional[Callable[[Dict[str, Any]], str]] = None,
        other_policies: Optional[Dict[str, Callable[[Any], Any]]] = None,
    ):
        self.env = env
        self._choose_controlled = controlled_agent
        self.other_policies = other_policies or {}
        self.current_controlled: Optional[str] = None
        self._last_observations: Dict[str, Any] = {}
        self.observation_space = None
        self.action_space = None

    def _flatten_obs(self, nested: Any) -> Any:
        # Accept either nested {"observation":..., "action_mask":...} or already-flat obs
        if isinstance(nested, dict) and "observation" in nested:
            inner = deepcopy(nested.get("observation", {}))
            mask = deepcopy(nested.get("action_mask", {}))
        else:
            inner = deepcopy(nested)
            mask = {}
        inner["action_mask"] = mask
        return inner

    def _infer_mask_space(self, mask: Any):
        # Build a Gym Dict/Space that matches the mask structure (MultiBinary)
        if isinstance(mask, dict):
            spaces = {}
            for k, v in mask.items():
                arr = np.asarray(v)
                spaces[k] = MultiBinary(int(arr.size))
            return GymDict(spaces)
        else:
            arr = np.asarray(mask)
            return MultiBinary(int(arr.size))

    def reset(self, *, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        # choose controlled agent
        if callable(self._choose_controlled):
            self.current_controlled = self._choose_controlled(observations)
        elif isinstance(self._choose_controlled, str):
            self.current_controlled = self._choose_controlled
        else:
            self.current_controlled = next(iter(observations.keys()))
        if self.current_controlled not in observations:
            raise RuntimeError(f"controlled agent {self.current_controlled} not in env agents")

        # flatten and store
        flat = {a: self._flatten_obs(v) for a, v in observations.items()}
        self._last_observations = flat

        # lazy spaces based on controlled agent
        if self.observation_space is None:
            try:
                inner_space = self.env.observation_space(self.current_controlled)
            except Exception:
                inner_space = None
            sample_obs = flat[self.current_controlled]
            if "action_mask" in sample_obs and sample_obs["action_mask"]:
                mask_space = self._infer_mask_space(sample_obs["action_mask"])
                if isinstance(inner_space, GymDict):
                    combined = GymDict({**inner_space.spaces, "action_mask": mask_space})
                elif inner_space is not None:
                    combined = GymDict({"observation": inner_space, "action_mask": mask_space})
                else:
                    combined = GymDict({"action_mask": mask_space})
                self.observation_space = combined
            else:
                self.observation_space = inner_space or gym.spaces.Box(-float("inf"), float("inf"), shape=(0,))
            self.action_space = getattr(self.env, "action_space")(self.current_controlled)

        return deepcopy(self._last_observations[self.current_controlled]), infos

    def step(self, action):
        # build actions for all active agents
        active_agents: Iterable[str] = getattr(self.env, "agents", list(self._last_observations.keys()))
        actions: Dict[str, Any] = {}
        for ag in active_agents:
            if ag == self.current_controlled:
                actions[ag] = action
            else:
                obs_for_other = self._last_observations.get(ag)
                policy = self.other_policies.get(ag)
                if policy is not None and obs_for_other is not None:
                    actions[ag] = policy(obs_for_other)
                else:
                    actions[ag] = self.env.action_space(ag).sample()

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        flat = {a: self._flatten_obs(v) for a, v in observations.items()}
        self._last_observations = flat

        obs = deepcopy(flat[self.current_controlled])
        reward = rewards.get(self.current_controlled, 0.0)
        terminated = terminations.get(self.current_controlled, False)
        truncated = truncations.get(self.current_controlled, False)
        info = infos.get(self.current_controlled, {})

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return getattr(self.env, "render", lambda *a, **k: None)(mode)

    def close(self):
        return getattr(self.env, "close", lambda *a, **k: None)()
