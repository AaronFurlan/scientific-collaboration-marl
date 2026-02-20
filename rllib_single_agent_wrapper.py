"""
rllib_single_agent_wrapper.py

Single-agent Gymnasium wrapper around your PettingZoo ParallelEnv that:

1) Flattens nested observations (including action_mask) into a single 1D Box(float32)
   -> RLlib new API stack can auto-pick an encoder.

2) Flattens the Dict action-space into ONE Discrete macro-action
   -> PPO can output a simple categorical distribution.

3) Decodes the macro-action back into the env's expected dict action:
   {"choose_project": int, "collaborate_with": np.ndarray[int8], "put_effort": int}

4) Repairs (clips) invalid decoded actions using the env-provided action_mask
   -> prevents "Couldn't find project" spam and reduces garbage transitions.

5) Supports fixed policies for non-controlled agents via `other_policies`.
   `other_policies[agent_id]` must accept the nested obs:
   {"observation": ..., "action_mask": ...} and return an env-valid action dict.

Notes:
- This macro-action approach is only feasible for SMALL max_peer_group_size (e.g. 8).
  Because ACTION_N = CP * PE * 2^CB.
- If you set max_peer_group_size=100, this is impossible. Then you need a multi-head model.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import logging

logger = logging.getLogger(__name__)


class RLLibSingleAgentWrapper(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        env,
        controlled_agent: Optional[Callable[[Dict[str, Any]], str]] = None,
        other_policies: Optional[Dict[str, Callable[[Any], Any]]] = None,
        *,
        force_episode_horizon: Optional[int] = None,
    ):
        """
        Args:
            env: PettingZoo ParallelEnv (your PeerGroupEnvironment)
            controlled_agent: None -> pick first agent each reset,
                              str -> fixed agent id,
                              callable -> choose agent given observations dict.
            other_policies: dict(agent_id -> callable(nested_obs)-> action_dict)
                           For non-controlled agents.
            force_episode_horizon: if set, wrapper truncates after this many steps
                                  (guarantees RLlib gets episodes).
        """
        super().__init__()
        self.env = env
        self._choose_controlled = controlled_agent
        self.other_policies = other_policies or {}

        self.current_controlled: Optional[str] = None
        self._last_observations: Dict[str, Any] = {}

        # Horizon enforcement (helpful for RLlib metrics)
        # If the caller didn't provide a force horizon, use the env's own max episode length
        # (this guarantees we eventually truncate episodes so RLlib receives episode metrics).
        if force_episode_horizon is None and hasattr(env, "n_steps"):
            self._force_horizon = int(getattr(env, "n_steps"))
            logger.debug("No force_episode_horizon provided; using env.n_steps=%s", self._force_horizon)
        else:
            self._force_horizon = force_episode_horizon

        self._t = 0

        # Choose a stable reference agent
        self._ref_agent = getattr(env, "possible_agents", [None])[0]
        if self._ref_agent is None:
            raise ValueError("env.possible_agents is missing/empty; cannot select ref agent.")

        # --- Flatten action space into one Discrete macro-action ---
        # These attributes exist in your PeerGroupEnvironment
        self._CP = int(self.env.n_projects_per_step + 1)      # choose_project
        self._PE = int(self.env.max_projects_per_agent + 1)   # put_effort
        self._CB = int(self.env.max_peer_group_size)          # collaborate bits

        if self._CB > 16:
            # Not a hard error, but warn loudly via exception to avoid silent insanity.
            # 2^17 already 131072; action space explodes fast.
            raise ValueError(
                f"max_peer_group_size={self._CB} is too large for macro-action encoding. "
                "Use <= 12-ish, or implement a multi-head model."
            )

        self._COLLAB_BASE = 1 << self._CB
        self._ACTION_N = self._CP * self._PE * self._COLLAB_BASE
        self.action_space = gym.spaces.Discrete(self._ACTION_N)

        # --- Build observation space by sampling one real observation and flattening it ---
        observations, infos = self.env.reset()
        sample_agent = self._ref_agent if self._ref_agent in observations else next(iter(observations.keys()))
        sample_vec = self._flatten_to_vector(observations[sample_agent])

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(sample_vec.size),),
            dtype=np.float32,
        )

        # Store reset state so first step has consistent memory
        self._last_observations = observations
        self.current_controlled = sample_agent

    # -----------------------------
    # Action encoding/decoding
    # -----------------------------

    def _decode_action(self, a: int) -> Dict[str, Any]:
        """
        Decode macro-action a into env dict action.
        """
        a = int(a)

        collab_code = a % self._COLLAB_BASE
        a //= self._COLLAB_BASE

        put_effort = a % self._PE
        a //= self._PE

        choose_project = a % self._CP

        collab_bits = np.array(
            [(collab_code >> i) & 1 for i in range(self._CB)],
            dtype=np.int8,
        )

        return {
            "choose_project": int(choose_project),
            "collaborate_with": collab_bits,
            "put_effort": int(put_effort),
        }

    def _apply_action_mask(self, decoded: Dict[str, Any], nested_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Repair invalid actions using the env-provided action_mask.
        Treat mask values >0 as allowed (your env sometimes uses 2).
        """
        mask = nested_obs.get("action_mask", {})
        if not isinstance(mask, dict):
            return decoded

        # choose_project
        cp_mask = np.asarray(mask.get("choose_project", []))
        if cp_mask.size:
            cp = int(decoded["choose_project"])
            if cp < 0 or cp >= cp_mask.size or cp_mask[cp] <= 0:
                decoded["choose_project"] = 0

        # put_effort
        pe_mask = np.asarray(mask.get("put_effort", []))
        if pe_mask.size:
            pe = int(decoded["put_effort"])
            if pe < 0 or pe >= pe_mask.size or pe_mask[pe] <= 0:
                decoded["put_effort"] = 0

        # collaborate_with
        c_mask = np.asarray(mask.get("collaborate_with", []))
        if c_mask.size:
            c = np.asarray(decoded["collaborate_with"], dtype=np.int8).copy()
            allowed = (c_mask > 0)
            # Align lengths defensively
            L = min(len(c), len(allowed))
            c[:L][~allowed[:L]] = 0
            if len(c) > L:
                c[L:] = 0
            decoded["collaborate_with"] = c

        return decoded

    # -----------------------------
    # Observation flattening
    # -----------------------------

    def _flatten_to_vector(self, nested_obs: Any) -> np.ndarray:
        """
        Your env returns per-agent:
          {"observation": <dict>, "action_mask": <dict>}
        We flatten observation recursively + flatten mask and append it.
        """
        if not (isinstance(nested_obs, dict) and "observation" in nested_obs):
            raise TypeError("Expected nested obs: {'observation': ..., 'action_mask': ...}")

        obs_part = nested_obs.get("observation", {})
        mask_part = nested_obs.get("action_mask", {})

        obs_vec = self._flatten_any(obs_part)
        mask_vec = self._flatten_mask(mask_part)

        out = np.concatenate([obs_vec, mask_vec]).astype(np.float32, copy=False)
        return out

    def _flatten_any(self, x: Any) -> np.ndarray:
        """
        Recursively flatten dict/arrays/scalars into 1D float32.
        Dict keys are sorted to guarantee stable ordering.
        """
        if isinstance(x, dict):
            parts: List[np.ndarray] = []
            for k in sorted(x.keys()):
                parts.append(self._flatten_any(x[k]))
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)

        arr = np.asarray(x)
        if arr.dtype == object:
            raise TypeError(f"Non-numeric object in observation: {type(x)}")
        return arr.astype(np.float32, copy=False).ravel()

    def _flatten_mask(self, mask: Any) -> np.ndarray:
        """
        Flatten mask dict into 1D float32 (0/1) in sorted key order.
        """
        if isinstance(mask, dict):
            parts: List[np.ndarray] = []
            for k in sorted(mask.keys()):
                v = np.asarray(mask[k])
                v01 = (v > 0).astype(np.float32).ravel()
                parts.append(v01)
            return np.concatenate(parts) if parts else np.zeros((0,), dtype=np.float32)

        v = np.asarray(mask)
        return (v > 0).astype(np.float32).ravel()

    # -----------------------------
    # Gymnasium API
    # -----------------------------

    def reset(self, *, seed=None, options=None):
        logger.debug("Wrapper.reset() called; delegating to env.reset(seed=%s)", seed)
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_observations = observations
        self._t = 0

        # choose controlled agent
        if callable(self._choose_controlled):
            self.current_controlled = self._choose_controlled(observations)
        elif isinstance(self._choose_controlled, str):
            self.current_controlled = self._choose_controlled
        else:
            self.current_controlled = next(iter(observations.keys()))

        if self.current_controlled not in observations:
            raise RuntimeError(f"controlled agent {self.current_controlled} not in env agents")

        obs_vec = self._flatten_to_vector(observations[self.current_controlled])
        info = infos.get(self.current_controlled, {})

        # Validate observation vector shape/dtype
        if not isinstance(obs_vec, np.ndarray):
            logger.warning("reset returned non-ndarray observation for %s: %s", self.current_controlled, type(obs_vec))
            obs_vec = np.asarray(obs_vec, dtype=np.float32)
        if obs_vec.ndim != 1:
            logger.warning("reset flattened observation not 1D for %s, reshaping: shape=%s", self.current_controlled, getattr(obs_vec, 'shape', None))
            obs_vec = obs_vec.ravel()
        obs_vec = obs_vec.astype(np.float32, copy=False)

        logger.info("Episode START for %s (obs_len=%d, force_horizon=%s)", self.current_controlled, obs_vec.size, self._force_horizon)
        # Also print to stdout to ensure visibility in simple subprocess runs / remote workers.
        print(f"[WRAPPER] Episode START for {self.current_controlled} (obs_len={obs_vec.size}, force_horizon={self._force_horizon})")
        return obs_vec, info

    def step(self, action):
        if self.current_controlled is None:
            raise RuntimeError("Wrapper not reset() before step().")

        active_agents: Iterable[str] = getattr(self.env, "agents", list(self._last_observations.keys()))
        actions: Dict[str, Any] = {}

        for ag in active_agents:
            if ag == self.current_controlled:
                nested_obs = self._last_observations[ag]
                decoded = self._decode_action(action)
                decoded = self._apply_action_mask(decoded, nested_obs)
                actions[ag] = decoded
            else:
                policy = self.other_policies.get(ag)
                if policy is not None:
                    actions[ag] = policy(self._last_observations[ag])
                else:
                    actions[ag] = self.env.action_space(ag).sample()

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_observations = observations

        obs_vec = self._flatten_to_vector(observations[self.current_controlled])
        reward = float(rewards.get(self.current_controlled, 0.0))
        terminated = bool(terminations.get(self.current_controlled, False))
        truncated = bool(truncations.get(self.current_controlled, False))
        info = infos.get(self.current_controlled, {})

        # Force horizon if requested (ensures RLlib sees completed episodes)
        self._t += 1
        if self._force_horizon is not None and self._t >= int(self._force_horizon):
            truncated = True

        # Validate obs/reward/flags
        if not isinstance(obs_vec, np.ndarray):
            logger.warning("step returned non-ndarray observation for %s: %s", self.current_controlled, type(obs_vec))
            print(f"[WRAPPER] WARNING: step returned non-ndarray observation for {self.current_controlled}: {type(obs_vec)}")
            obs_vec = np.asarray(obs_vec, dtype=np.float32)
        if obs_vec.ndim != 1:
            logger.warning("step flattened observation not 1D for %s, reshaping: shape=%s", self.current_controlled, getattr(obs_vec, 'shape', None))
            print(f"[WRAPPER] WARNING: step flattened observation not 1D for {self.current_controlled}, reshaping: shape={getattr(obs_vec,'shape',None)}")
            obs_vec = obs_vec.ravel()
        obs_vec = obs_vec.astype(np.float32, copy=False)

        if not isinstance(reward, float):
            try:
                reward = float(reward)
            except Exception:
                logger.error("Non-convertible reward for %s: %s", self.current_controlled, type(reward))
                print(f"[WRAPPER] ERROR: Non-convertible reward for {self.current_controlled}: {type(reward)}")
                reward = 0.0

        if terminated or truncated:
            logger.info(
                "Episode END for %s (t=%s) -> terminated=%s truncated=%s reward=%s",
                self.current_controlled,
                self._t,
                terminated,
                truncated,
                reward,
            )
            print(f"[WRAPPER] Episode END for {self.current_controlled} (t={self._t}) -> terminated={terminated} truncated={truncated} reward={reward}")
            # reset per-episode counter so next reset shows a fresh episode start
            self._t = 0

        return obs_vec, reward, terminated, truncated, info

    def render(self, mode="human"):
        return getattr(self.env, "render", lambda *a, **k: None)(mode)

    def close(self):
        return getattr(self.env, "close", lambda *a, **k: None)()