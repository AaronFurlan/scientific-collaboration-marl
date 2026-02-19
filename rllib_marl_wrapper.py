from copy import deepcopy
from gymnasium.spaces import Dict as GymDict, MultiBinary
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

from debug_peer_group_env import truncations


class RLLibMARLWrapper(ParallelEnv):
    """
    Wrapper to convert a PettingZoo environment (like PeerGroupEnvironment) that returns per-agent
    {'observation': ..., 'action_mask': ...} into a flattend observation where 'action_mask' is top-level key
    and observation_space includes that mask.
    """

    def __init__(self, env):
        self.env = env
        self.metadata = getattr(env, 'metadata', {})
        self.agents = getattr(env, 'agents', [])
        self.possible_agents = getattr(env, 'possible_agents', [])

    def _flatten_obs(self, nested_obs):
        if isinstance(nested_obs, dict) and 'observation' in nested_obs:
            inner = deepcopy(nested_obs.get('observation', {}))
            mask = deepcopy(nested_obs.get('action_mask', {}))
        else:
            inner = deepcopy(nested_obs)
            mask = {}
        inner['action_mask'] = mask
        return inner

    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self.agents = getattr(self.env, 'agents', list(observations.keys()))
        self.possible_agents = getattr(self.env, 'possible_agents', self.possible_agents)
        flat = {a: self._flatten_obs(v) for a, v in observations.items()}
        return flat, infos

    def step(self, actions):
        observations, rewards, dones, truncations, infos = self.env.step(actions)
        self.agents = getattr(self.env, 'agents', list(observations.keys()))
        flat = {a: self._flatten_obs(v) for a, v in observations.items()}
        return flat, rewards, dones, truncations, infos

    def observation_space(self, agent):
        inner_space = self.env.observation_space(agent)
        # Mask space matching PeerGroupEnvironment._get_action_mask
        n_projects = getattr(self.env, 'n_projects_per_step', 1)
        max_peer_group = getattr(self.env, "max_peer_group_size", 1)
        max_projects_per_agent = getattr(self.env, "max_projects_per_agent", 1)
        mask_space = GymDict(
            {
                "choose_project": MultiBinary(n_projects + 1),
                "collaborate_with": MultiBinary(max_peer_group),
                "put_effort": MultiBinary(max_projects_per_agent + 1),
            }
        )
        combined = GymDict({**inner_space.spaces, "action_mask": mask_space})
        return combined

    def action_space(self, agent):
        return self.env.action_space(agent)

    def render(self, mode="human"):
        return getattr(self.env, "render", lambda *a, **k: None)(mode)

    def close(self):
        return getattr(self.env, "close", lambda *a, **k: None)()