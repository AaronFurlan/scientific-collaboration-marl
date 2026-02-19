import unittest
import numpy as np

from gymnasium.spaces import Box, Dict as GymDict, Discrete, MultiBinary
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper

class MockParallelEnv:
    """
    Minimal ParallelEnv-like mock returning nested observations with
    {'observation': {...}, 'action_mask': {...}} for two agents.
    """

    def __init__(self):
        self.agents = ['agent_0', 'agent_1']
        self._obs_space = GymDict({
            'vec': Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        self._act_space = Discrete(3)

    def reset(self, seed=None, options=None):
        self.agents = ["agent_0", "agent_1"]
        obs = {
            "agent_0": {
                "observation": {"vec": np.zeros(3, dtype=np.float32)},
                "action_mask": {"choose_project": np.array([1, 0, 1], dtype=np.int8)}
            },
            "agent_1": {
                "observation": {"vec": np.ones(3, dtype=np.float32)},
                "action_mask": {"choose_project": np.array([1, 1, 0], dtype=np.int8)}
            }
        }
        infos = {a: {} for a in obs}
        return obs, infos

    def step(self, actions):
        # reward for agent_0 = 1.0 iff action == 1
        reward0 = 1.0 if actions.get("agent_0") == 1 else 0.0
        rewards = {"agent_0": reward0, "agent_1": 0.0}

        obs = {}
        for a in self.agents:
            base = np.zeros(3, dtype=np.float32) if a == "agent_0" else np.ones(3, dtype=np.float32)
            obs[a] = {
                "observation": {"vec": base + 0.5},
                "action_mask": {"choose_project": np.array([1, 1, 1], dtype=np.int8)}
            }

        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def render(self, mode="human"):
        return None

    def close(self):
        return None

class TestSingleAgentWrapper(unittest.TestCase):
    def test_reset_flattens_and_sets_spaces(self):
        env = MockParallelEnv()
        wrapper = RLLibSingleAgentWrapper(env, controlled_agent="agent_0", other_policies={})

        obs, infos = wrapper.reset()
        # flattened observation must include top-level `action_mask`
        self.assertIsInstance(obs, dict)
        self.assertIn("action_mask", obs)

        # observation_space should be set (and include mask when appropriate)
        self.assertIsNotNone(wrapper.observation_space)

    def test_step_calls_other_policy_and_returns_controlled_reward(self):
        env = MockParallelEnv()
        called = {"agent_1": False}

        def other_policy(obs):
            # ensure we receive flattened obs with `action_mask`
            self.assertIsInstance(obs, dict)
            self.assertIn("action_mask", obs)
            called["agent_1"] = True
            return 2  # arbitrary action for agent_1

        wrapper = RLLibSingleAgentWrapper(env, controlled_agent="agent_0", other_policies={"agent_1": other_policy})

        obs, infos = wrapper.reset()
        self.assertIn("action_mask", obs)

        # step with controlled action 1 -> per MockParallelEnv reward 1.0
        step_obs, reward, terminated, truncated, info = wrapper.step(1)
        self.assertEqual(reward, 1.0)
        self.assertTrue(called["agent_1"])
        self.assertIsInstance(step_obs, dict)
        self.assertIn("action_mask", step_obs)

if __name__ == '__main__':
    unittest.main()
