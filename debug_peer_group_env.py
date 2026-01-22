import json
import os
import random

import numpy as np
from env.peer_group_environment import PeerGroupEnvironment
from log_simulation import SimLog
from stats_tracker import SimulationStats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Convert numpy arrays to lists for JSON serialization (handles nested structures)
def convert_numpy(obj):
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    return obj


# # Initialize environment
# env = PeerGroupEnvironment(
#     max_agents=2_000,
#     start_agents=100,
#     n_groups=20,
#     max_peer_group_size=100,
#     n_projects_per_step=1,
#     max_projects_per_agent=10,
#     max_agent_age=750,
#     max_rewardless_steps=250,
# )
# Initialize environment
env = PeerGroupEnvironment(
    start_agents=10,
    max_agents=100,
    n_groups=2,
    max_peer_group_size=100,
    n_projects_per_step=1,
    max_projects_per_agent=5,
    max_agent_age=500,
    max_rewardless_steps=250,
)

obs, infos = env.reset(seed=SEED)
for i, agent in enumerate(env.possible_agents):
    env.action_space(agent).seed(SEED + i)

stats = SimulationStats()
log = SimLog(
    "log",
    "debug_sim_actions.jsonl",
    "debug_sim_observations.jsonl",
    "debug_sim_projects.json",
)
log.start()

for step in range(500):
    actions = {}
    for agent in env.agents:
        actions[agent] = env.action_space(agent).sample(mask=env.action_masks[agent])
    obs, rewards, terminations, truncations, infos = env.step(actions)
    stats.update(env, obs, rewards, terminations, truncations)

    log.log_observation(
        {
            a: ob if env.active_agents[env.agent_to_id[a]] == 1 else None
            for a, ob in obs.items()
        }
    )
    log.log_action(
        {
            a: (
                act | {"archetype": "random"}
                if env.active_agents[env.agent_to_id[a]] == 1
                else None
            )
            for a, act in actions.items()
        }
    )
    # for agent in env.agents:
    # obs_converted = convert_numpy(obs[agent])
    # if agent == "agent_0" and step > 50:
    #     #     print(env._get_active_projects(0))
    #     #     print(env.action_masks[agent])
    #     # print(env.agent_active_projects[0])
    #     print(f"{agent} obs: {json.dumps(obs_converted, indent=2)}")

    # Uncomment and fix the rewards line too
    # rewards_converted = convert_numpy(rewards[agent])
    # print(f"{agent} reward: {json.dumps(rewards_converted, indent=2)}")
    # breakpoint()
    # Print a concise stats summary each step
    # if step > 500:
    #     active_agent_1 = list(env.active_agents).index(1)
    #     print(f"agent {active_agent_1}")
    #     print(env.action_masks[f"agent_{active_agent_1}"])
    #     print(actions[f"agent_{active_agent_1}"])
    #     breakpoint()
    # Print progress
    if step % 10 == 0:
        print(f"Step {step}: {stats.summary_line()}")

log.log_projects(env.projects.values())
env.area.save("log/debug_sim_area.pickle")
