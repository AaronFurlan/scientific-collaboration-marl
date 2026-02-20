"""
train_ppo_rllib.py

Close in spirit to `run_policy_simulation.py`, but uses RLlib PPO to control
ONE agent while the rest of the population follows fixed (hand-crafted) policies.

Key similarities to run_policy_simulation.py:
- You pick a policy distribution (careerist / orthodox_scientist / mass_producer)
  for the *other* agents (fixed policies).
- You configure the same environment knobs (n_agents, max_steps, n_groups, etc.).
- You run "episodes" (rollouts) for a fixed horizon.

Key differences:
- RLlib PPO controls exactly one "controlled agent" (by default agent_0).
- The wrapper handles:
  - observation flattening (for RLlib encoders),
  - macro-action encoding/decoding,
  - action-mask repair,
  - fixed policies for non-controlled agents.

Install:
  pip install "ray[rllib]" torch
  # or
  pip install "ray[rllib]" tensorflow

Run:
  python train_ppo_rllib.py --iterations 5 --policy-config Balanced
"""

from __future__ import annotations

import argparse
import pprint
from typing import Any, Callable, Dict, Optional

import ray
from ray import tune

from agent_policies import (
    create_mixed_policy_population,
    create_per_group_policy_population,
    do_nothing_policy,
    get_policy_function,
)
from env.peer_group_environment import PeerGroupEnvironment
from rllib_single_agent_wrapper import RLLibSingleAgentWrapper


# Match run_policy_simulation.py style
POLICY_CONFIGS: Dict[str, Dict[str, float]] = {
    "All Careerist": {"careerist": 1.0, "orthodox_scientist": 0.0, "mass_producer": 0.0},
    "All Orthodox": {"careerist": 0.0, "orthodox_scientist": 1.0, "mass_producer": 0.0},
    "All Mass Producer": {"careerist": 0.0, "orthodox_scientist": 0.0, "mass_producer": 1.0},
    "Balanced": {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    "Careerist Heavy": {"careerist": 0.5, "orthodox_scientist": 0.5, "mass_producer": 0.0},
    "Orthodox Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
    "Mass Producer Heavy": {"careerist": 0.5, "orthodox_scientist": 0.0, "mass_producer": 0.5},
}


def _set_rollout_workers_compat(config, num_workers: int):
    """
    RLlib 2.x naming drift:
      - older 2.x: config.rollouts(num_rollout_workers=...)
      - newer 2.x: config.env_runners(num_env_runners=...)
    """
    if hasattr(config, "env_runners"):
        return config.env_runners(num_env_runners=num_workers)
    return config.rollouts(num_rollout_workers=num_workers)


def make_env_creator(
    *,
    # Env params (similar to run_policy_simulation.py)
    n_agents: int,
    start_agents: int,
    max_steps: int,
    max_rewardless_steps: int,
    n_groups: int,
    max_peer_group_size: int,
    n_projects_per_step: int,
    max_projects_per_agent: int,
    max_agent_age: int,
    acceptance_threshold: float,
    reward_function: str,
    seed: int,
    # Fixed-policy population params
    policy_distribution: Dict[str, float],
    group_policy_homogenous: bool,
    prestige_threshold: float,
    novelty_threshold: float,
    effort_threshold: int,
    # Which agent PPO controls
    controlled_agent_id: str,
) -> Callable[[Optional[Dict[str, Any]]], Any]:
    """
    Returns an RLlib-compatible env creator: f(env_config) -> gymnasium.Env
    We keep env_config support so RLlib can recreate envs if needed.
    """

    # Pre-build policy functions (same as run_policy_simulation.py)
    careerist_fn = get_policy_function("careerist")
    orthodox_fn = get_policy_function("orthodox_scientist")
    mass_prod_fn = get_policy_function("mass_producer")

    def _policy_from_name(policy_name: Optional[str]):
        # Returns a callable(nested_obs)->action_dict
        if policy_name is None:
            def _do_nothing(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return do_nothing_policy(obs, mask)
            return _do_nothing

        if policy_name == "careerist":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return careerist_fn(obs, mask, prestige_threshold)
            return _fn

        if policy_name == "orthodox_scientist":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return orthodox_fn(obs, mask, novelty_threshold)
            return _fn

        if policy_name == "mass_producer":
            def _fn(nested_obs):
                obs = nested_obs["observation"]
                mask = nested_obs["action_mask"]
                return mass_prod_fn(obs, mask, effort_threshold)
            return _fn

        # Fallback
        def _fallback(nested_obs):
            obs = nested_obs["observation"]
            mask = nested_obs["action_mask"]
            return do_nothing_policy(obs, mask)
        return _fallback

    def _env_creator(env_config: Optional[Dict[str, Any]] = None):
        env_config = env_config or {}

        # 1) Build env (same knobs as your simulation script)
        env = PeerGroupEnvironment(
            start_agents=env_config.get("start_agents", start_agents),
            max_agents=env_config.get("n_agents", n_agents),
            max_steps=env_config.get("max_steps", max_steps),
            n_groups=env_config.get("n_groups", n_groups),
            max_peer_group_size=env_config.get("max_peer_group_size", max_peer_group_size),
            n_projects_per_step=env_config.get("n_projects_per_step", n_projects_per_step),
            max_projects_per_agent=env_config.get("max_projects_per_agent", max_projects_per_agent),
            max_agent_age=env_config.get("max_agent_age", max_agent_age),
            max_rewardless_steps=env_config.get("max_rewardless_steps", max_rewardless_steps),
            acceptance_threshold=env_config.get("acceptance_threshold", acceptance_threshold),
            reward_mode=env_config.get("reward_function", reward_function),
            render_mode=None,
        )

        # 2) Create fixed-policy assignments (same logic as run_policy_simulation.py)
        if group_policy_homogenous:
            agent_policy_names = create_per_group_policy_population(
                n_agents, policy_distribution
            )
        else:
            agent_policy_names = create_mixed_policy_population(
                n_agents, policy_distribution, seed=seed
            )

        # 3) Build other_policies mapping: agent_id -> callable(nested_obs)->action_dict
        other_policies: Dict[str, Callable[[Any], Any]] = {}
        for agent_id in env.possible_agents:
            if agent_id == controlled_agent_id:
                continue
            idx = env.agent_to_id[agent_id]
            pol_name = agent_policy_names[idx]
            other_policies[agent_id] = _policy_from_name(pol_name)

        # 4) Wrap to single-agent env for PPO
        # Force horizon -> ensures RLlib gets completed episodes & metrics
        wrapper = RLLibSingleAgentWrapper(
            env,
            controlled_agent=controlled_agent_id,
            other_policies=other_policies,
            force_episode_horizon=max_steps,
        )

        return wrapper

    return _env_creator


def main(
    *,
    iterations: int,
    framework: str,
    policy_config_name: str,
    group_policy_homogenous: bool,
    seed: int,
    # Env knobs
    n_agents: int,
    start_agents: int,
    max_steps: int,
    max_rewardless_steps: int,
    n_groups: int,
    max_peer_group_size: int,
    n_projects_per_step: int,
    max_projects_per_agent: int,
    max_agent_age: int,
    acceptance_threshold: float,
    reward_function: str,
    # Threshold knobs for heuristics
    prestige_threshold: float,
    novelty_threshold: float,
    effort_threshold: int,
    # Controlled agent
    controlled_agent_id: str,
):
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
    except Exception as e:
        raise RuntimeError(
            "RLlib not importable. Install dependencies like:\n"
            '  pip install "ray[rllib]" torch\n'
            "Underlying error:\n"
            f"{e}"
        )

    if framework not in {"torch", "tf2"}:
        raise ValueError('framework must be "torch" or "tf2"')

    if policy_config_name not in POLICY_CONFIGS:
        raise ValueError(f"Unknown policy config '{policy_config_name}'. Options: {list(POLICY_CONFIGS.keys())}")

    policy_distribution = POLICY_CONFIGS[policy_config_name]

    # IMPORTANT: macro-action encoding explodes with large max_peer_group_size.
    if max_peer_group_size > 16:
        raise ValueError(
            f"max_peer_group_size={max_peer_group_size} is too large for the current "
            "macro-action wrapper approach. Use <= 12-ish for PPO training, or implement "
            "a multi-head action model."
        )

    # 1) Ray init
    ray.init(ignore_reinit_error=True)

    # 2) Register env
    env_name = "peer_group_single_agent_fixed_population"
    env_creator = make_env_creator(
        n_agents=n_agents,
        start_agents=start_agents,
        max_steps=max_steps,
        max_rewardless_steps=max_rewardless_steps,
        n_groups=n_groups,
        max_peer_group_size=max_peer_group_size,
        n_projects_per_step=n_projects_per_step,
        max_projects_per_agent=max_projects_per_agent,
        max_agent_age=max_agent_age,
        acceptance_threshold=acceptance_threshold,
        reward_function=reward_function,
        seed=seed,
        policy_distribution=policy_distribution,
        group_policy_homogenous=group_policy_homogenous,
        prestige_threshold=prestige_threshold,
        novelty_threshold=novelty_threshold,
        effort_threshold=effort_threshold,
        controlled_agent_id=controlled_agent_id,
    )
    tune.register_env(env_name, env_creator)

    # 3) Build PPO config
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config={
                "n_agents": n_agents,
                "start_agents": start_agents,
                "max_steps": max_steps,
                "max_rewardless_steps": max_rewardless_steps,
                "n_groups": n_groups,
                "max_peer_group_size": max_peer_group_size,
                "n_projects_per_step": n_projects_per_step,
                "max_projects_per_agent": max_projects_per_agent,
                "max_agent_age": max_agent_age,
                "acceptance_threshold": acceptance_threshold,
                "reward_function": reward_function,
            },
        )
        .framework(framework)
        # Keep small for debugging; increase later
        .training(train_batch_size=200)
        .debugging(log_level="WARN")
    )

    # Local-mode sampling for debugging
    config = _set_rollout_workers_compat(config, num_workers=1)

    # 4) Train
    algo = config.build_algo()
    try:
        print("\n=== PPO TRAINING (single controlled agent) ===")
        print(f"controlled_agent_id: {controlled_agent_id}")
        print(f"policy_config: {policy_config_name}  (group_policy_homogenous={group_policy_homogenous})")
        print(f"env: n_agents={n_agents}, start_agents={start_agents}, max_steps={max_steps}, "
              f"n_groups={n_groups}, max_peer_group_size={max_peer_group_size}")
        print(f"thresholds: prestige={prestige_threshold}, novelty={novelty_threshold}, effort={effort_threshold}")
        print(f"reward_function: {reward_function}, acceptance_threshold: {acceptance_threshold}\n")

        for i in range(iterations):
            result = algo.train()
            print(f"\nIteration {i + 1}/{iterations}")
            pprint.pprint(
                {
                    "episode_reward_mean": result.get("episode_reward_mean"),
                    "episode_len_mean": result.get("episode_len_mean"),
                    "timesteps_total": result.get("timesteps_total"),
                }
            )
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # RLlib
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "tf2"],
        help='Deep learning backend: "torch" or "tf2"',
    )

    # Match run_policy_simulation style
    parser.add_argument(
        "--policy-config",
        type=str,
        default="Balanced",
        choices=list(POLICY_CONFIGS.keys()),
        help="Which fixed-policy mixture to use for the non-controlled agents",
    )
    parser.add_argument(
        "--group-policy-homogenous",
        action="store_true",
        help="If set, assigns the same archetype per group (like create_per_group_policy_population). "
             "If not set, mixes per agent (like create_mixed_policy_population).",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Env knobs (keep small for PPO + macro-action)
    parser.add_argument("--n-agents", type=int, default=104)
    parser.add_argument("--start-agents", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-rewardless-steps", type=int, default=50)
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--max-peer-group-size", type=int, default=13)  # keep small!
    parser.add_argument("--n-projects-per-step", type=int, default=1)
    parser.add_argument("--max-projects-per-agent", type=int, default=6)
    parser.add_argument("--max-agent-age", type=int, default=750)

    # Reward knobs
    parser.add_argument("--acceptance-threshold", type=float, default=0.5)
    parser.add_argument("--reward-function", type=str, default="multiply", choices=["multiply", "evenly", "by_effort"])

    # Heuristic thresholds (same as your simulation script)
    parser.add_argument("--prestige-threshold", type=float, default=0.2)
    parser.add_argument("--novelty-threshold", type=float, default=0.8)
    parser.add_argument("--effort-threshold", type=int, default=22)

    # Controlled agent
    parser.add_argument("--controlled-agent-id", type=str, default="agent_0")

    args = parser.parse_args()

    main(
        iterations=args.iterations,
        framework=args.framework,
        policy_config_name=args.policy_config,
        group_policy_homogenous=args.group_policy_homogenous,
        seed=args.seed,
        n_agents=args.n_agents,
        start_agents=args.start_agents,
        max_steps=args.max_steps,
        max_rewardless_steps=args.max_rewardless_steps,
        n_groups=args.n_groups,
        max_peer_group_size=args.max_peer_group_size,
        n_projects_per_step=args.n_projects_per_step,
        max_projects_per_agent=args.max_projects_per_agent,
        max_agent_age=args.max_agent_age,
        acceptance_threshold=args.acceptance_threshold,
        reward_function=args.reward_function,
        prestige_threshold=args.prestige_threshold,
        novelty_threshold=args.novelty_threshold,
        effort_threshold=args.effort_threshold,
        controlled_agent_id=args.controlled_agent_id,
    )