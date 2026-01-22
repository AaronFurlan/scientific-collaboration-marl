"""
Example script showing how to use the agent policies with the peer group environment.
"""

import json

import numpy as np
from agent_policies import (create_mixed_policy_population,
                            create_per_group_policy_population,
                            do_nothing_policy, get_policy_function)
from env.peer_group_environment import PeerGroupEnvironment
from log_simulation import SimLog
from stats_tracker import SimulationStats

# Define different policy distributions to test
POLICY_CONFIGS = {
    "All Careerist": {
        "careerist": 1.0,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.0,
    },
    "All Orthodox": {
        "careerist": 0.0,
        "orthodox_scientist": 1.0,
        "mass_producer": 0.0,
    },
    "All Mass Producer": {
        "careerist": 0.0,
        "orthodox_scientist": 0.0,
        "mass_producer": 1.0,
    },
    "Balanced": {
        "careerist": 1 / 3,
        "orthodox_scientist": 1 / 3,
        "mass_producer": 1 / 3,
    },
    "Careerist Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.5,
        "mass_producer": 0.0,
    },
    "Orthodox Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.5,
    },
    "Mass Producer Heavy": {
        "careerist": 0.5,
        "orthodox_scientist": 0.0,
        "mass_producer": 0.5,
    },
}


def run_simulation_with_policies(
    n_agents: int = 100,
    max_steps: int = 1_000,
    max_rewardless_steps: int = 250,
    start_agents: int = 60,
    n_groups: int = 8,
    max_peer_group_size: int = 40,
    policy_distribution: dict = None,
    output_file_prefix: str = None,
    group_policy_homogenous=True,
    acceptance_threshold: float = 0.5,
    novelty_threshold: float = 0.8,
    prestige_threshold: float = 0.2,
    effort_threshold: int = 22,
    seed=42,
    reward_function: str = "multiply",
):
    """
    Run a simulation with different agent policies.

    Args:
        n_agents: Number of agents in the simulation
        max_steps: Maximum number of simulation steps
        policy_distribution: Distribution of policies among agents
        output_file: File to save results
    """

    # Create environment
    env = PeerGroupEnvironment(
        start_agents=start_agents,
        max_steps=max_steps,
        max_agents=n_agents,
        n_groups=n_groups,
        max_peer_group_size=max_peer_group_size,
        n_projects_per_step=1,
        max_projects_per_agent=8,
        max_agent_age=750,
        max_rewardless_steps=max_rewardless_steps,
        acceptance_threshold=acceptance_threshold,
        reward_mode=reward_function,
    )
    if group_policy_homogenous:
        # Create agent policy assignments
        agent_policies = create_per_group_policy_population(
            n_agents, policy_distribution
        )
    else:
        agent_policies = create_mixed_policy_population(
            n_agents, policy_distribution, seed=seed
        )
    print(
        f"Agent policy distribution: {dict(zip(*np.unique(agent_policies, return_counts=True)))}"
    )

    # Initialize stats tracker
    stats = SimulationStats()

    log = SimLog(
        "log",
        f"{output_file_prefix}_actions.jsonl",
        f"{output_file_prefix}_observations.jsonl",
        f"{output_file_prefix}_projects.json",
    )
    log.start()

    # Reset environment
    observations, infos = env.reset(seed=seed)

    # Simulation loop
    for step in range(max_steps):
        actions = {}

        # Generate actions for each agent based on their policy
        for agent in env.agents:
            agent_idx = env.agent_to_id[agent]
            policy_name = agent_policies[agent_idx]
            if env.active_agents[agent_idx] == 0:
                policy_func = do_nothing_policy
                policy_name = None
            else:
                policy_func = get_policy_function(policy_name)

            # Get agent's observation and action mask
            obs = observations[agent]["observation"]
            action_mask = observations[agent]["action_mask"]
            # Generate action using the agent's policy
            if policy_name == "careerist":
                action = policy_func(obs, action_mask, prestige_threshold)
            elif policy_name == "orthodox_scientist":
                action = policy_func(obs, action_mask, novelty_threshold)
            elif policy_name == "mass_producer":
                action = policy_func(obs, action_mask, effort_threshold)
            else:
                action = policy_func(obs, action_mask)

            actions[agent] = action

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # if step > 500:
        #     active_agent_1 = list(env.active_agents).index(1)
        #     print(env.action_masks[f"agent_{active_agent_1}"])
        if output_file_prefix not in ["sensitivity", "calibration"]:
            log.log_observation(
                {
                    a: obs if env.active_agents[env.agent_to_id[a]] == 1 else None
                    for a, obs in observations.items()
                }
            )
            log.log_action(
                {
                    a: (
                        act | {"archetype": agent_policies[env.agent_to_id[a]]}
                        if env.active_agents[env.agent_to_id[a]] == 1
                        else None
                    )
                    for a, act in actions.items()
                }
            )
        # Update stats
        stats.update(env, observations, rewards, terminations, truncations)

        # Print progress
        if step % 100 == 0:
            print(f"Step {step}: {stats.summary_line()}")

        # Check if all agents are done
        if all(terminations.values()):
            print(f"Simulation ended at step {step}")
            break

    if output_file_prefix not in ["sensitivity", "calibration"]:
        env.area.save(f"log/{output_file_prefix}_area.pickle")

    log.log_projects(env.projects.values())
    # Save results
    results = {
        "final_stats": stats.to_dict(),
        "agent_policies": agent_policies,
        "policy_distribution": policy_distribution
        or {"careerist": 1 / 3, "orthodox_scientist": 1 / 3, "mass_producer": 1 / 3},
    }

    if output_file_prefix not in ["sensitivity", "calibration"]:
        with open(output_file_prefix + "_summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nFinal Results:")
    print(f"Total Steps: {stats.total_steps}")
    print(f"Finished Projects: {stats.finished_projects_count}")
    print(f"Successful Projects: {stats.successful_projects_count}")
    print(
        f"Success Rate: {stats.successful_projects_count / max(stats.finished_projects_count, 1):.3f}"
    )
    print(f"Total Rewards: {stats.total_rewards_distributed:.2f}")

    return results


def compare_policy_performances():
    """Compare the performance of different policy distributions."""

    results = {}

    for config_name, policy_dist in POLICY_CONFIGS.items():
        print(f"\n{'='*50}")
        print(f"Testing: {config_name}")
        print(f"{'='*50}")

        result = run_simulation_with_policies(
            n_agents=2_000,
            start_agents=100,
            max_steps=5_000,
            n_groups=50,
            max_peer_group_size=100,
            policy_distribution=policy_dist,
            output_file_prefix=f"policy_{config_name.lower().replace(' ', '_')}",
        )

        results[config_name] = result["final_stats"]

    # Print comparison
    print(f"\n{'='*80}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*80}")

    for config_name, stats in results.items():
        success_rate = stats["successful_projects"] / max(stats["finished_projects"], 1)
        print(
            f"{config_name:20} | Success Rate: {success_rate:.3f} | "
            f"Finished: {stats['finished_projects']:3d} | "
            f"Rewards: {stats['total_rewards_distributed']:6.2f}"
        )


def run_all_reward_functions(seeds=range(10)):
    reward_functions = ["multiply", "evenly", "by_effort"]

    for seed in seeds:
        for reward_fn in reward_functions:
            print(
                f"Running simulation for reward function '{reward_fn}' with seed {seed}..."
            )

            run_simulation_with_policies(
                n_agents=2000,
                start_agents=200,
                max_steps=600,
                n_groups=20,
                max_peer_group_size=100,
                max_rewardless_steps=50,
                policy_distribution={
                    "careerist": 1 / 3,
                    "orthodox_scientist": 1 / 3,
                    "mass_producer": 1 / 3,
                },
                output_file_prefix=f"balanced_{reward_fn}_seed{seed}",
                group_policy_homogenous=False,
                acceptance_threshold=0.44,
                novelty_threshold=0.4,
                prestige_threshold=0.4,
                effort_threshold=38,
                seed=seed,
                reward_function=reward_fn,
            )

    print("All simulations completed.")


if __name__ == "__main__":
    # Run a single simulation with balanced policies
    run_simulation_with_policies(
        n_agents=2000,
        start_agents=200,
        max_steps=600,
        n_groups=20,
        max_peer_group_size=100,
        max_rewardless_steps=50,
        policy_distribution={
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        },
        output_file_prefix="balanced_multiply_seed42",
        group_policy_homogenous=False,
        acceptance_threshold=0.44,
        novelty_threshold=0.4,
        prestige_threshold=0.4,
        effort_threshold=38,
        seed=42,
        reward_function="multiply",
    )

    # uncomment to run simulation for all three reward function on 10 random seeds each (~10-15h)
    # run_all_reward_functions()
