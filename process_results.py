import json
import pandas as pd

def build_reward_dataframe(reward_steps, agents, seed):
    """
    Builds a DataFrame of accumulated rewards per agent per step,
    annotated with archetype.
    """
    agent_archetype = {}
    for a in agents:
        for k, v in a.items():
            if v is not None:
                agent_archetype[k] = v["archetype"]

    records = []
    for step_idx, step in enumerate(reward_steps):
        for agent_id, data in step.items():
            if data is not None:
                data = data.get("observation", None)
                if data and "accumulated_rewards" in data:
                    archetype = agent_archetype.get(agent_id, None)
                    if archetype is not None:
                        records.append({
                            "step": step_idx,
                            "archetype": archetype,
                            "agent_id": agent_id,
                            "accumulated_rewards": data["accumulated_rewards"][0],
                            "h_index": data["peer_h_index"][0],
                            "age": data["age"][0],
                            # "accumulated_citations": len(data.get("citations", [])),
                            # "societal_value": data['societal_value_score'],
                            "seed": seed,
                        })

    return pd.DataFrame(records)



def build_reward_summary_by_archetype(reward_steps, agents, seed, strategy):
    """
    Returns a DataFrame with rows:
      step, archetype, mean_reward, std_reward, n_agents, seed, strategy

    Raises:
        ValueError if any agent has no archetype mapping.
    """
    # === 1. Build agent_id -> archetype map ===
    agent_archetype = {}
    for a in agents:
        if not isinstance(a, dict):
            continue
        for agent_id, v in a.items():
            if v is None:
                continue
            if isinstance(v, dict) and "archetype" in v:
                agent_archetype[agent_id] = v["archetype"]

    if not agent_archetype:
        raise ValueError("No archetypes found in agents data.")

    # === 2. Build per-step stats grouped by archetype ===
    records = []
    for step_idx, step in enumerate(reward_steps):
        if not isinstance(step, dict):
            continue

        # Gather rewards per archetype for this step
        arch_rewards = {}
        for agent_id, data in step.items():
            if data is None:
                continue

            # Every agent MUST have an archetype
            if agent_id not in agent_archetype:
                raise ValueError(f"Missing archetype for agent_id '{agent_id}' at step {step_idx}")

            obs = data.get("observation", {}) if isinstance(data, dict) else {}
            if "accumulated_rewards" not in obs:
                continue

            reward = obs["accumulated_rewards"][0]
            archetype = agent_archetype[agent_id]
            arch_rewards.setdefault(archetype, []).append(reward)

        # Compute mean/std per archetype
        for archetype, rewards in arch_rewards.items():
            s = pd.Series(rewards)
            records.append({
                "step": step_idx,
                "archetype": archetype,
                "mean_reward": float(s.mean()),
                "std_reward": float(s.std(ddof=1)),  # sample std
                "n_agents": len(rewards),
                "seed": int(seed),
                "strategy": strategy,
            })

    return pd.DataFrame(records)

def main():
    dfs = {}
    dfs_all = {}
    for name in ["multiply", "evenly", "by_effort"]:
        all_summaries = []
        all_rewards = []
        for seed in range(10):
            with open(f"../log/balanced_{name}_seed{seed}_actions.jsonl", "r") as f:
                balanced_actions = [json.loads(line) for line in f]
            with open(f"../log/balanced_{name}_seed{seed}_observations.jsonl", "r") as f:
                balanced_observations = [json.loads(line) for line in f]

            df_all = build_reward_dataframe(balanced_observations, balanced_actions, seed)
            df_summary = build_reward_summary_by_archetype(
                balanced_observations, balanced_actions, seed, name
            )
            all_rewards.append(df_all)
            all_summaries.append(df_summary)
        df_all = pd.concat(all_rewards, ignore_index=True)
        df_summary_all = pd.concat(all_summaries, ignore_index=True)
        dfs[name] = df_summary_all
        df_summary_all.to_parquet(f"reward_summary_by_archetype_{name}.parquet", index=False)
        print(f"Saved {name} summary to reward_summary_by_archetype_{name}.parquet "
            f"({len(df_summary_all)} records across {len(all_summaries)} seeds).")
        dfs_all[name] = df_all
        df_all.to_parquet(f"reward_trajectories_{name}.parquet", index=False)
        print(f"Saved {name} simulation to reward_trajectories_{name}.parquet "
                f"({len(df_all)} records).")

if __name__=="main":
    main()