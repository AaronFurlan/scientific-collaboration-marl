"""
Agent Policy Functions for Peer Group Environment

This module contains three different agent policy functions that can be used
instead of random sampling in the peer group environment simulation.

1. careerist: Chooses projects with high potential reward
2. orthodox_scientist: Chooses projects with good fit
3. mass_producer: Chooses projects with low effort and short completion time
"""

from itertools import zip_longest
from typing import Any, Dict, List

import numpy as np
from env.area import Area


def _mask_allowed(mask_arr: Any, idx: int) -> bool:
    if mask_arr is None:
        return True
    try:
        return mask_arr[idx] > 0
    except Exception:
        return False


def _emergency_continue_any(running_projects: Dict[str, Any]) -> bool:
    """
    figure out if there is too much work to begin new projects.
    """
    for proj in running_projects.values():
        contributors = len(proj.get("contributors", []))
        time_left = proj.get("time_left")[0]
        required = proj.get("required_effort")[0]
        current = proj.get("current_effort")[0]
        if (time_left / contributors) < (required - current):
            return True
    return False


def _select_effort_closest_deadline_under_required(
    running_projects: Dict[str, Any], put_effort_mask: np.ndarray
) -> int:
    """
    choose the running project with smallest time_left that still needs effort
    """
    candidates: List[tuple] = []
    for slot_idx, proj in enumerate(running_projects.values()):
        time_left = proj["time_left"][0]
        if time_left <= 0:
            continue
        if proj["current_effort"][0] < proj["required_effort"][0] and _mask_allowed(
            put_effort_mask, slot_idx + 1
        ):
            candidates.append((slot_idx + 1, time_left))
    if candidates:
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    return 0


def _select_effort_best_fit_or_threshold(
    running_projects: Dict[str, Any],
    put_effort_mask: np.ndarray,
    threshold_ratio: float = 0.9,
) -> int:
    # If any project is above threshold, immediately work on it; else choose by best peer_fit
    candidates: List[tuple] = []
    for slot_idx, proj in enumerate(running_projects.values()):
        required = proj["required_effort"][0]
        threshold = required * threshold_ratio
        if (
            proj["current_effort"][0] > threshold
            and proj["current_effort"][0] <= required
            and _mask_allowed(put_effort_mask, slot_idx + 1)
        ):
            return slot_idx
        if _mask_allowed(put_effort_mask, slot_idx + 1):
            fit = float(np.sum(proj["peer_fit"])) if len(proj["peer_fit"]) > 0 else 0.0
            candidates.append((slot_idx + 1, fit))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return 0


def maximally_collaborative_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    **kwargs,
) -> Dict[str, Any]:
    collaborate_mask = action_mask.get("collaborate_with")
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )
    return {
        "choose_project": 1,
        "collaborate_with": collaborate_mask,
        "put_effort": put_effort,
    }


def careerist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    prestige_threshold: float = 0.6,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    current = observation.get("running_projects", {})
    if _emergency_continue_any(current):
        chosen_project = 0
    else:
        choose_project_mask = action_mask.get("choose_project")
        # print(project_opportunities)
        if len(project_opportunities) > 0:
            opp = project_opportunities[0]
            meets = (
                opp is not None
                and float(opp.get("prestige")[0]) >= prestige_threshold
                and _mask_allowed(choose_project_mask, 1)
            )
            chosen_project = 1 if meets else 0
        else:
            chosen_project = 0

    # Collaboration: active peers with above-average reputation
    peer_reputation = np.array(observation.get("peer_reputation", []), dtype=np.float32)
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")
    own_reputation = observation.get("accumulated_rewards")[0]
    avg_rep = (
        float(peer_reputation[peer_group_active == 1].mean())
        if peer_group_active.sum() > 0
        else 0.0
    )
    # collaborate only with higher than average reputation
    higher_reputation = peer_reputation >= own_reputation
    if higher_reputation.sum() >= 1:
        desired = ((peer_reputation >= avg_rep).astype(np.int8)) * (
            peer_group_active > 0
        ).astype(np.int8)
    # if highest reputation collaborate with all
    else:
        desired = (peer_group_active > 0).astype(np.int8)
    collaborate_with = (desired > 0) & (collaborate_mask > 0)
    collaborate_with = collaborate_with.astype(np.int8)

    # Effort: project closest to deadline still under required_effort
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def orthodox_scientist_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    novelty_threshold: float = 0.4,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    current = observation.get("running_projects", {})
    if _emergency_continue_any(current):
        chosen_project = 0
    else:
        choose_project_mask = action_mask.get("choose_project")
        if len(project_opportunities) > 0:
            opp = project_opportunities[0]
            meets = (
                opp is not None
                and float(opp.get("novelty")[0]) <= novelty_threshold
                and _mask_allowed(choose_project_mask, 1)
            )
            chosen_project = 1 if meets else 0
        else:
            chosen_project = 0

    # Collaboration: only collaborate with agents who are close in topic centroid
    peer_group_active = np.array(observation.get("peer_group"), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")

    # Get agent's own topic centroid
    own_centroid = np.array(observation.get("self_centroid")[0])
    peer_centroids = np.array(observation.get("peer_centroids"))

    # Calculate distances to all peers
    if len(peer_centroids) > 0 and len(own_centroid) > 0:
        distances = Area.distance(own_centroid, peer_centroids)
        # Collaborate only with peers closer than average distance
        avg_distance = np.mean(distances)
        close_peers = (distances < avg_distance) & (peer_group_active > 0)
        collaborate_with = (close_peers & (collaborate_mask > 0)).astype(np.int8)
    else:
        # Fallback: collaborate with all active peers if centroids not available
        collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
            np.int8
        )

    # Effort: best fitting active project or above 90% threshold
    running_projects = observation.get("running_projects", {})
    put_effort_mask = action_mask.get("put_effort")
    put_effort = _select_effort_best_fit_or_threshold(
        running_projects, put_effort_mask, threshold_ratio=0.9
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def mass_producer_policy(
    observation: Dict[str, Any],
    action_mask: Dict[str, np.ndarray],
    effort_threshold: 22,
    **kwargs,
) -> Dict[str, Any]:

    project_opportunities = list(observation.get("project_opportunities").values())
    # Efficiency: prestige / (effort * time). With binary space, accept if efficiency > 0 and allowed
    choose_project_mask = action_mask.get("choose_project")
    if len(project_opportunities) > 0:
        opp = project_opportunities[0]
        if opp.get("required_effort")[0] <= effort_threshold:
            chosen_project = 1
        else:
            chosen_project = 0
        # if opp is not None and _mask_allowed(choose_project_mask, 1):
        #     effort = float(opp.get("required_effort")[0])
        #     time_w = float(opp.get("time_window")[0])
        #     prestige = float(opp.get("prestige")[0])
        #     eff = prestige / (effort * time_w) if effort > 0 and time_w > 0 else 0.0
        #     chosen_project = 1 if eff > 0 else 0
        # else:
        #     chosen_project = 0
    else:
        chosen_project = 0

    # Collaborate with all active peers within mask
    peer_group_active = np.array(observation.get("peer_group", []), dtype=np.int8)
    collaborate_mask = action_mask.get("collaborate_with")
    collaborate_with = ((peer_group_active > 0) & (collaborate_mask > 0)).astype(
        np.int8
    )
    # if sum(collaborate_with) == 0:
    #     breakpoint()

    # Effort: project closest to deadline under required_effort
    put_effort_mask = action_mask.get("put_effort")
    running_projects = observation.get("running_projects", {})
    put_effort = _select_effort_closest_deadline_under_required(
        running_projects, put_effort_mask
    )

    return {
        "choose_project": chosen_project,
        "collaborate_with": collaborate_with,
        "put_effort": put_effort,
    }


def do_nothing_policy(_: Dict[str, Any], action_mask: Dict[str, np.ndarray]):
    return {
        "choose_project": 0,
        "collaborate_with": np.zeros_like(action_mask["collaborate_with"]),
        "put_effort": 0,
    }


def interleave(lists):
    return [elem for group in zip_longest(*lists) for elem in group if elem is not None]


def get_policy_function(policy_name: str):
    policies = {
        "careerist": careerist_policy,
        "orthodox_scientist": orthodox_scientist_policy,
        "mass_producer": mass_producer_policy,
        "maximally_collaborative": maximally_collaborative_policy,
    }
    if policy_name not in policies:
        raise ValueError(
            f"Unknown policy: {policy_name}. Available policies: {list(policies.keys())}"
        )
    return policies[policy_name]


def create_mixed_policy_population(
    n_agents: int, policy_distribution: Dict[str, float] = None, seed=None
) -> List[str]:
    if seed is not None:
        np.random.seed(seed)
    if policy_distribution is None:
        policy_distribution = {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        }
    total_proportion = sum(policy_distribution.values())
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")
    agent_policies = []
    for policy_name, proportion in policy_distribution.items():
        n_policy_agents = int(n_agents * proportion)
        agent_policies.extend([policy_name] * n_policy_agents)
    while len(agent_policies) < n_agents:
        agent_policies.append(list(policy_distribution.keys())[0])
    np.random.shuffle(agent_policies)
    return agent_policies


def create_per_group_policy_population(
    n_agents: int, policy_distribution: Dict[str, float] = None
) -> List[str]:
    if policy_distribution is None:
        policy_distribution = {
            "careerist": 1 / 3,
            "orthodox_scientist": 1 / 3,
            "mass_producer": 1 / 3,
        }
    total_proportion = sum(policy_distribution.values())
    if abs(total_proportion - 1.0) > 1e-6:
        raise ValueError(f"Policy distribution must sum to 1.0, got {total_proportion}")
    policy_groups = []
    for policy_name, proportion in policy_distribution.items():
        if proportion > 0:
            n_policy_agents = int(n_agents * proportion)
            policy_groups.append([policy_name] * n_policy_agents)
    while sum([len(group) for group in policy_groups]) < n_agents:
        policy_groups[-1].append(list(policy_distribution.keys())[0])
    return interleave(policy_groups)


if __name__ == "__main__":
    # Keep minimal manual check without noisy prints
    print(create_per_group_policy_population(10))
