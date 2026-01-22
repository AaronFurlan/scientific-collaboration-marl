import functools
from collections import Counter
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Discrete, MultiBinary
from pettingzoo import ParallelEnv
from scipy.special import softmax

from .area import Area
from .project import Project
from .utils import GaussianMixture, sigmoid


class PeerGroupEnvironment(ParallelEnv):
    """Multi-agent environment with peer groups and project opportunities."""

    metadata = {
        "name": "peer_group_environment_v0",
    }

    def __init__(
        self,
        start_agents: int = 20,
        max_agents: int = 80,
        max_steps: int = 600,
        max_peer_group_size: int = 60,
        n_groups: int = 4,
        n_projects_per_step: int = 1,
        max_projects_per_agent: int = 6,
        max_agent_age: int = 1000,
        max_rewardless_steps: int = 50,
        growth_rate: float = 0.04,
        acceptance_threshold: float = 0.5,
        reward_mode: str = "multiply",
        render_mode: Optional[str] = None,
    ) -> None:
        self.n_agents: int = max_agents
        self.n_steps: int = max_steps
        self.starting_population_size: int = start_agents
        self.n_groups: int = n_groups
        self.max_peer_group_size: int = max_peer_group_size
        self.n_projects_per_step: int = n_projects_per_step
        self.max_projects_per_agent: int = max_projects_per_agent
        self.max_agent_age: int = max_agent_age
        self.age_distribution = GaussianMixture(
            weights=[0.5, 0.5],
            mus=[52, self.max_agent_age],
            sds=[52, self.max_agent_age / 4],
            rng=np.random,
        )
        self.reward_function_name = reward_mode
        self.max_rewardless_steps: int = max_rewardless_steps
        self.growth_rate: float = growth_rate
        self.acceptance_threshold: float = acceptance_threshold
        self.render_mode: Optional[str] = render_mode

        self.possible_agents: List[str] = [f"agent_{i}" for i in range(self.n_agents)]
        self.agent_to_id: Dict[str, int] = {
            agent: i for i, agent in enumerate(self.possible_agents)
        }
        # Cache for action and observation spaces
        self._space_cache: Dict[str, Any] = {}
        self._peer_groups_changed: bool = False
        self.timestep: int = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_ages = self.age_distribution.sample(self.n_agents)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_rewards = np.zeros(
            (self.n_agents, self.n_steps + 1), dtype=np.float32
        )
        self.agent_completed_projects = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_successful_projects: List[List[str]] = [
            [] for _ in range(self.n_agents)
        ]
        self.agent_active_projects: List[List[Optional[str]]] = [
            [None for _ in range(self.max_projects_per_agent)]
            for _ in range(self.n_agents)
        ]  # List of project indices
        self.agent_project_effort: List[Dict[str, float]] = [
            {} for _ in range(self.n_agents)
        ]  # {project_idx: effort}
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.action_masks: Dict[str, Dict[str, Any]] = {}
        self.rewards: Dict[str, float] = {}
        self.agents: List[str] = []
        self.active_agents = np.zeros(self.n_agents, dtype=np.int8)
        self.terminated_agents = np.zeros(self.n_agents, dtype=np.int8)

        # Will be initialized in _init_peer_groups
        self.peer_groups: List[List[int]] = [[] for _ in range(n_groups)]
        self.agent_peer_idx: List[int] = []

        # Will be initialized in _generate_projects
        self.open_projects: List[Dict[str, Any]] = []
        # The chosen projects will be added here
        self.projects: Dict[str, Project] = {}
        # the ared in which the projects will be situated
        self.area: Area = None
        self.distances = []
        self.global_density = 1
        self.project_templates: List[Dict[str, Any]] = [
            # low novelty, low effort, low prestige
            {
                "required_effort": 20,
                "prestige": 0.5,
                "novelty": 0.2,
            },
        ]

    def _init_project_topic_plane(self, n_gaussians=40) -> None:
        self.area = Area(xlim=(-1, 1), ylim=(-1, 1))

        # Add Gaussian areas
        for i in range(n_gaussians):
            value = 1.0 if i % 2 == 0 else -1.0
            self.area.add_gaussian_area(
                *self.area.random_gaussian_point(), sigma=0.05, value=value
            )

    def _init_peer_groups(self) -> None:
        if self.n_agents < self.n_groups:
            raise ValueError(
                f"{self.n_agents} agents can not be distributed into {self.n_groups}!"
            )
        if self.max_peer_group_size > self.n_agents:
            raise ValueError(f"Peer_group_size can't be bigger than number of agents!")
        # n_groups = self.n_agents // self.peer_group_size
        self.peer_groups = [[] for _ in range(self.n_groups)]
        self.peer_group_centroids = [
            self.area.random_gaussian_point() for _ in range(self.n_groups)
        ]
        for i in range(self.n_agents):
            self.peer_groups[i % self.n_groups].append(i)
        # Each agent has a fixed set of peers (not necessarily symmetric)
        self.agent_peer_idx = []  # List of sets of peer agent ids for each agent
        for i in range(self.n_agents):
            # Peers are the all agents in the same peer group
            self.agent_peer_idx.append(i % self.n_groups)

    def _connect_peer_groups(self) -> None:
        # Pick two random groups.
        if len(self.peer_groups) % 2 != 0:
            raise ValueError(f"Peer groups must be even, found {len(self.peer_groups)}")

        group_pairs = list(
            np.random.permutation(
                np.array(list(enumerate(self.peer_groups)), dtype=object)
            )
        )
        for i in range(0, len(group_pairs), 2):
            group_idx1, group1 = group_pairs[i]
            group_idx2, group2 = group_pairs[i + 1]
            if (
                len(group1) >= self.max_peer_group_size
                or len(group2) >= self.max_peer_group_size
            ):
                continue
            # Pick a random agent from each group which isn't already in the other group.
            group1 = set(group1)
            group2 = set(group2)
            if len(group1 - group2) == 0 or len(group2 - group1) == 0:
                continue
            try:
                active_only_group1 = self.active_agents[list(group1 - group2)]
                active_only_group2 = self.active_agents[list(group2 - group1)]
                agent_idx1 = np.random.choice(active_only_group1)
                agent_idx2 = np.random.choice(active_only_group2)

            except ValueError:
                print(
                    f"Warning: Groups {group_idx1} and {group_idx2} couldn't be connected because all members already know each other."
                )

            # Add agent 1 to agents 2's peer group and vice versa.
            if agent_idx2 not in self.peer_groups[group_idx1]:
                self.peer_groups[group_idx1].append(agent_idx2)
            if agent_idx1 not in self.peer_groups[group_idx2]:
                self.peer_groups[group_idx2].append(agent_idx1)

    def _activate_agent(self, group_idx: int):
        group = self.peer_groups[group_idx]
        active_in_group = self.active_agents[group]
        for agent_i, active in zip(group, active_in_group):
            if active == 0 and self.terminated_agents[agent_i] == 0:
                self.active_agents[agent_i] = 1
                self.agent_rewards[agent_i, self.timestep :] = 0
                return agent_i
        return None

    def _generate_projects(self) -> None:
        self.open_projects = self.project_templates.copy()
        for i, project in enumerate(self.open_projects):
            project["required_effort"] = max(
                1, int(np.random.gumbel(project["required_effort"], 10))
            )
            project["prestige"] = np.clip(
                np.random.normal(project["prestige"], 0.15), 0.1, 1
            )
            project["novelty"] = np.clip(
                np.random.gumbel(project["novelty"], 0.05), 0.1, 1
            )

            project["time_window"] = np.ceil(
                project["required_effort"] * np.random.uniform(0.8, 2)
            )
            project["current_effort"] = 0
            project["contributors"] = []
            project["start_time"] = 0
            project["finished"] = False

    def _get_active_projects(self, agent: int) -> List[str]:
        return [p for p in self.agent_active_projects[agent] if p is not None]

    def _add_active_project(self, agent: int, proj_id: str) -> int:
        for i, slot in enumerate(self.agent_active_projects[agent]):
            if slot is None:
                self.agent_active_projects[agent][i] = proj_id
                return i
        return -1

    def _remove_active_project(self, agent: int, proj_id: str) -> None:
        try:
            idx = self.agent_active_projects[agent].index(proj_id)
            self.agent_active_projects[agent][idx] = None
        except ValueError:
            pass

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            np.random.seed(seed)
            Area.seed(seed)
            Project.seed(seed)

        # Reset state variables
        self.timestep = 0
        self.agent_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_ages = self.age_distribution.sample(self.n_agents)
        self.rewardless_steps = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_h_indexes = np.zeros(self.n_agents, dtype=np.int16)    
        self.agent_rewards = np.zeros(
            (self.n_agents, self.n_steps + 1), dtype=np.float32
        )
        self.agent_rewards.fill(np.nan)
        self.agent_rewards[: self.starting_population_size, :] = 0
        self.agent_completed_projects = np.zeros(self.n_agents, dtype=np.int32)
        self.agent_successful_projects = [[] for _ in range(self.n_agents)]
        self.agent_active_projects = [
            [None for _ in range(self.max_projects_per_agent)]
            for _ in range(self.n_agents)
        ]
        self.agent_project_effort = [{} for _ in range(self.n_agents)]
        self.active_agents = np.zeros(self.n_agents, dtype=np.int8)
        # activate a subset of agents equal to the starting population size
        self.active_agents[: self.starting_population_size] = 1
        self.terminated_agents = np.zeros(self.n_agents, dtype=np.int8)
        self.distances = []
        self.global_density = 1

        # Reinitialize core structures
        self._init_project_topic_plane()
        self._init_peer_groups()
        self._generate_projects()
        self.projects = {}
        self.agents = copy(self.possible_agents)
        self.actions = {}
        self.observations = {}
        self.action_masks = {}
        self.rewards = {}
        # Clear cache and return initial observations
        self._clear_space_cache()
        observations = {}
        for agent in self.agents:
            obs = self._get_observation(agent)
            mask = self._get_action_mask(agent)
            self.observations[agent] = obs
            self.action_masks[agent] = mask
            observations[agent] = {"observation": obs, "action_mask": mask}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def _clear_space_cache(self) -> None:
        """Clear the cached action and observation spaces."""
        self._space_cache.clear()
        self._peer_groups_changed = False

    def _get_action_mask(self, agent: str) -> Dict[str, np.ndarray]:
        idx = self.agent_to_id[agent]
        ## this agent dropped out
        if self.active_agents[idx] == 0:
            return {
                "choose_project": np.zeros(self.n_projects_per_step + 1, dtype=np.int8),
                "collaborate_with": np.zeros(self.max_peer_group_size, dtype=np.int8),
                "put_effort": np.zeros(self.max_projects_per_agent + 1, dtype=np.int8),
            }
        mask = {}
        # Project selection: can only select if under max_projects_per_agent
        can_choose = int(
            len(self._get_active_projects(idx)) < self.max_projects_per_agent
        )
        mask["choose_project"] = np.zeros(self.n_projects_per_step + 1, dtype=np.int8)
        if can_choose:
            mask["choose_project"][:] = 1
        else:
            mask["choose_project"][:] = 0
            mask["choose_project"][0] = 1  # Only 'no project' allowed

        if len(mask["choose_project"]) == 6:
            p = [1 / 12, 1 / 12, 1 / 12, 1 / 4, 1 / 4, 1 / 4]
        else:
            p = None
        # high reward is rarer
        not_choosable_this_time = np.random.choice(
            list(range(1, len(mask["choose_project"]))),
            np.random.randint(0, len(mask["choose_project"])),
            replace=False,
            p=p,
        )
        mask["choose_project"][not_choosable_this_time] = 0
        # Peer collaboration: MultiBinary for peer group
        peer_group = self.peer_groups[self.agent_peer_idx[idx]]
        mask["collaborate_with"] = np.zeros(self.max_peer_group_size, dtype=np.int8)
        mask["collaborate_with"][: len(peer_group)] = np.where(
            self.active_agents[peer_group].astype(bool),
            2,  ## if active unmask
            mask["collaborate_with"][: len(peer_group)],  # else keep 0
        )
        if sum(mask["collaborate_with"]) == 0:
            breakpoint()

        # Effort: can only put effort into active projects
        mask["put_effort"] = np.zeros(self.max_projects_per_agent + 1, dtype=np.int8)
        mask["put_effort"][0] = 1  # no effort always possible
        for i, p_idx in enumerate(self.agent_active_projects[idx]):
            if p_idx is not None:
                mask["put_effort"][i + 1] = 1
        return mask

    def _find_project_setting(
        self, project_idx: int, peer_group: np.ndarray, intents: np.ndarray
    ) -> List[Tuple[str, List[int]]]:
        # print(peer_group)
        # print(intents)
        if len(peer_group) == 0:
            return []
        ## no collaboration
        elif not np.any(intents):
            new_projects = []
            for agent in peer_group:
                running_project_idx = self._start_open_project(project_idx, [agent])
                if running_project_idx is not None:
                    new_projects.append((running_project_idx, [agent]))
            return new_projects
        else:
            ## Find the biggest overlap of collaborators on the same project as the largest clique in the collaboration graph
            grouped_collaborators = set()
            running_project_idx = None
            for collaborators in sorted(
                list(nx.find_cliques(nx.from_numpy_array(intents))),
                key=len,
                reverse=True,
            ):
                already_at_limit = set(
                    [
                        c
                        for c in collaborators
                        if len(self._get_active_projects(peer_group[c]))
                        >= self.max_projects_per_agent
                    ]
                )
                intents[:, list(already_at_limit)] = 0
                intents[list(already_at_limit), :] = 0
                collaborators = list(set(collaborators) - already_at_limit)

                if len(collaborators) > 1:
                    running_project_idx = self._start_open_project(
                        project_idx, peer_group[collaborators]
                    )
                    grouped_collaborators |= set(collaborators)

    def _start_open_project(
        self, project_idx: int, contributors: List[int]
    ) -> Optional[str]:
        project_id = f"project_{len(self.projects)}-{project_idx}-{self.timestep}"
        suffix = [str(project_idx), str(self.timestep)]
        ## if the project was already added make sure all the contributors are there.
        for contributor in contributors:
            for proj in self._get_active_projects(contributor):
                # the project was already started in the same timestep by this agent
                if proj.split("-")[-2:] == suffix:
                    # ignore
                    if len(contributors) <= len(self.projects[proj].contributors):
                        return None
                    # reconfigure the project
                    else:
                        project_id = proj

        new_running_proj = deepcopy(self.open_projects[project_idx])
        new_running_proj["id"] = project_id
        new_running_proj["start_time"] = self.timestep
        new_running_proj["contributors"] = contributors
        new_running_proj["peer_fit"] = {i: 0 for i in contributors}
        for contributor in contributors:
            self._add_active_project(contributor, new_running_proj["id"])
            self.agent_project_effort[contributor][new_running_proj["id"]] = 0
        proj_object = Project.from_dict(new_running_proj)
        proj_object.kene = self._locate_project_in_plane(proj_object)
        proj_object.peer_fit = [
            self._determine_agent_fit(proj_object, agent_i)
            for agent_i in proj_object.contributors
        ]
        self.projects[new_running_proj["id"]] = proj_object
        return project_id

    def _locate_project_in_plane(self, new_project: Project) -> np.array:
        """
        The project must only be located once (side effects)
        """
        # select a random generator project from all authors in peer group
        all_contributors_projects = []
        probabilities = []
        max_reputation = (
            np.max(
                [
                    np.nansum(self.agent_rewards[agent_i, :])
                    for agent_i in new_project.contributors
                ]
            )
            or 1
        )
        # choose weighted by contributor reputation
        for agent_i in new_project.contributors:
            all_contributors_projects += self.agent_successful_projects[agent_i]
            probabilities += [
                0.5 * self.projects[pid].societal_value_score
                + 0.5 * np.nansum(self.agent_rewards[agent_i, :]) / max_reputation
                for pid in self.agent_successful_projects[agent_i]
            ]
        generator_project = None
        if len(all_contributors_projects) > 0:
            probabilities = softmax(probabilities)
            generator_project = self.projects[
                np.random.choice(all_contributors_projects, p=probabilities)
            ]
            new_kene = generator_project.kene
            new_project.generator_project_id = generator_project.project_id
            all_contributors_projects = [
                p
                for p in all_contributors_projects
                if p != generator_project.project_id
            ]
        else:
            peer_group_idx = self.agent_peer_idx[new_project.contributors[0]]
            new_kene = self.peer_group_centroids[peer_group_idx]

        # select 10-20 projects as citation which are in the area of novelty around the generator
        all_project_kenes = np.array(
            [
                self.projects[project_id].kene
                for project_id in all_contributors_projects
                if project_id != generator_project.project_id
            ]
        )

        # no projects
        if len(all_contributors_projects) == 0:
            new_project.citations = []
            new_kene = new_kene + np.random.normal(0, new_project.novelty / 2, 2)
            new_kene = np.tanh(new_kene)
            return new_kene

        projects_in_vicinity = np.array(all_contributors_projects)[
            Area.distance(all_project_kenes, new_kene) <= new_project.novelty
        ]
        # no projects in vicinity
        if len(projects_in_vicinity) == 0:
            new_project.citations = []
            new_kene = new_kene + np.random.normal(0, new_project.novelty / 2, 2)
            new_kene = np.tanh(new_kene)
            return new_kene

        n_cited = min(len(projects_in_vicinity), np.random.randint(10, 21))
        n_citations_vicinity = np.array(
            [len(self.projects[p].citations) + 1 for p in projects_in_vicinity]
        ).sum()
        sv_exp = np.exp(
            [self.projects[p].societal_value_score for p in projects_in_vicinity]
        )
        societal_value_vicinity_probs = sv_exp / sv_exp.sum()
        citation_popularity = [
            0.6 * ((len(self.projects[p].citations) + 1) / n_citations_vicinity)
            + 0.4 * psv
            for p, psv in zip(projects_in_vicinity, societal_value_vicinity_probs)
        ]
        # weighted by n citations?
        cited_projects = np.random.choice(
            projects_in_vicinity, n_cited, p=citation_popularity
        )
        m = 0
        for cited_project in cited_projects:
            cited_project = self.projects[cited_project]
            cited_project.cited_by.append(new_project.project_id)
            cited_position = cited_project.kene
            m += np.random.uniform(0, 0.1)
            new_kene += (new_kene - cited_position) * (1 - m) / 2
        new_project.citations = cited_projects
        new_kene = np.tanh(new_kene)
        return new_kene

    def _determine_agent_fit(self, project: Project, agent_i: int) -> float:
        if len(self.agent_successful_projects[agent_i]) > 0:
            agent_centroid = np.array(
                [
                    self.projects[p_idx].kene
                    for p_idx in self.agent_successful_projects[agent_i]
                ]
            ).mean(axis=0)
        else:
            return 0.5
        max_dist = np.sqrt(
            (self.area.ylim[0] - self.area.ylim[1]) ** 2
            + (self.area.xlim[0] - self.area.xlim[1]) ** 2
        )
        return 1 - (np.linalg.norm(project.kene - agent_centroid) / max_dist)

    def _update_distances(self, distances_to_previous, distances_between):
        if len(self.distances) > 0:
            self.distances = np.block(
                [
                    [self.distances, np.array(distances_to_previous).T],
                    [np.array(distances_to_previous), np.array(distances_between)],
                ]
            )
        else:
            self.distances = np.array(distances_between)
        return self.distances

    def _distribute_rewards_evenly(self, p, reward):
        for idx in p.contributors:
            self._remove_active_project(idx, p.project_id)
            if reward > 0:
                self.agent_successful_projects[idx].append(p.project_id)
            self.agent_rewards[idx, self.timestep] += reward / len(p.contributors)
            self.agent_completed_projects[idx] += 1
            self.rewards[f"agent_{idx}"] += reward / len(p.contributors)

    def _distribute_rewards_multiply(self, p, reward):
        for idx in p.contributors:
            self._remove_active_project(idx, p.project_id)
            if reward > 0:
                self.agent_successful_projects[idx].append(p.project_id)
            self.agent_rewards[idx, self.timestep] += reward
            self.agent_completed_projects[idx] += 1
            self.rewards[f"agent_{idx}"] += reward

    def _distribute_rewards_by_effort(self, p, reward):
        max_effort = max(
            [self.agent_project_effort[c][p.project_id] for c in p.contributors]
        )
        for idx in p.contributors:
            effort = self.agent_project_effort[idx][p.project_id]
            rel_effort = (
                effort / max_effort if max_effort > 0 else 1 / len(p.contributors)
            )
            self._remove_active_project(idx, p.project_id)
            if reward > 0:
                self.agent_successful_projects[idx].append(p.project_id)
            self.agent_rewards[idx, self.timestep] += reward * rel_effort
            self.agent_completed_projects[idx] += 1
            self.rewards[f"agent_{idx}"] += reward * rel_effort

    def h_index(self, a) -> float:
        cite_counts = []
        for p in self.agent_successful_projects[a]:
            citations = len(self.projects[p].cited_by)
            cite_counts.append(citations)

        cite_counts.sort(reverse=True)
        if len(cite_counts) > 0:
            for i, nc in enumerate(cite_counts):
                if nc < i + 1:
                    return i
            return len(cite_counts)
        return 0

    def step(self,   actions: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self.actions = actions
        self.timestep += 1

        # Track which open projects are selected to be started this step
        agent_project_choices: Dict[int] = {}
        for agent, action in actions.items():
            idx = self.agent_to_id[agent]
            chosen_project = action["choose_project"]

            if (
                chosen_project > 0
                and len(self._get_active_projects(idx)) < self.max_projects_per_agent
            ):
                open_proj_idx = chosen_project - 1
                if open_proj_idx < len(self.open_projects):
                    agent_project_choices[idx] = open_proj_idx

            # Apply effort to running projects
            if (
                action["put_effort"] > 0
                and len(self._get_active_projects(idx)) >= action["put_effort"]
            ):

                selected_project = action["put_effort"] - 1
                effort_project_id = self.agent_active_projects[idx][selected_project]
                if effort_project_id is not None:
                    effort_project = self.projects[effort_project_id]
                    contributors_idx = (
                        list(effort_project.contributors).index(idx)
                        if idx in effort_project.contributors
                        else None
                    )
                    if contributors_idx is not None:
                        effort_amount = effort_project.peer_fit[contributors_idx]
                    else:
                        effort_amount = 0

                    self.projects[effort_project_id].add_effort(effort_amount)
                    self.agent_project_effort[idx][effort_project_id] += effort_amount
                else:
                    print(f"Couldn't find project: {selected_project}")
        # Collaboration intents (for each agent, with their peers)
        for peer_group in self.peer_groups:
            peer_group = np.array((peer_group))
            peer_group_choices: List[Optional[int]] = [
                agent_project_choices.get(pc_idx, None) for pc_idx in peer_group
            ]
            peer_group_intents = np.array(
                [
                    actions[f"agent_{gm_idx}"]["collaborate_with"]
                    for gm_idx in peer_group
                ]
            )

            np.fill_diagonal(peer_group_intents, 0)  # no self collaboration

            for choice, _ in Counter(peer_group_choices).most_common():
                if choice is not None:
                    # get all collaborators which took this choice
                    potential_collaborators = np.where(
                        np.array(peer_group_choices) == choice
                    )[0]
                    collaborator_group = peer_group[potential_collaborators]
                    # find overlaps in collaboration intents of collaborators
                    collaborators_intents = peer_group_intents[
                        np.ix_(potential_collaborators, potential_collaborators)
                    ]
                    ## if not enough peers chose the project to form the group don't start
                    # Only keep edges where both i→j and j→i exist
                    collaborators_intents = (
                        collaborators_intents & collaborators_intents.T
                    )
                    not_enough_collaborators = np.sum(collaborators_intents, axis=0)
                    not_enough_collaborators = not_enough_collaborators[
                        not_enough_collaborators < len(collaborator_group)
                    ]
                    collaborators_intents = np.delete(
                        collaborators_intents, not_enough_collaborators, axis=0
                    )
                    collaborators_intents = np.delete(
                        collaborators_intents, not_enough_collaborators, axis=1
                    )
                    collaborator_group = np.delete(
                        collaborator_group, not_enough_collaborators
                    )

                    self._find_project_setting(
                        choice, collaborator_group, collaborators_intents
                    )

        # Check project completion and assign rewards
        self.rewards = {a: 0.0 for a in self.agents}
        published_projects = [
            p for p in self.projects.values() if p.finished and p.final_reward > 0
        ]
        due_projects = [
            p
            for p in self.projects.values()
            if p.is_due(self.timestep) and p.finished is False
        ]

        if len(published_projects) > 1 and self.timestep % 10 == 0:
            # sample up to 1000 projects and take the average distance to their 10 closest neighbors
            if len(self.distances) > 1000:
                idx = np.random.choice(self.distances.shape[0], 1000, replace=False)
                distance_sample = self.distances[np.ix_(idx, idx)]
            else:
                distance_sample = self.distances
            self.global_density = np.mean(
                np.sum(
                    np.sort(distance_sample, axis=1)[
                        :, 1 : min(10, len(published_projects))
                    ],
                    axis=1,
                )
            )

        new_distances = []
        for p in due_projects:
            if len(published_projects) > 0:
                distances = Area.distance(
                    [pp.kene for pp in published_projects], p.kene
                )
                local_density = np.sum(
                    np.sort(distances)[: min(10, len(published_projects))]
                )
            else:
                distances = None
                local_density = 1
            quality = p.calculate_quality(
                topic_area=self.area,
                relative_density=sigmoid(local_density - self.global_density),
                noise_factor=0.1,
            )
            quality = np.clip(quality, 0, 1)

            reward = p.calculate_reward(
                quality, threshold=self.acceptance_threshold, noise_factor=0.2
            )

            if reward > 0 and distances is not None:
                new_distances.append(distances)
                # update hindex
                for citedp in p.citations:
                    for c in self.projects[citedp].contributors:
                        self.agent_h_indexes[c] = self.h_index(c)

            if self.reward_function_name == "multiply":
                self._distribute_rewards_multiply(p, reward)
            elif self.reward_function_name == "evenly":
                self._distribute_rewards_evenly(p, reward)
            elif self.reward_function_name == "by_effort":
                self._distribute_rewards_by_effort(p, reward)

            p.finished = True

        new_projects = [p for p in due_projects if p.final_reward > 0]
        if len(new_projects) > 0:
            self._update_distances(
                distances_to_previous=new_distances,
                distances_between=[
                    Area.distance([pp.kene for pp in new_projects], p.kene)
                    for p in new_projects
                ],
            )

        in_window_rewards = self.agent_rewards[
            self.active_agents.astype(bool),
            max(0, self.timestep - self.max_rewardless_steps) : self.timestep,
        ]
        per_step_reward_mean = np.nanmean(in_window_rewards, axis=1)
        cutoff = np.percentile(per_step_reward_mean, 25)
        # Update agent ages, steps, rewardless steps
        active_idx = 0
        for idx, agent in enumerate(self.agents):
            if self.active_agents[idx] == 1:
                self.agent_steps[idx] += 1
                if (
                    self.rewards[agent] > 0
                    and per_step_reward_mean[active_idx] > cutoff
                ):
                    self.rewardless_steps[idx] = 0
                else:
                    self.rewardless_steps[idx] += 1
                active_idx += 1

        truncations = {a: False for a in self.agents}
        # Drop agents randomly way more likely with too many rewardless steps or max timesteps

        terminations = {}
        for a in self.agents:
            idx = self.agent_to_id[a]
            if self.active_agents[idx] != 1:
                terminations[a] = False
                continue

            # Probability of termination becomes higher nearing the max value
            rewardless_dist = self.rewardless_steps[idx] - self.max_rewardless_steps
            age_dist = self.agent_steps[idx] - self.agent_ages[idx]
            rewardless_prob = sigmoid(
                rewardless_dist,
                midpoint=self.max_rewardless_steps * 0.5,
                sharpness=1 / (self.max_rewardless_steps * 0.0625),
            )
            age_prob = sigmoid(
                age_dist,
                midpoint=self.agent_ages[idx] * 0.5,
                # TODO: set this sharpness as parameter?
                sharpness=1 / (self.agent_ages[idx] * 0.0625),
            )

            termination_prob = np.mean([rewardless_prob, age_prob])
            # Stochastic decision
            terminations[a] = np.random.rand() < termination_prob

        self.terminated_agents = self.terminated_agents | np.fromiter(
            terminations.values(), dtype=bool
        )
        agents_activated_in_step = []
        # connect peer groups
        if self.timestep % 50 == 0:
            self._connect_peer_groups()

        for agent_id, terminated in terminations.items():
            agent_id = self.agent_to_id[agent_id]
            # replace
            if terminated:
                self.active_agents[agent_id] = 0
                group = self.agent_peer_idx[agent_id]
                agents_activated_in_step.append(self._activate_agent(group))
        # grow active agents
        if self.growth_rate < 1:
            if self.timestep % (1 // self.growth_rate) == 0 and np.random.rand() < 0.7:
                # TODO: choice weighted by success?
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))
        else:
            each_step = np.floor(self.growth_rate)
            for _ in range(each_step):
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))
            if self.timestep % (1 // (self.growth_rate - each_step)) == 0:
                group = np.random.choice(range(self.n_groups))
                agents_activated_in_step.append(self._activate_agent(group))

        # if len(agents_activated_in_step) > 0:
        #     print(
        #         f"Activated {len(agents_activated_in_step)} agents in step {self.timestep}"
        #     )

        if not all([a is not None for a in agents_activated_in_step]):
            print("No more agents to activate!")
        # Prepare next obs/mask
        observations = {}
        for agent in self.agents:
            obs = self._get_observation(agent)
            mask = self._get_action_mask(agent)
            self.observations[agent] = obs
            self.action_masks[agent] = mask
            observations[agent] = {"observation": obs, "action_mask": mask}
        infos = {a: {} for a in self.agents}
        return observations, self.rewards, terminations, truncations, infos

    def _get_observation(self, agent: str) -> Dict[str, Any]:
        idx = self.agent_to_id[agent]
        # Peer group: array of peer agent ids
        peer_group = np.array(
            self.peer_groups[self.agent_peer_idx[idx]], dtype=np.int32
        )
        peer_group_obs = np.zeros(self.max_peer_group_size, dtype=np.int8)
        peer_reputation = np.zeros((self.max_peer_group_size), dtype=np.float32)
        # peer_h_indexes = np.zeros((self.max_peer_group_size), dtype=np.int16)
        peer_centroids = np.zeros((self.max_peer_group_size, 2), dtype=np.float64)
        self_centroid = np.array([0, 0])
        for i, agent_i in enumerate(peer_group):
            if self.active_agents[agent_i] == 1:
                peer_group_obs[i] = 1
                peer_reputation[i] = np.nansum(self.agent_rewards[agent_i, :]).astype(
                    np.float32
                )
                # peer_h_indexes[i] = self.agent_h_indexes[agent_i].astype(np.float16)
                centroids = np.array(
                    [
                        self.projects[pid].kene
                        for pid in self.agent_successful_projects[agent_i]
                    ]
                )
                if len(centroids) == 0:
                    peer_centroids[i] = self.peer_group_centroids[
                        self.agent_peer_idx[agent_i]
                    ]
                else:
                    peer_centroids[i] = centroids.mean(axis=0)
                if agent_i == idx:
                    self_centroid = peer_centroids[i]

        obs = {
            "peer_group": peer_group_obs,
            "peer_reputation": peer_reputation,
            "accumulated_rewards": np.array(
                [np.nansum(self.agent_rewards[idx, :])], dtype=np.float32
            ),
            "peer_centroids": np.array(peer_centroids, dtype=np.float64),
            "peer_h_index": np.array([self.agent_h_indexes[idx]], dtype=np.int16),
            "self_centroid": np.array([self_centroid], dtype=np.float64),
            "project_opportunities": self._get_open_projects_obs(idx),
            "running_projects": self._get_running_projects_obs(idx, peer_group),
            "age": np.array([self.agent_steps[idx]], dtype=np.int32),
        }
        return obs

    def _get_open_projects_obs(self, agent_idx: int) -> List[Dict[str, Any]]:
        agent_open_projs = {}
        for i, proj in enumerate(self.open_projects):
            proj_obs = {
                "required_effort": np.array(
                    [proj["required_effort"]], dtype=np.int32
                ),  # Box(0, 200, (1,), dtype=np.int32),
                "prestige": np.array(
                    [proj["prestige"]], dtype=np.float32
                ),  # Box(0, 1, (1,), dtype=np.float32),
                "novelty": np.array(
                    [proj["novelty"]], dtype=np.float32
                ),  # Box(0, 1, (1,), dtype=np.float32),
                "time_window": np.array(
                    [proj["time_window"]], dtype=np.int32
                ),  # Box(0, 200, (1,), dtype=np.int32),
            }
            agent_open_projs[f"project_{i}"] = proj_obs
        return agent_open_projs

    def _get_running_projects_obs(
        self, agent_idx: int, peer_group: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        # Projects the agent is currently working on
        running_obs = {}
        for p_idx in self._get_active_projects(agent_idx):
            p = self.projects[p_idx].as_observation_dict()
            fit_among_peers = np.zeros(self.max_peer_group_size, dtype=np.float32)
            effort_among_peers = np.zeros(self.max_peer_group_size, dtype=np.float32)
            contributors_among_peers = np.zeros(self.max_peer_group_size, dtype=np.int8)
            for i, agent_i in enumerate(peer_group):
                contributors_index = (
                    list(p["contributors"]).index(agent_i)
                    if agent_i in p["contributors"]
                    else None
                )
                if contributors_index is not None:
                    contributors_among_peers[i] = 1
                    fit_among_peers[i] = p["peer_fit"][contributors_index]
                    effort_among_peers[i] = self.agent_project_effort[agent_i][p_idx]

            p["contributors"] = contributors_among_peers
            p["peer_fit"] = fit_among_peers
            p["contributor_effort"] = effort_among_peers
            p["time_left"] = np.array(
                [max(0, p["time_window"] - (self.timestep - p["start_time"]))],
                dtype=np.int32,
            )
            del p["start_time"]
            del p["time_window"]
            running_obs[f"project_{p_idx}"] = p
        return running_obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> GymDict:
        space = GymDict(
            {
                "peer_group": Box(
                    0, self.n_agents, (self.max_peer_group_size,), dtype=np.int32
                ),
                "peer_reputation": Box(
                    0, 1e4, (self.max_peer_group_size,), dtype=np.float32
                ),
                "peer_h_index": Box(
                    0, 1e5, (self.max_peer_group_size,), dtype=np.int16
                ),
                "peer_centroids": Box(
                    -1, 1, (self.max_peer_group_size, 2), dtype=np.float64
                ),
                "self_centroids": Box(-1, 1, (1, 2), dtype=np.float64),
                "project_opportunities": GymDict(
                    {
                        f"project_{i}": GymDict(
                            {
                                "required_effort": Box(1, 200, (1,), dtype=np.int32),
                                "prestige": Box(0.1, 1, (1,), dtype=np.float32),
                                "novelty": Box(0.1, 1, (1,), dtype=np.float32),
                                "time_window": Box(1, 200, (1,), dtype=np.int32),
                            }
                        )
                        for i in range(self.n_projects_per_step)
                    }
                ),
                "running_projects": GymDict(
                    {
                        f"project_{i}": GymDict(
                            {
                                "required_effort": Box(1, 200, (1,), dtype=np.int32),
                                "prestige": Box(0.1, 1, (1,), dtype=np.float32),
                                "novelty": Box(0.1, 1, (1,), dtype=np.float32),
                                "peer_fit": Box(
                                    -1, 1, (self.max_peer_group_size,), dtype=np.float32
                                ),
                                "time_left": Box(0, 250, (1,), dtype=np.int32),
                                "current_effort": Box(
                                    0,
                                    self.max_peer_group_size * 200,
                                    (1,),
                                    dtype=np.float32,
                                ),
                                "contributors": MultiBinary(
                                    self.max_peer_group_size,
                                ),
                                "contributor_effort": Box(
                                    0,
                                    200,
                                    (self.max_peer_group_size,),
                                    dtype=np.float32,
                                ),
                            }
                        )
                        for i in range(self.max_projects_per_agent)
                    }
                ),
                "age": Box(0, 1e5, (1,), dtype=np.int32),
                "accumulated_rewards": Box(0, 1e5, (1,), dtype=np.float32),
            }
        )
        return space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> GymDict:
        space = GymDict(
            {
                "choose_project": Discrete(self.n_projects_per_step + 1),
                "collaborate_with": MultiBinary(self.max_peer_group_size),
                "put_effort": Discrete(self.max_projects_per_agent + 1),
            }
        )
        return space
