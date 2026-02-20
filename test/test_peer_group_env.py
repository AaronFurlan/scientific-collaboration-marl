import numpy as np

from env.peer_group_environment import PeerGroupEnvironment


def test_connect_peer_groups_adds_agents():
    """Ensure that connecting two peer groups adds at least one cross-group member to each group
    when there are active candidates available.
    """
    env = PeerGroupEnvironment(start_agents=2, max_agents=4, n_groups=2, max_peer_group_size=4)

    # Set up two groups with known members and make all agents active
    env.peer_groups = [[0, 1], [2, 3]]
    env.agent_peer_idx = [0, 0, 1, 1]
    env.active_agents = np.array([1, 1, 1, 1], dtype=np.int8)

    len0_before = len(env.peer_groups[0])
    len1_before = len(env.peer_groups[1])

    env._connect_peer_groups()

    # After connecting, each group should be larger than before (an agent was added)
    assert len(env.peer_groups[0]) > len0_before
    assert len(env.peer_groups[1]) > len1_before


def test_get_action_mask_handles_no_peers():
    """Ensure _get_action_mask returns a valid mask even when no peers are active
    (the previous code hit a breakpoint in this situation).
    """
    env = PeerGroupEnvironment(start_agents=1, max_agents=2, n_groups=2, max_peer_group_size=2)

    # Configure peer groups so agent_0 has no active peers (only itself active)
    env.peer_groups = [[0], [1]]
    env.agent_peer_idx = [0, 1]
    env.active_agents = np.array([1, 0], dtype=np.int8)

    mask = env._get_action_mask("agent_0")
    assert isinstance(mask, dict)
    assert "collaborate_with" in mask
    assert isinstance(mask["collaborate_with"], np.ndarray)
    # Should be length equal to max_peer_group_size and not raise
    assert mask["collaborate_with"].shape[0] == env.max_peer_group_size

