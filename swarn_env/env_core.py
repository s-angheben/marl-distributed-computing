import jax
import jax.numpy as jnp
from flax import struct

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey

@struct.dataclass
class SwarmEnvConfig:
    step: int
    max_steps: int
    swarm_size: int
    swarm_topology: Array
    resource_types: int
    resource_max_per_type: int
    resource_power: Array
    resource_speed: Array
    resource_active_cost: Array
    resource_spawn_cost: Array
    resource_energy_cost: Array
    ms_types: int
    ms_power_req: Array
    ms_step_req: Array
    ms_spawn_rate: Array

def create_default_config():
    return SwarmEnvConfig(
        step=0,
        max_steps=100,
        swarm_size=6,
        swarm_topology=jnp.array([[0, 1, 0, 0, 0, 0],
                                  [1, 0, 1, 0, 0, 0],
                                  [0, 1, 0, 1, 0, 0],
                                  [0, 0, 1, 0, 1, 1],
                                  [0, 0, 0, 1, 0, 1],
                                  [0, 0, 0, 1, 1, 0]]),
        resource_types=3,
        resource_max_per_type=5,
        resource_power=jnp.array([20, 55, 100]),
        resource_speed=jnp.array([1, 2, 3]),
        resource_active_cost=jnp.array([0.1, 0.2, 0.3]),
        resource_spawn_cost=jnp.array([0.1, 0.2, 0.3]),
        resource_energy_cost=jnp.array([0.1, 0.2, 0.3]),
        ms_types=10,
        ms_power_req=jnp.array([5, 10, 15, 20, 35, 40, 45, 70, 85, 90]),
        ms_step_req=jnp.array([2, 3, 1, 4, 1, 3, 4, 2, 6, 2]),
        ms_spawn_rate=jnp.array([0.8, 0.7, 0.4, 0.5, 0.1, 0.3, 0.2, 0.2, 0.05, 0.1])
    )

@struct.dataclass
class EnvState():
    """ State of the Swarm Environment 
        n: number of agents
        m: number of microservices
        t: number of resources type
        k: max number of resources
    """
    swarm_topology: Array    # (n, n)
    swarm_resources: Array   # (n, t, k) one hot encoding

    swarm_addition_idx: Array    # (n)
    swarm_removal_idx: Array     # (n)

    ms_active: Array         # (n, m)
    ms_time_remaining: Array # (n, m)

@struct.dataclass
class AgentObservation():
    """ Observation of a single agent """
    # TODO add goal
    resources: Array        # (t, k)

    number_of_neighbors: int
    best_neighbor_resources: Array # (t, k)
    possible_swarm_connections_resources: Array # (t, k)
    possible_swarm_removal_resources: Array # (t, k)

    ms_active: Array        # (m)
    ms_time_remaining: Array # (m)


@struct.dataclass
class AgentAction():
    """ Action of a single agent 
        t: number of resources types
        s: number of scheduling algorithms

        actions:
        - 0: do nothing
        - 1 to k: spawn resource
        - k+1 to 2k: remove resource
        - 2k+1 to 2k+3: move reource (easier, harder)
        - 2k+4 to 2k+s: change scheduling algorithm
        - 2k+s+1: connect to random swarm
        - 2k+s+2: remove random swarm
    """
    action: Array  # (2k + s + 2)
 
class SwarmEnv():
    def __init__(self, config: SwarmEnvConfig):
        self.config = config

        self.init_jit = jax.jit(self.init)


    def init(self, key: PRNGKey) -> EnvState:
        """ Initialize the environment state """
        # Initial topology
        swarm_topology = self.config.swarm_topology

        # Zero active resources
        swarm_resources = jnp.zeros((self.config.swarm_size, 
                                     self.config.resource_types,
                                     self.config.resource_max_per_type), dtype=jnp.int32)

        # zero active microservices
        ms_active = jnp.zeros((self.config.swarm_size, self.config.ms_types), dtype=jnp.int32)
        ms_time_remaining = jnp.zeros((self.config.swarm_size, self.config.ms_types), dtype=jnp.int32)

        # Compute swarm_addition_idx (random two-step neighbor not already a neighbor)
        # JAX-compatible implementation that avoids dynamic shapes
        def get_two_step_not_neighbor(agent_id, key):
            n = swarm_topology.shape[0]
            
            # Create a mask of direct neighbors
            neighbor_mask = swarm_topology[agent_id] == 1
            
            # Compute two-step connections
            two_step = jnp.dot(swarm_topology[agent_id], swarm_topology)
            
            # Create mask of potential two-step neighbors
            valid_mask = (two_step > 0) & ~neighbor_mask & (jnp.arange(n) != agent_id)
            
            # Generate random values for each position
            rand_vals = jax.random.uniform(key, shape=(n,))
            
            # Set invalid positions to -inf so they're never selected
            rand_vals = jnp.where(valid_mask, rand_vals, -jnp.inf)
            
            # Take the argmax to select the valid position with highest random value
            selected_idx = jnp.argmax(rand_vals)
            
            # If no valid indices, use agent_id as fallback
            return jax.lax.cond(
                jnp.any(valid_mask),
                lambda _: selected_idx,
                lambda _: jnp.array(agent_id),
                operand=None
            )

        # Compute swarm_removal_idx (random neighbor)
        def get_random_neighbor(agent_id, key):
            n = swarm_topology.shape[0]
            
            # Create a mask of neighbors
            neighbor_mask = swarm_topology[agent_id] == 1
            
            # Generate random values for each position
            rand_vals = jax.random.uniform(key, shape=(n,))
            
            # Set non-neighbor positions to -inf so they're never selected
            rand_vals = jnp.where(neighbor_mask, rand_vals, -jnp.inf)
            
            # Take the argmax to select the neighbor with highest random value
            selected_idx = jnp.argmax(rand_vals)
            
            # If no neighbors, use agent_id as fallback
            return jax.lax.cond(
                jnp.any(neighbor_mask),
                lambda _: selected_idx,
                lambda _: jnp.array(agent_id),
                operand=None
            )

        # Split and distribute keys
        keys_add, keys_rem = jax.random.split(key, 2)
        keys_add = jax.random.split(keys_add, self.config.swarm_size)
        keys_rem = jax.random.split(keys_rem, self.config.swarm_size)

        # Apply functions to all agents
        swarm_addition_idx = jax.vmap(get_two_step_not_neighbor, in_axes=(0, 0))(jnp.arange(self.config.swarm_size), keys_add)
        swarm_removal_idx = jax.vmap(get_random_neighbor, in_axes=(0, 0))(jnp.arange(self.config.swarm_size), keys_rem)

        return EnvState(
            swarm_topology=self.config.swarm_topology,
            swarm_resources=swarm_resources,
            swarm_addition_idx=swarm_addition_idx,
            swarm_removal_idx=swarm_removal_idx,
            ms_active=ms_active,
            ms_time_remaining=ms_time_remaining
        )

    
    def step(self, state: EnvState, actions: AgentAction, key: PRNGKey) -> EnvState:
        """ Step the environment: apply actions for all agents in a JAX-performant way """
        # Split PRNG keys for each agent
        keys = jax.random.split(key, self.config.swarm_size)

        # Get action indices for each agent (assuming one-hot encoding)
        action_indices = jnp.argmax(actions.action, axis=-1)  # shape: (num_agents,)

        # Define per-action handler functions (pure, functional, JAX-friendly)
        def do_nothing(state, agent_id, action, key):
            # TODO: implement do nothing
            return state

        def spawn_resource(state, agent_id, action, key):
            # TODO: implement spawn resource
            return state

        def remove_resource(state, agent_id, action, key):
            # TODO: implement remove resource
            return state

        def move_resource(state, agent_id, action, key):
            # TODO: implement move resource
            return state

        def change_scheduling(state, agent_id, action, key):
            # TODO: implement change scheduling algorithm
            return state

        def connect_swarm(state, agent_id, action, key):
            # TODO: implement connect to random swarm
            return state

        def remove_swarm(state, agent_id, action, key):
            # TODO: implement remove random swarm
            return state

        # List of handlers in order matching action indices
        action_handlers = (
            do_nothing,
            spawn_resource,
            remove_resource,
            move_resource,
            change_scheduling,
            connect_swarm,
            remove_swarm,
        )
        # TODO: Expand action_handlers to match your full action space

        # Per-agent step function using lax.switch for dispatch
        def agent_step(agent_id, state, action, key):
            handler = lambda idx: action_handlers[idx](state, agent_id, action, key)
            return jax.lax.switch(action_indices[agent_id], handler, jnp.arange(len(action_handlers)))

        # Vectorize over all agents (vmap)
        # TODO: You may need to use scan or a custom loop if state updates are not independent
        # For now, assume independence for maximum speed:
        new_state = state
        # TODO: Use vmap or scan to apply agent_step to all agents

        # TODO: Update global state (e.g., time, rewards, etc.) if needed

        return new_state

    
    def agent_observation(self, state: EnvState, agent_id: int) -> AgentObservation:
        """ Get the observation for a specific agent (deterministic, uses indices from state) """
        resources = state.swarm_resources[agent_id]
        ms_active = state.ms_active[agent_id]
        ms_time_remaining = state.ms_time_remaining[agent_id]

        # Find direct neighbors
        neighbors = jnp.where(state.swarm_topology[agent_id] == 1)[0]

        # For each neighbor, sum their active microservices
        neighbor_ms_counts = state.ms_active[neighbors]  # shape: (num_neighbors, m)
        neighbor_total_ms = jnp.sum(neighbor_ms_counts, axis=1)  # shape: (num_neighbors,)

        # Find the neighbor with the less active microservices
        best_neighbor_idx = jnp.argmin(neighbor_total_ms)
        best_neighbor_id = neighbors[best_neighbor_idx] if neighbors.size > 0 else agent_id

        # Get that neighbor's resources
        best_neighbor_resources = state.swarm_resources[best_neighbor_id]

        # Use swarm_addition_idx and swarm_removal_idx for possible connections/removals
        possible_swarm_id = state.swarm_addition_idx[agent_id]
        possible_swarm_connections_resources = state.swarm_resources[possible_swarm_id]

        possible_removal_id = state.swarm_removal_idx[agent_id]
        possible_swarm_removal_resources = state.swarm_resources[possible_removal_id]

        return AgentObservation(
            resources=resources,
            number_of_neighbors=jnp.sum(state.swarm_topology[agent_id]),
            best_neighbor_resources=best_neighbor_resources,
            possible_swarm_connections_resources=possible_swarm_connections_resources,
            possible_swarm_removal_resources=possible_swarm_removal_resources,
            ms_active=ms_active,
            ms_time_remaining=ms_time_remaining
        )


if __name__ == "__main__":
    config = create_default_config()
    env = SwarmEnv(config)

    key = jax.random.PRNGKey(0)
    state = env.init_jit(key)

    # Print initial state
    print("\n===== INITIAL STATE =====")
    print("Step count:", env.config.step)
    print("Agent topology:\n", state.swarm_topology)
    print("Active resources:\n", state.swarm_resources)
    print("Active microservices:\n", state.ms_active)
    print("Time remaining:\n", state.ms_time_remaining)

    # Get observation for agent 0
    agent_id = 0
    agent_obs = env.agent_observation(state, agent_id)
    print("\n===== AGENT OBSERVATION =====")
    print("Agent ID:", agent_id)
    print("Resources:\n", agent_obs.resources)
    print("Number of neighbors:", agent_obs.number_of_neighbors)
    print("Best neighbor resources:\n", agent_obs.best_neighbor_resources)
    print("Possible swarm connections resources:\n", agent_obs.possible_swarm_connections_resources)
    print("Possible swarm removal resources:\n", agent_obs.possible_swarm_removal_resources)
    print("Active microservices:\n", agent_obs.ms_active)
    print("Time remaining:\n", agent_obs.ms_time_remaining)






