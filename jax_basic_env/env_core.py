import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional, Any, Tuple
from dataclasses import dataclass, field
Array = Any
PRNGKey = Any


class EnvState(NamedTuple):
    step_count: Array
    agents_capacity: jnp.ndarray
    agents_topology: jnp.ndarray
    ms_active: jnp.ndarray    # Binary mask indicating active microservices (agents_num, ms_num)
    ms_cpu: jnp.ndarray       # CPU requirement for each microservice TYPE (ms_num,)
    ms_time: jnp.ndarray      # Fixed execution time for each microservice TYPE (ms_num,)
    ms_time_remaining: jnp.ndarray  # Remaining execution time for each microservice (agents_num, ms_num)
    ms_running: jnp.ndarray   # Binary mask indicating services actively being executed (agents_num, ms_num)
    ms_spawn: jnp.ndarray     # Spawn rate for each microservice TYPE (ms_num,)
    ms_moved: jnp.ndarray     # Binary mask indicating services successfully moved in this step (agents_num, ms_num)


@dataclass(frozen=True)
class MicroserviceEnvConfig:
    agents_capacity: jnp.ndarray = field(
        default_factory=lambda: jnp.array([100, 100, 100, 50, 50, 20, 50, 50, 30])
    )
    agents_topology: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            [1, 1, 1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1, 1],
        ])
    )
    agents_speed: jnp.ndarray = field(
        default_factory=lambda: jnp.array([3, 2, 3, 2, 1, 1, 1, 1, 1])
    )  # Speed of each agent: how many time units processed per step
    agents_num: int = 9
    ms_num: int = 10
    ms_spawn: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.1, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    )  # Spawn rate for each microservice type
    ms_max_exec_time: int = 5
    ms_cpu_mean: float = 30
    ms_cpu_std: float = 10

    def __post_init__(self):
        """Set derived attributes after initialization"""
        if self.agents_num != len(self.agents_capacity):
            object.__setattr__(self, 'agents_num', len(self.agents_capacity))
        object.__setattr__(self, 'agents_max_capacity', jnp.max(self.agents_capacity))


class MicroserviceEnv():
    def __init__(self, config: MicroserviceEnvConfig):
        self.config = config
        
        # JIT-compile key methods for significant speedup
        self._step_jit = jax.jit(self.step)
        self._process_moves_jit = jax.jit(self._process_moves)
        self._spawn_microservices_jit = jax.jit(self._spawn_microservices)

    def init(self, key: PRNGKey) -> EnvState:
        """Initialize the environment state"""
        k1 = key

        # Generate fixed CPU requirements for microservices (single value per type)
        ms_cpu = jax.random.normal(
            k1,
            (self.config.ms_num,)
        ) * self.config.ms_cpu_std + self.config.ms_cpu_mean

        ms_cpu = jnp.clip(
            jnp.round(ms_cpu),
            5,
            self.config.agents_max_capacity
        ).astype(jnp.int32)

        # Generate fixed execution times for microservices (single value per type)
        ms_time = jnp.arange(1, self.config.ms_num + 1)  # Fixed times: 1, 2, ..., ms_num

        # Generate initial active mask (all zeros)
        ms_active = jnp.zeros((self.config.agents_num, self.config.ms_num), dtype=jnp.int32)

        # Generate initial time remaining (all zeros)
        ms_time_remaining = jnp.zeros((self.config.agents_num, self.config.ms_num), dtype=jnp.int32)

        # Generate initial running mask (all zeros)
        ms_running = jnp.zeros((self.config.agents_num, self.config.ms_num), dtype=jnp.int32)

        # Use spawn rates from the configuration
        ms_spawn = self.config.ms_spawn

        # Initialize moved mask (all zeros)
        ms_moved = jnp.zeros((self.config.agents_num, self.config.ms_num), dtype=jnp.int32)

        return EnvState(
            step_count=jnp.int32(0),
            agents_capacity=self.config.agents_capacity,
            agents_topology=self.config.agents_topology,
            ms_active=ms_active,
            ms_cpu=ms_cpu,
            ms_time=ms_time,
            ms_time_remaining=ms_time_remaining,
            ms_running=ms_running,
            ms_spawn=ms_spawn,
            ms_moved=ms_moved,
        )

    def _spawn_microservices(self, state: EnvState, key: PRNGKey) -> EnvState:
        """Spawn new microservices in the environment with fully vectorized operations."""
        # Generate random values for all potential spawn locations at once
        random_values = jax.random.uniform(key, state.ms_active.shape)
        
        # Explicitly broadcast spawn rates to match (agents_num, ms_num) shape
        # state.ms_spawn has shape (ms_num,)
        spawn_rates = state.ms_spawn[jnp.newaxis, :]  # Shape (1, ms_num)
        
        # Vectorized mask creation - all operations happen element-wise across arrays
        available_slots = (state.ms_active == 0)
        spawn_decisions = (random_values < spawn_rates)
        spawn_mask = available_slots & spawn_decisions
        
        # Vectorized update of time remaining using broadcasting
        ms_time_broadcast = state.ms_time[jnp.newaxis, :]
        new_time_remaining = jnp.where(spawn_mask, ms_time_broadcast, state.ms_time_remaining)
        
        # Vectorized update of active mask
        new_active = state.ms_active | spawn_mask
        
        # Efficiently update only the changed fields
        return state._replace(
            ms_active=new_active,
            ms_time_remaining=new_time_remaining,
        )

    def _process_moves(self, state: EnvState, move_actions: Array) -> EnvState:
        """Process move actions with vectorized operations instead of sequential loops."""
        # Create masks for basic validations
        attempted_moves = (move_actions != -1)
        has_service = state.ms_active
        valid_target_range = jnp.logical_or(
            jnp.logical_and(
                jnp.greater_equal(move_actions, 0), 
                jnp.less(move_actions, self.config.agents_num)
            ),
            jnp.logical_not(attempted_moves)
        )
        
        # Basic validation mask
        basic_valid = attempted_moves & has_service & valid_target_range
        
        # Check if target is neighbor with a broadcasted operation
        source_indices = jnp.arange(self.config.agents_num)[:, jnp.newaxis]
        target_indices = jnp.where(basic_valid, move_actions, 0)
        
        # Create a mask checking if target is a neighbor for each valid move
        is_neighbor = jnp.zeros_like(basic_valid, dtype=bool)
        for i in range(self.config.agents_num):
            source_mask = (source_indices == i)
            for j in range(self.config.agents_num):
                target_mask = (target_indices == j) & source_mask & basic_valid
                is_neighbor = is_neighbor | (target_mask & (state.agents_topology[i, j] == 1))
        
        # Final validation mask
        valid_moves = basic_valid & is_neighbor
        
        # Handle conflict resolution with priority
        # Create a priority mask where lower agent indices get priority
        agent_priority = jnp.arange(self.config.agents_num)[:, jnp.newaxis, jnp.newaxis]
        service_indices = jnp.arange(self.config.ms_num)[jnp.newaxis, :, jnp.newaxis]
        
        # Prepare 3D arrays for target tracking
        targets_3d = jnp.broadcast_to(move_actions[:, :, jnp.newaxis], 
                                     (self.config.agents_num, self.config.ms_num, self.config.agents_num))
        valid_3d = jnp.broadcast_to(valid_moves[:, :, jnp.newaxis], 
                                   (self.config.agents_num, self.config.ms_num, self.config.agents_num))
        
        # For each service and target, the agent with lowest index wins
        target_agents = jnp.arange(self.config.agents_num)[jnp.newaxis, jnp.newaxis, :]
        target_matches = (targets_3d == target_agents) & valid_3d
        
        # For each service-target pair, find the winning agent (lowest index)
        winning_agent_mask = target_matches & (agent_priority == jnp.min(
            jnp.where(target_matches, agent_priority, float('inf')), axis=0, keepdims=True))
        
        # Final move validity: must be valid and must be the winning agent for conflicts
        final_valid_moves = valid_moves & jnp.any(winning_agent_mask, axis=2)
        
        # Prepare source and target arrays
        source_agents = jnp.arange(self.config.agents_num)[:, jnp.newaxis]
        target_agents = jnp.where(final_valid_moves, move_actions, -1)
        
        # Initialize output arrays
        ms_active_new = state.ms_active.copy()
        ms_time_remaining_new = state.ms_time_remaining.copy()
        ms_moved_new = jnp.zeros_like(state.ms_moved)
        
        # Apply all valid moves at once
        for service in range(self.config.ms_num):
            for source in range(self.config.agents_num):
                if final_valid_moves[source, service]:
                    target = target_agents[source, service]
                    # Remove from source
                    ms_active_new = ms_active_new.at[source, service].set(0)
                    ms_time_remaining_new = ms_time_remaining_new.at[source, service].set(0)
                    
                    # Add to target
                    ms_active_new = ms_active_new.at[target, service].set(1)
                    ms_time_remaining_new = ms_time_remaining_new.at[target, service].set(
                        state.ms_time_remaining[source, service])
                    ms_moved_new = ms_moved_new.at[target, service].set(1)
        
        return state._replace(
            ms_active=ms_active_new,
            ms_time_remaining=ms_time_remaining_new,
            ms_running=jnp.where(final_valid_moves, 0, state.ms_running),
            ms_moved=ms_moved_new
        )

    def step(self, state: EnvState, run_masks: Array, move_actions: Array, key: PRNGKey) -> Tuple[EnvState, Array, PRNGKey]:
        """Update environment state based on agent actions with vectorized operations."""
        # First, process move actions
        state_after_moves = self._process_moves(state, move_actions)
        
        # Vectorize agent step processing using vmap instead of scan
        def process_agent_run(agent_id, state):
            """Process run actions for a single agent."""
            # Extract agent's action for execution
            run_mask = run_masks[agent_id]

            # Determine which services to run
            valid_run = state.ms_active[agent_id] & run_mask.astype(jnp.int32)

            # Calculate total CPU usage
            run_cpu = jnp.sum(valid_run * state.ms_cpu)

            # If CPU capacity exceeded, don't run any services
            valid_run = jnp.where(
                run_cpu > state.agents_capacity[agent_id],
                jnp.zeros_like(valid_run),
                valid_run
            )
            
            # Get the agent's speed (how fast it processes services)
            agent_speed = self.config.agents_speed[agent_id]
            
            # Calculate completed services mask - decrement by agent speed
            time_remaining_new = state.ms_time_remaining[agent_id] - (valid_run * agent_speed)
            done = (time_remaining_new <= 0) & valid_run
            
            # Return the results for this agent
            return {
                'valid_run': valid_run,
                'time_remaining_new': time_remaining_new,
                'done': done
            }
        
        # Use vmap to process all agents in parallel
        agent_ids = jnp.arange(self.config.agents_num)
        agent_results = jax.vmap(process_agent_run, in_axes=(0, None))(agent_ids, state_after_moves)
        
        # Extract results from vmap
        valid_runs = agent_results['valid_run']
        time_remaining_new = agent_results['time_remaining_new']
        dones = agent_results['done']
        
        # Update state with vectorized operations
        ms_active_new = state_after_moves.ms_active * jnp.logical_not(dones)
        ms_time_remaining_new = jnp.where(dones, 0, time_remaining_new)
        ms_running_new = valid_runs
        
        # Update state efficiently
        updated_state = state_after_moves._replace(
            ms_active=ms_active_new,
            ms_time_remaining=ms_time_remaining_new,
            ms_running=ms_running_new
        )
        
        # Calculate rewards: progress on microservice execution
        reward = jnp.sum(
            (state.ms_active == 1) &
            (state.ms_time_remaining > 0) &
            (updated_state.ms_time_remaining < state.ms_time_remaining),
            axis=1
        )

        # Spawn new microservices
        spawn_key, new_key = jax.random.split(key)
        updated_state = self._spawn_microservices(updated_state, spawn_key)

        # Increment step counter
        final_state = updated_state._replace(step_count=state.step_count + 1)

        return final_state, reward, new_key
