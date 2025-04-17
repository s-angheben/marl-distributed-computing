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
    agents_num: int = 9
    ms_num: int = 10
    ms_spawn_rate: float = 0.3  # Default spawn rate for all microservices
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
        )

    def _spawn_microservices(self, state: EnvState, key: PRNGKey) -> EnvState:
        """Spawn new microservices in the environment."""
        k1 = key

        # Generate random values for spawning
        random_values = jax.random.uniform(k1, state.ms_active.shape)

        # Determine where to spawn new microservices based on ms_spawn
        spawn_mask = (
            (random_values < state.ms_spawn) &  # Use per-type spawn rate
            (state.ms_active == 0)  # Only spawn where no service is active
        )

        # Set the time remaining for newly spawned services based on fixed ms_time
        new_time_remaining = jnp.where(
            spawn_mask,
            state.ms_time,  # Use fixed execution time for each service type
            state.ms_time_remaining
        )

        # Update active mask
        new_active = state.ms_active | spawn_mask

        return EnvState(
            step_count=state.step_count,
            agents_capacity=state.agents_capacity,
            agents_topology=state.agents_topology,
            ms_active=new_active,
            ms_cpu=state.ms_cpu,
            ms_time=state.ms_time,
            ms_time_remaining=new_time_remaining,
            ms_running=state.ms_running,
            ms_spawn=state.ms_spawn,  # Keep spawn rates unchanged
        )

    def step(self, state: EnvState, run_masks: Array, key: PRNGKey) -> Tuple[EnvState, Array, PRNGKey]:
        """Update environment state based on agent actions."""

        def agent_step(carry, agent_id):
            current_state = carry

            # Extract agent's action for execution
            run_mask = run_masks[agent_id]

            # Determine which services to run
            valid_run = current_state.ms_active[agent_id] & run_mask.astype(jnp.int32)

            # Calculate total CPU usage
            run_cpu = jnp.sum(valid_run * current_state.ms_cpu)

            # If CPU capacity exceeded, don't run any services
            valid_run = jax.lax.cond(
                run_cpu > current_state.agents_capacity[agent_id],
                lambda: jnp.zeros_like(valid_run),
                lambda: valid_run
            )

            # Update time remaining for running services
            updated_time_remaining = current_state.ms_time_remaining.at[agent_id].add(-valid_run)

            # Mark completed services
            done = (updated_time_remaining[agent_id] <= 0) & valid_run
            updated_active = current_state.ms_active.at[agent_id].set(
                jnp.where(done, 0, current_state.ms_active[agent_id])
            )
            updated_time_remaining = updated_time_remaining.at[agent_id].set(
                jnp.where(done, 0, updated_time_remaining[agent_id])
            )

            # Update running services mask
            updated_running = current_state.ms_running.at[agent_id].set(valid_run)

            # Create updated state
            updated_state = EnvState(
                step_count=current_state.step_count,
                agents_capacity=current_state.agents_capacity,
                agents_topology=current_state.agents_topology,
                ms_active=updated_active,
                ms_cpu=current_state.ms_cpu,
                ms_time=current_state.ms_time,
                ms_time_remaining=updated_time_remaining,
                ms_running=updated_running,
                ms_spawn=current_state.ms_spawn,
            )

            return updated_state, None

        # Process all agents
        updated_state, _ = jax.lax.scan(
            agent_step,
            state,
            jnp.arange(self.config.agents_num)
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
        final_state = EnvState(
            step_count=state.step_count + 1,
            agents_capacity=updated_state.agents_capacity,
            agents_topology=updated_state.agents_topology,
            ms_active=updated_state.ms_active,
            ms_cpu=updated_state.ms_cpu,
            ms_time=updated_state.ms_time,
            ms_time_remaining=updated_state.ms_time_remaining,
            ms_running=updated_state.ms_running,
            ms_spawn=updated_state.ms_spawn,
        )

        return final_state, reward, new_key