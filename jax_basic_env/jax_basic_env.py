import jax
import jax.numpy as jnp
from env_core import MicroserviceEnvConfig, MicroserviceEnv

# Example usage
if __name__ == "__main__":
    config = MicroserviceEnvConfig()
    env = MicroserviceEnv(config)
    
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    # Print initial state
    print("\n===== INITIAL STATE =====")
    print("Step count:", state.step_count)
    print("Active services:\n", state.ms_active)
    print("Running services:\n", state.ms_running)
    print("CPU requirements per service type:\n", state.ms_cpu)
    print("Time remaining:\n", state.ms_time_remaining)
    print("Agent topology:\n", state.agents_topology)
    
    # Create execution masks - run all active services
    run_masks = jnp.zeros((config.agents_num, config.ms_num), dtype=jnp.int32)
    # run only the first service
    run_masks = run_masks.at[:, 0].set(1)
    
    # Create move actions - by default no moves (-1)
    move_actions = jnp.full((config.agents_num, config.ms_num), -1, dtype=jnp.int32)
    
    # Example: Set up some moves for demonstration
    # Later in step 1: Try to move service 0 from agent 0 to agent 1
    move_actions = move_actions.at[0, 0].set(1)
    
    print("Run masks:\n", run_masks)
    print("Move actions:\n", move_actions)
    
    # Execute spawns to create some active microservices
    spawn_key, key = jax.random.split(key)
    state = env._spawn_microservices(state, spawn_key)
    
    print("\n===== AFTER SPAWNING =====")
    print("Active services:\n", state.ms_active)
    print("Time remaining:\n", state.ms_time_remaining)
    
    # Step 1
    new_state, rewards, key = env.step(state, run_masks, move_actions, key)
    
    print("\n===== STEP 1 =====")
    print("Step count:", new_state.step_count)
    print("Active services:\n", new_state.ms_active)
    print("Running services:\n", new_state.ms_running)
    print("Time remaining:\n", new_state.ms_time_remaining)
    print("Rewards:\n", rewards)
    
    # Update move actions for step 2
    # Try to move service 1 from agent 2 to agent 3 (if it exists)
    move_actions = jnp.full((config.agents_num, config.ms_num), -1, dtype=jnp.int32)
    move_actions = move_actions.at[2, 1].set(3)
    
    print("\nMove actions for step 2:\n", move_actions)
    
    # Step 2
    newer_state, rewards, key = env.step(new_state, run_masks, move_actions, key)
    
    print("\n===== STEP 2 =====")
    print("Step count:", newer_state.step_count)
    print("Active services:\n", newer_state.ms_active)
    print("Running services:\n", newer_state.ms_running)
    print("Time remaining:\n", newer_state.ms_time_remaining)
    print("Rewards:\n", rewards)
    
    # Print a summary of active services count and their distribution
    print("\n===== SUMMARY =====")
    initial_active = jnp.sum(state.ms_active)
    step1_active = jnp.sum(new_state.ms_active)
    step2_active = jnp.sum(newer_state.ms_active)
    print(f"Active services: Initial={initial_active}, Step1={step1_active}, Step2={step2_active}")
    
    # Distribution of services across agents
    print("\nService distribution by agent:")
    print("Initial:", jnp.sum(state.ms_active, axis=1))
    print("Step 1:", jnp.sum(new_state.ms_active, axis=1))
    print("Step 2:", jnp.sum(newer_state.ms_active, axis=1))
