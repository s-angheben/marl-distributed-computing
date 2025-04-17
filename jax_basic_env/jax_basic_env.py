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
    
    # Create execution masks - run all active services
    run_masks = jnp.zeros((config.agents_num, config.ms_num), dtype=jnp.int32)
    # run only the first service
    run_masks = run_masks.at[:, 0].set(1)
    print("Run masks:\n", run_masks)
    
    # Step 1
    new_state, rewards, key = env.step(state, run_masks, key)
    
    print("\n===== STEP 1 =====")
    print("Step count:", new_state.step_count)
    print("Active services:\n", new_state.ms_active)
    print("Running services:\n", new_state.ms_running)  # Use new_state, not state
    print("CPU requirements per service type:\n", new_state.ms_cpu)
    print("Time remaining:\n", new_state.ms_time_remaining)
    print("Rewards:\n", rewards)
    
    # Step 2
    newer_state, rewards, key = env.step(new_state, run_masks, key)
    
    print("\n===== STEP 2 =====")
    print("Step count:", newer_state.step_count)
    print("Active services:\n", newer_state.ms_active)
    print("Running services:\n", newer_state.ms_running)  # Use newer_state, not state
    print("CPU requirements per service type:\n", newer_state.ms_cpu)
    print("Time remaining:\n", newer_state.ms_time_remaining)
    print("Rewards:\n", rewards)
    
    # Print a summary of active services count
    print("\n===== SUMMARY =====")
    initial_active = jnp.sum(state.ms_active)
    step1_active = jnp.sum(new_state.ms_active)
    step2_active = jnp.sum(newer_state.ms_active)
    print(f"Active services: Initial={initial_active}, Step1={step1_active}, Step2={step2_active}")
