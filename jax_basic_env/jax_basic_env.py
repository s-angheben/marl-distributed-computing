import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional

NUM_AGENTS = 4
NUM_SERVICES = 3
NUM_NEIGHBORS = 2
SPAWN_RATE = 0.3
MAX_EXEC_TIME = 5
CPU_CAPACITY = 100  
CPU_MEAN = 30       
CPU_STD = 10        

NEIGHBORS = jnp.array([
    [1, 2],
    [0, 3],
    [0, 3],
    [1, 2],
])

@jax.jit
def reset_env(rng_key):
    k1, k2, k3 = jax.random.split(rng_key, 3)
    service_mask = jax.random.randint(k1, (NUM_AGENTS, NUM_SERVICES), 0, 2)
    
    # Generate discrete CPU values
    service_cpu_float = jax.random.normal(k2, (NUM_AGENTS, NUM_SERVICES)) * CPU_STD + CPU_MEAN
    service_cpu = jnp.clip(jnp.round(service_cpu_float), 5, CPU_CAPACITY).astype(jnp.int32) * service_mask
    
    service_time = jax.random.randint(k3, (NUM_AGENTS, NUM_SERVICES), 1, MAX_EXEC_TIME + 1) * service_mask
    cpu_capacity = jnp.ones((NUM_AGENTS,), dtype=jnp.int32) * CPU_CAPACITY

    state = {
        "mask": service_mask,
        "cpu": service_cpu,
        "time": service_time,
        "capacity": cpu_capacity
    }
    return state

@jax.jit
def spawn_microservices(state, rng_key):
    k1, k2, k3 = jax.random.split(rng_key, 3)
    spawn_mask = (jax.random.uniform(k1, state["mask"].shape) < SPAWN_RATE) & (state["mask"] == 0)

    # Generate discrete CPU values
    new_cpu_float = jax.random.normal(k2, state["cpu"].shape) * CPU_STD + CPU_MEAN
    new_cpu = jnp.clip(jnp.round(new_cpu_float), 5, CPU_CAPACITY).astype(jnp.int32) * spawn_mask
    
    new_time = jax.random.randint(k3, state["time"].shape, 1, MAX_EXEC_TIME + 1) * spawn_mask

    new_mask = state["mask"] | spawn_mask
    new_state = {
        "mask": new_mask,
        "cpu": jnp.where(spawn_mask, new_cpu, state["cpu"]),
        "time": jnp.where(spawn_mask, new_time, state["time"]),
        "capacity": state["capacity"]
    }
    return new_state

@jax.jit
def step_env(state, run_masks, migrations, rng_key):
    def agent_step(carry, agent_id):
        s, run_masks, migrations = carry
        mask, cpu, time, cap = s["mask"], s["cpu"], s["time"], s["capacity"]

        run_mask = run_masks[agent_id]
        svc_to_move, neighbor_idx = migrations[agent_id]

        valid_run = mask[agent_id] & run_mask.astype(jnp.int32)
        run_cpu = valid_run * cpu[agent_id]
        total_cpu = jnp.sum(run_cpu)

        # Modified to do nothing when capacity is exceeded
        def do_nothing():
            return jnp.zeros_like(valid_run)

        valid_run = jax.lax.cond(total_cpu > cap[agent_id], do_nothing, lambda: valid_run)

        time = time.at[agent_id].add(-valid_run)
        done = (time[agent_id] <= 0) & valid_run

        mask = mask.at[agent_id].set(jnp.where(done, 0, mask[agent_id]))
        cpu = cpu.at[agent_id].set(jnp.where(done, 0, cpu[agent_id]))
        time = time.at[agent_id].set(jnp.where(done, 0, time[agent_id]))

        def do_migration(s):
            svc_ok = (svc_to_move >= 0) & (svc_to_move < NUM_SERVICES)
            n_ok = (neighbor_idx >= 0) & (neighbor_idx < NUM_NEIGHBORS)
            cond = svc_ok & n_ok & (mask[agent_id, svc_to_move] == 1)

            neighbor_id = NEIGHBORS[agent_id, neighbor_idx]

            def migrate():
                # Create new objects instead of modifying in place
                new_mask = s["mask"].at[agent_id, svc_to_move].set(0)
                new_mask = new_mask.at[neighbor_id, svc_to_move].set(1)
                
                new_cpu = s["cpu"].at[neighbor_id, svc_to_move].set(cpu[agent_id, svc_to_move])
                new_cpu = new_cpu.at[agent_id, svc_to_move].set(0)
                
                new_time = s["time"].at[neighbor_id, svc_to_move].set(time[agent_id, svc_to_move])
                new_time = new_time.at[agent_id, svc_to_move].set(0)
                
                return {
                    "mask": new_mask,
                    "cpu": new_cpu,
                    "time": new_time,
                    "capacity": s["capacity"]
                }

            return jax.lax.cond(cond, migrate, lambda: s)

        s = {"mask": mask, "cpu": cpu, "time": time, "capacity": cap}
        s = do_migration(s)
        return (s, run_masks, migrations), None

    (updated_state, _, _), _ = jax.lax.scan(agent_step, (state, run_masks, migrations), jnp.arange(NUM_AGENTS))
    rng_spawn, rng_next = jax.random.split(rng_key)
    new_state = spawn_microservices(updated_state, rng_spawn)

    reward = jnp.sum((state["mask"] == 1) & (state["time"] > 0) & (new_state["time"] < state["time"]), axis=1)
    return new_state, reward, rng_next

@jax.jit
def get_observations_with_neighbors(state):
    mask = state["mask"]
    cpu = state["cpu"]
    time = state["time"]
    cap = state["capacity"]

    def build_own_obs(agent_id):
        used_cpu = jnp.sum(cpu[agent_id] * mask[agent_id])
        return jnp.concatenate([
            mask[agent_id],
            cpu[agent_id],
            time[agent_id],
            jnp.array([cap[agent_id] - used_cpu])
        ])

    def neighbor_summary(agent_id):
        def summarize(n_id):
            used = jnp.sum(cpu[n_id] * mask[n_id])
            free = cap[n_id] - used
            num_services = jnp.sum(mask[n_id])
            return jnp.array([used, free, num_services])

        neighbors = NEIGHBORS[agent_id]
        summaries = jax.vmap(summarize)(neighbors)
        return summaries.flatten()

    def full_obs(agent_id):
        return jnp.concatenate([
            build_own_obs(agent_id),
            neighbor_summary(agent_id)
        ])

    obs = jax.vmap(full_obs)(jnp.arange(NUM_AGENTS))
    return obs

# Example usage
key = jax.random.PRNGKey(0)
state = reset_env(key)
run_masks = jnp.array([
    [1, 0, 1],
    [1, 1, 0],
    [0, 0, 1],
    [1, 1, 1],
])
migrations = jnp.array([
    [1, 0],
    [-1, -1],
    [0, 1],
    [-1, -1],
])

state, reward, key = step_env(state, run_masks, migrations, key)
obs = get_observations_with_neighbors(state)
print("Next CPU:\n", state["cpu"])
print("Time Left:\n", state["time"])
print("Mask:\n", state["mask"])
print("Reward:\n", reward)
print("Observations:\n", obs)
