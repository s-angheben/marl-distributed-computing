import svgwrite
import numpy as np
from pathlib import Path
from typing import Union, Sequence, Dict, Tuple
import networkx as nx
from env_core import EnvState


class EnvVisualizer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self.node_size = 80 * self.scale
        self.canvas_padding = 80 * self.scale
        
        # Simple color scheme
        self.bg_color = "#FFFFFF"
        self.text_color = "#000000"
        self.node_color = "#EEEEEE"
        self.edge_color = "#CCCCCC"
        self.bar_bg_color = "#F5F5F5"
        self.bar_fill_color = "#4CAF50"
        
        # Service color palette (for different service types)
        self.service_colors = [
            "#E57373", "#F06292", "#BA68C8", "#9575CD", "#7986CB", 
            "#64B5F6", "#4FC3F7", "#4DD0E1", "#4DB6AC", "#81C784", 
            "#AED581", "#DCE775", "#FFF176", "#FFD54F", "#FFB74D"
        ]
    
    def _create_layout(self, topology: np.ndarray) -> Dict[int, Tuple[float, float]]:
        """Create a graph layout based on the topology matrix."""
        G = nx.Graph()
        num_agents = topology.shape[0]
        
        # Add nodes and edges based on topology
        for i in range(num_agents):
            G.add_node(i)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if topology[i, j] == 1:
                    G.add_edge(i, j)
        
        # Get a spring layout
        layout = nx.spring_layout(G, seed=42)
        
        # Scale layout to fit canvas
        max_x = max(pos[0] for pos in layout.values())
        min_x = min(pos[0] for pos in layout.values())
        max_y = max(pos[1] for pos in layout.values())
        min_y = min(pos[1] for pos in layout.values())
        
        width = max(0.001, max_x - min_x)  # Avoid division by zero
        height = max(0.001, max_y - min_y)  # Avoid division by zero
        
        # Scale and shift
        scaling_factor = 1.2  # Increase this value to increase the distance between nodes
        for node, pos in layout.items():
            layout[node] = (
                (pos[0] - min_x) / width * 800 * self.scale * scaling_factor + self.canvas_padding,
                (pos[1] - min_y) / height * 600 * self.scale * scaling_factor + self.canvas_padding
            )
        
        return layout
    
    def get_dwg(self, state: EnvState) -> svgwrite.Drawing:
        """Generate an SVG drawing of the state."""
        # Get node positions
        layout = self._create_layout(np.array(state.agents_topology))
        
        # Calculate canvas size
        max_x = max(pos[0] for pos in layout.values()) + self.node_size + self.canvas_padding
        max_y = max(pos[1] for pos in layout.values()) + self.node_size + self.canvas_padding
        
        # Create SVG
        dwg = svgwrite.Drawing(size=(f"{max_x}px", f"{max_y}px"))
        
        # Background
        dwg.add(dwg.rect(insert=("0px", "0px"), size=('100%', '100%'), fill=self.bg_color))
        
        # Create main group
        frame = dwg.g(id="frame")
        
        # Draw edges
        num_agents = state.agents_topology.shape[0]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if state.agents_topology[i, j] == 1:
                    frame.add(dwg.line(
                        start=layout[i],
                        end=layout[j],
                        stroke=self.edge_color,
                        stroke_width=f"{2 * self.scale}px"
                    ))
        
        # Draw each agent node
        for agent_id, (x, y) in layout.items():
            # Get CPU usage data
            cpu_capacity = state.agents_capacity[agent_id]
            running_services = np.array(state.ms_running[agent_id])
            active_services = np.array(state.ms_active[agent_id])
            ms_cpu = np.array(state.ms_cpu)
            
            # Calculate CPU usage from running services
            cpu_usage = np.sum(running_services * ms_cpu)
            
            # Draw agent node
            frame.add(dwg.circle(
                center=(x, y),
                r=f"{self.node_size / 2}px",
                fill=self.node_color,
                stroke=self.text_color,
                stroke_width=f"{1.5 * self.scale}px"
            ))
            
            # Agent ID
            frame.add(dwg.text(
                f"A{agent_id}",
                insert=(x, y - self.node_size / 4),
                text_anchor="middle",
                dominant_baseline="middle",
                fill=self.text_color,
                font_size=f"{12 * self.scale}px"
            ))
            
            # CPU usage bar
            bar_width = self.node_size * 0.8
            bar_height = 8 * self.scale
            
            # Bar background
            frame.add(dwg.rect(
                insert=(f"{x - bar_width/2}px", f"{y}px"),
                size=(f"{bar_width}px", f"{bar_height}px"),
                fill=self.bar_bg_color,
                stroke=self.text_color,
                stroke_width=f"{1 * self.scale}px"
            ))
            
            # Usage fill
            usage_width = (cpu_usage / cpu_capacity) * bar_width
            if usage_width > 0:
                frame.add(dwg.rect(
                    insert=(f"{x - bar_width/2}px", f"{y}px"),
                    size=(f"{usage_width}px", f"{bar_height}px"),
                    fill=self.bar_fill_color
                ))
            
            # Add CPU usage text below the bar
            frame.add(dwg.text(
                f"{int(cpu_usage)}/{cpu_capacity}",
                insert=(f"{x}px", f"{y + bar_height + 10 * self.scale}px"),
                text_anchor="middle",
                dominant_baseline="middle",
                fill=self.text_color,
                font_size=f"{10 * self.scale}px"
            ))
            
            # Draw microservices as simple circles around the agent
            ms_num = state.ms_active.shape[1]
            for ms_id in range(ms_num):
                if active_services[ms_id] == 1:
                    # Position in circle around agent
                    angle = 2 * np.pi * ms_id / ms_num
                    orbit_radius = self.node_size * 0.7  # Orbit radius
                    ms_radius = self.node_size * 0.2  # Smaller circle for simplicity
                    ms_x = x + orbit_radius * np.cos(angle)
                    ms_y = y + orbit_radius * np.sin(angle)
                    
                    # Service color
                    service_color = self.service_colors[ms_id % len(self.service_colors)]
                    
                    # Check if the service is running
                    is_running = running_services[ms_id] == 1
                    
                    # Draw service circle with shadow for non-running services
                    frame.add(dwg.circle(
                        center=(f"{ms_x}px", f"{ms_y}px"),
                        r=f"{ms_radius}px",
                        fill=service_color if is_running else "#D3D3D3",  # Gray shadow for non-running
                        fill_opacity="0.8",  # Slightly transparent
                        stroke=self.text_color,
                        stroke_width=f"{1 * self.scale}px"
                    ))
                    
                    # Add service information inside the circle
                    service_capacity = state.ms_cpu[ms_id]  # Capacity required
                    steps_remaining = state.ms_time_remaining[agent_id][ms_id]  # Steps remaining
                    frame.add(dwg.text(
                        f"S{ms_id}\n{service_capacity}\n{steps_remaining}",
                        insert=(f"{ms_x}px", f"{ms_y}px"),
                        text_anchor="middle",
                        dominant_baseline="middle",
                        fill=self.text_color,
                        font_size=f"{6 * self.scale}px"
                    ))
        
        # Step counter
        frame.add(dwg.text(
            f"Step: {state.step_count}",
            insert=(f"{self.canvas_padding}px", f"{20 * self.scale}px"),
            fill=self.text_color,
            font_size=f"{14 * self.scale}px"
        ))
        
        dwg.add(frame)
        return dwg
    
    def save_svg(self, state: EnvState, filename: Union[str, Path]) -> None:
        """Save a single state as an SVG file."""
        dwg = self.get_dwg(state)
        dwg.saveas(filename)
    
    def save_svg_animation(
        self,
        states: Sequence[EnvState],
        filename: Union[str, Path],
        frame_duration_seconds: float = 1.0,
    ) -> None:
        """Save a sequence of states as an animated SVG file."""
        if not states:
            raise ValueError("No states provided for animation")
        
        # Create base drawing
        dwg = self.get_dwg(states[0])
        del dwg.elements[-1]  # Remove initial frame
        
        frame_groups = []
        
        # Process each state
        for i, state in enumerate(states):
            state_dwg = self.get_dwg(state)
            group = state_dwg.elements[-1]
            group["id"] = f"_fr{i:x}"
            group["class"] = "frame"
            frame_groups.append(group)
        
        # Animation CSS
        total_seconds = frame_duration_seconds * len(frame_groups)
        style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
        style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"
        
        # Add frames with timing
        for i, group in enumerate(frame_groups):
            dwg.add(group)
            style += f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
        
        dwg.defs.add(svgwrite.container.Style(content=style))
        dwg.saveas(filename)


# Example usage
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from env_core import MicroserviceEnvConfig, MicroserviceEnv
    
    config = MicroserviceEnvConfig()
    env = MicroserviceEnv(config)
    key = jax.random.PRNGKey(0)
    state = env.init(key)
    
    visualizer = EnvVisualizer(scale=1.2)
    visualizer.save_svg(state, "environment_init.svg")
    
    # Animation example
    states = [state]
    run_masks = jnp.ones((config.agents_num, config.ms_num), dtype=jnp.int32)
    # run only the first service
    #run_masks = run_masks.at[:, 0].set(1)
    
    for i in range(5):
        state, rewards, key = env.step(state, run_masks, key)
        states.append(state)
        
        # Print step details to the console
        print("\n===== STEP", i + 1, "=====")
        print("Step count:", state.step_count)
        print("Active services:\n", state.ms_active)
        print("run masks:\n", run_masks)
        print("Running services:\n", state.ms_running)
        print("CPU requirements per service type:\n", state.ms_cpu)
        print("Time remaining:\n", state.ms_time_remaining)
        print("Rewards:\n", rewards)
    
    visualizer.save_svg_animation(states, "environment_animation.svg", frame_duration_seconds=1.0)
    
    print("Visualization complete. Check environment_init.svg and environment_animation.svg")