import os
import time
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import json

class Logger:
    """Logger for tracking and saving training metrics and statistics."""
    
    def __init__(
        self,
        log_dir: str = './logs',
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = False
    ):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (default: timestamp)
            use_tensorboard: Whether to use TensorBoard for logging
        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"bin_packing_{int(time.time())}"
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize TensorBoard if requested
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.experiment_dir)
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
        
        # Initialize metrics storage
        self.metrics = {
            'train': {},
            'eval': {},
            'test': {}
        }
        
        self.step_counter = 0
        print(f"Logger initialized. Logs will be saved to {self.experiment_dir}")
    
    def log_scalar(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        group: str = 'train'
    ):
        """Log a scalar value.
        
        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Step number (default: auto-incremented)
            group: Group name ('train', 'eval', or 'test')
        """
        # Auto-increment step if not provided
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Store in metrics
        if name not in self.metrics[group]:
            self.metrics[group][name] = []
        
        self.metrics[group][name].append((step, value))
        
        # Log to TensorBoard if enabled
        if self.use_tensorboard:
            self.writer.add_scalar(f"{group}/{name}", value, step)
    
    def log_scalars(
        self,
        metrics_dict: Dict[str, float],
        step: Optional[int] = None,
        group: str = 'train'
    ):
        """Log multiple scalar values.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            step: Step number (default: auto-incremented)
            group: Group name ('train', 'eval', or 'test')
        """
        # Auto-increment step if not provided
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Log each metric
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step, group)
    
    def log_histogram(
        self,
        name: str,
        values: Union[torch.Tensor, np.ndarray],
        step: Optional[int] = None,
        group: str = 'train'
    ):
        """Log a histogram of values.
        
        Args:
            name: Name of the histogram
            values: Tensor or array of values
            step: Step number (default: auto-incremented)
            group: Group name ('train', 'eval', or 'test')
        """
        # Auto-increment step if not provided
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Convert to numpy if necessary
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        
        # Log to TensorBoard if enabled
        if self.use_tensorboard:
            self.writer.add_histogram(f"{group}/{name}", values, step)
    
    def log_episode_stats(
        self,
        episode_id: int,
        total_reward: float,
        volume_utilized: float,
        items_placed: int,
        episode_length: int,
        additional_metrics: Optional[Dict[str, float]] = None,
        group: str = 'train'
    ):
        """Log statistics for an episode.
        
        Args:
            episode_id: Episode identifier
            total_reward: Total reward for the episode
            volume_utilized: Volume utilization ratio
            items_placed: Number of items placed
            episode_length: Episode length in steps
            additional_metrics: Additional metrics to log
            group: Group name ('train', 'eval', or 'test')
        """
        # Create metrics dictionary
        metrics = {
            'reward': total_reward,
            'volume_utilized': volume_utilized,
            'items_placed': items_placed,
            'episode_length': episode_length,
        }
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Log all metrics
        self.log_scalars(metrics, step=episode_id, group=group)
        
        # Print summary
        print(f"Episode {episode_id} ({group}): " + 
              f"Reward={total_reward:.4f}, Volume={volume_utilized:.2%}, " +
              f"Items={items_placed}, Length={episode_length}")
    
    def save_metrics(self, filename: Optional[str] = None):
        """Save all logged metrics to a JSON file.
        
        Args:
            filename: Name of the file (default: 'metrics.json')
        """
        if filename is None:
            filename = 'metrics.json'
        
        # Convert metrics to a serializable format
        serializable_metrics = {}
        for group, group_metrics in self.metrics.items():
            serializable_metrics[group] = {}
            for name, values in group_metrics.items():
                serializable_metrics[group][name] = values
        
        # Save to file
        with open(os.path.join(self.experiment_dir, filename), 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Metrics saved to {os.path.join(self.experiment_dir, filename)}")
    
    def plot_metric(
        self,
        metric_name: str,
        groups: Optional[List[str]] = None,
        window_size: int = 1,
        save: bool = True
    ) -> Figure:
        """Plot a metric over time.
        
        Args:
            metric_name: Name of the metric to plot
            groups: List of groups to include ('train', 'eval', 'test')
            window_size: Window size for moving average smoothing
            save: Whether to save the plot to a file
            
        Returns:
            Matplotlib figure
        """
        if groups is None:
            groups = ['train', 'eval', 'test']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each group
        for group in groups:
            if metric_name in self.metrics[group] and self.metrics[group][metric_name]:
                # Extract steps and values
                steps, values = zip(*self.metrics[group][metric_name])
                steps = np.array(steps)
                values = np.array(values)
                
                # Apply moving average if window_size > 1
                if window_size > 1:
                    values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                    steps = steps[:len(values)]
                
                # Plot
                ax.plot(steps, values, label=f"{group}")
        
        # Set labels and legend
        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} over time')
        ax.legend()
        ax.grid(True)
        
        # Save figure if requested
        if save:
            filename = f"{metric_name.replace('/', '_')}.png"
            fig.savefig(os.path.join(self.experiment_dir, filename))
            print(f"Plot saved to {os.path.join(self.experiment_dir, filename)}")
        
        return fig
    
    def close(self):
        """Close the logger and release resources."""
        # Save metrics
        self.save_metrics()
        
        # Close TensorBoard writer if used
        if self.use_tensorboard:
            self.writer.close()

class VisualLogger:
    """Logger for visualizing bin packing states."""
    
    def __init__(
        self,
        log_dir: str = './visualizations',
        experiment_name: Optional[str] = None,
        max_episodes: int = 10
    ):
        """Initialize the visual logger.
        
        Args:
            log_dir: Directory to save visualizations
            experiment_name: Name of the experiment (default: timestamp)
            max_episodes: Maximum number of episodes to visualize
        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"bin_packing_viz_{int(time.time())}"
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.max_episodes = max_episodes
        self.current_episodes = 0
        
        print(f"Visual logger initialized. Visualizations will be saved to {self.experiment_dir}")
    
    def visualize_height_map(
        self,
        height_map: Union[torch.Tensor, np.ndarray],
        title: str = "Height Map",
        save_path: Optional[str] = None
    ) -> Figure:
        """Visualize a height map.
        
        Args:
            height_map: Height map tensor or array [W, L]
            title: Title for the plot
            save_path: Path to save the visualization (default: None)
            
        Returns:
            Matplotlib figure
        """
        # Convert to numpy if necessary
        if isinstance(height_map, torch.Tensor):
            height_map = height_map.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot height map as a heatmap
        im = ax.imshow(height_map.T, cmap='viridis', origin='lower')
        
        # Add colorbar
        fig.colorbar(im, ax=ax, label='Height')
        
        # Set labels and title
        ax.set_xlabel('Width (W)')
        ax.set_ylabel('Length (L)')
        ax.set_title(title)
        
        # Add grid
        ax.grid(False)
        
        # Save figure if path provided
        if save_path:
            fig.savefig(save_path)
            print(f"Height map visualization saved to {save_path}")
        
        return fig
    
    def visualize_episode(
        self,
        episode_id: int,
        height_maps: List[np.ndarray],
        actions: Optional[List[Tuple[int, int, int, int]]] = None,
        rewards: Optional[List[float]] = None,
        buffer_states: Optional[List[np.ndarray]] = None,
        interval: int = 1  # Save every nth step
    ):
        """Visualize an episode by saving height maps at regular intervals.
        
        Args:
            episode_id: Episode identifier
            height_maps: List of height maps for each step
            actions: Optional list of actions (item, rotation, x, y)
            rewards: Optional list of rewards
            buffer_states: Optional list of buffer states
            interval: Interval for saving visualizations
        """
        # Skip if we've reached the maximum number of episodes
        if self.current_episodes >= self.max_episodes:
            return
        
        # Create episode directory
        episode_dir = os.path.join(self.experiment_dir, f"episode_{episode_id}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save height maps at regular intervals
        for i in range(0, len(height_maps), interval):
            # Create title
            title = f"Step {i}"
            if actions and i > 0:
                item, rot, x, y = actions[i-1]
                title += f" | Action: Item {item}, Rot {rot}, Pos ({x},{y})"
            if rewards and i > 0:
                title += f" | Reward: {rewards[i-1]:.4f}"
            
            # Visualize height map
            save_path = os.path.join(episode_dir, f"step_{i:04d}.png")
            self.visualize_height_map(height_maps[i], title=title, save_path=save_path)
            
            # Visualize buffer state if provided
            if buffer_states and i < len(buffer_states):
                buffer_path = os.path.join(episode_dir, f"buffer_{i:04d}.txt")
                with open(buffer_path, 'w') as f:
                    f.write(f"Buffer state at step {i}:\n")
                    for j, item in enumerate(buffer_states[i]):
                        f.write(f"Item {j}: {item}\n")
        
        # Save episode summary
        with open(os.path.join(episode_dir, "summary.txt"), 'w') as f:
            f.write(f"Episode {episode_id} Summary\n")
            f.write(f"Total steps: {len(height_maps)}\n")
            
            if rewards:
                total_reward = sum(rewards)
                f.write(f"Total reward: {total_reward:.4f}\n")
            
            if actions:
                f.write(f"Total actions: {len(actions)}\n")
        
        self.current_episodes += 1
        print(f"Episode {episode_id} visualization saved to {episode_dir}")
    
    def create_animation(
        self,
        episode_id: int,
        save_gif: bool = True
    ):
        """Create an animation from saved visualizations.
        
        Args:
            episode_id: Episode identifier
            save_gif: Whether to save as a GIF file
        """
        try:
            from matplotlib.animation import FuncAnimation
            import glob
        except ImportError:
            print("Animation requires additional packages. Install with: pip install imageio-ffmpeg")
            return
        
        # Get all image files for this episode
        episode_dir = os.path.join(self.experiment_dir, f"episode_{episode_id}")
        image_files = sorted(glob.glob(os.path.join(episode_dir, "step_*.png")))
        
        if not image_files:
            print(f"No images found for episode {episode_id}")
            return
        
        # Create animation
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            img = plt.imread(image_files[frame])
            ax.imshow(img)
            ax.axis('off')
            return [ax]
        
        ani = FuncAnimation(fig, update, frames=len(image_files), blit=True)
        
        # Save as GIF if requested
        if save_gif:
            try:
                from matplotlib.animation import PillowWriter
                gif_path = os.path.join(episode_dir, f"animation.gif")
                ani.save(gif_path, writer=PillowWriter(fps=2))
                print(f"Animation saved to {gif_path}")
            except ImportError:
                print("Saving GIF requires Pillow. Install with: pip install Pillow")
        
        return ani