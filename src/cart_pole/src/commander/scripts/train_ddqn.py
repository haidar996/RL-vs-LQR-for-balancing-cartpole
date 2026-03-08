#!/usr/bin/env python3
import rospy
import numpy as np
import time
import torch
import os
import matplotlib.pyplot as plt
from collections import deque
import sys  # Added for filtering ROS args

from DDQN import cartpoleenv
from DDQNAGENT import DDQNAgent

def normalize_state(state):
    # Normalize to [-1, 1] range based on expected max values
    state[0] = state[0] / 0.9  # Cart position (assuming 2.4m threshold)
    state[1] = state[1] / 3.0  # Cart velocity
    state[2] = state[2] / 0.2  # Pole angle (0.2 rad threshold)
    state[3] = state[3] / 3.0  # Pole angular velocity
    return state

def save_model(agent, episode, save_dir="saved_models"):
    """Save the current model"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, f"ddqn_model_episode_{episode}.pth")
    
    # Prepare save data
    save_data = {
        'episode': episode,
        'q_network_state_dict': agent.q_net.state_dict(),
        'target_network_state_dict': agent.target_net.state_dict(),
        'epsilon': agent.epsilon,
        'memory_size': len(agent.memory)
    }
    
    # Only save optimizer if it exists
    if agent.optimizer is not None:
        save_data['optimizer_state_dict'] = agent.optimizer.state_dict()
    
    torch.save(save_data, model_path)
    
    rospy.loginfo(f"Model saved to {model_path}")
    
    rospy.loginfo(f"Model saved to {model_path}")

def convert_loss_to_float(loss_value):
    """Convert loss value to float, handling PyTorch tensors"""
    if isinstance(loss_value, torch.Tensor):
        return loss_value.item()  # Convert tensor to Python float
    elif isinstance(loss_value, (float, int, np.number)):
        return float(loss_value)
    else:
        return 0.0

def plot_training_results(episode_rewards, episode_losses, window_size=100):
    """Plot training rewards and losses with moving averages"""
    if len(episode_rewards) == 0:
        rospy.logwarn("No episode rewards to plot")
        return
    
    episodes = list(range(len(episode_rewards)))
    rewards = episode_rewards
    
    # Calculate moving averages for rewards
    moving_avg_rewards = []
    if len(rewards) >= window_size:
        for i in range(len(rewards)):
            if i < window_size:
                moving_avg_rewards.append(np.mean(rewards[:i+1]))
            else:
                moving_avg_rewards.append(np.mean(rewards[i-window_size+1:i+1]))
    else:
        moving_avg_rewards = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
    
    # Calculate moving averages for losses if available
    moving_avg_losses = []
    if len(episode_losses) > 0:
        # Convert losses to numpy array of floats
        losses_float = [convert_loss_to_float(loss) for loss in episode_losses]
        
        if len(losses_float) >= window_size:
            for i in range(len(losses_float)):
                if i < window_size:
                    moving_avg_losses.append(np.mean(losses_float[:i+1]))
                else:
                    moving_avg_losses.append(np.mean(losses_float[i-window_size+1:i+1]))
        else:
            moving_avg_losses = [np.mean(losses_float[:i+1]) for i in range(len(losses_float))]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Episode rewards
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.6, linewidth=0.8, label='Episode Reward')
    axes[0, 0].plot(episodes, moving_avg_rewards, 'r-', linewidth=2, label=f'Moving Avg ({window_size} episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of rewards
    axes[0, 1].hist(rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 1].axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    axes[0, 1].set_xlabel('Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Episode losses (if available)
    if len(episode_losses) > 0:
        # Convert losses to floats for plotting
        losses_for_plot = [convert_loss_to_float(loss) for loss in episode_losses]
        loss_episodes = list(range(len(losses_for_plot)))
        
        axes[0, 2].plot(loss_episodes, losses_for_plot, 'g-', alpha=0.6, linewidth=0.8, label='Episode Loss')
        if len(moving_avg_losses) > 0:
            axes[0, 2].plot(loss_episodes, moving_avg_losses, 'orange', linewidth=2, label=f'Moving Avg ({window_size} episodes)')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].set_title('Training Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        # Use log scale only if there are positive losses
        if any(l > 0 for l in losses_for_plot):
            axes[0, 2].set_yscale('log')
    else:
        axes[0, 2].text(0.5, 0.5, 'No loss data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[0, 2].transAxes, fontsize=12)
        axes[0, 2].set_title('Training Loss')
    
    # Plot 4: Cumulative average reward
    cumulative_avg = [np.mean(rewards[:i+1]) for i in range(len(rewards))]
    axes[1, 0].plot(episodes, cumulative_avg, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cumulative Average Reward')
    axes[1, 0].set_title('Cumulative Average Reward Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Last 500 episodes rewards (or all if less)
    if len(rewards) > 500:
        last_episodes = episodes[-500:]
        last_rewards = rewards[-500:]
        last_moving_avg = moving_avg_rewards[-500:]
    else:
        last_episodes = episodes
        last_rewards = rewards
        last_moving_avg = moving_avg_rewards
    
    axes[1, 1].plot(last_episodes, last_rewards, 'b-', alpha=0.6, linewidth=0.8)
    axes[1, 1].plot(last_episodes, last_moving_avg, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Recent Training Performance (Rewards)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Last 500 episodes losses (if available)
    if len(episode_losses) > 0:
        losses_for_plot = [convert_loss_to_float(loss) for loss in episode_losses]
        
        if len(losses_for_plot) > 500:
            last_loss_episodes = list(range(len(losses_for_plot)))[-500:]
            last_losses = losses_for_plot[-500:]
            last_loss_moving_avg = moving_avg_losses[-500:] if len(moving_avg_losses) > 500 else []
        else:
            last_loss_episodes = list(range(len(losses_for_plot)))
            last_losses = losses_for_plot
            last_loss_moving_avg = moving_avg_losses if len(moving_avg_losses) > 0 else []
        
        axes[1, 2].plot(last_loss_episodes, last_losses, 'g-', alpha=0.6, linewidth=0.8, label='Loss')
        if len(last_loss_moving_avg) > 0:
            axes[1, 2].plot(last_loss_episodes, last_loss_moving_avg, 'orange', linewidth=2, label='Moving Avg')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Recent Training Performance (Loss)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        if any(l > 0 for l in last_losses):
            axes[1, 2].set_yscale('log')
    else:
        axes[1, 2].text(0.5, 0.5, 'No recent loss data available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 2].transAxes, fontsize=12)
        axes[1, 2].set_title('Recent Training Performance (Loss)')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "training_results_with_loss.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    rospy.loginfo(f"Training plot with loss saved to {plot_filename}")
    
    # Also save raw data
    data_filename = "training_data_with_loss.npz"
    np.savez(data_filename, 
             episodes=episodes, 
             rewards=rewards, 
             moving_avg_rewards=moving_avg_rewards,
             cumulative_avg=cumulative_avg,
             losses=[convert_loss_to_float(loss) for loss in episode_losses] if len(episode_losses) > 0 else [],
             moving_avg_losses=moving_avg_losses if len(moving_avg_losses) > 0 else [])
    rospy.loginfo(f"Training data with loss saved to {data_filename}")
    
    plt.show(block=False)
    plt.pause(0.1)  # Brief pause to display plot

def main():
    # Filter out ROS-specific arguments (__name:=, __log:=, etc.)
    filtered_argv = []
    for arg in sys.argv:
        if arg.startswith("__"):  # Skip ROS arguments
            continue
        filtered_argv.append(arg)
    
    # Replace sys.argv with filtered version
    sys.argv = filtered_argv
    
    rospy.init_node("ddqn_trainer")

    env = cartpoleenv()
    agent = DDQNAgent()
    agent.load_model()

    num_episodes = 10000
    max_steps = 1000
    target_update_freq = 1000
    save_interval = 100  # Save model every 100 episodes

    step_count = 0
    episode_rewards = []  # Store all episode rewards
    episode_losses = []   # Store episode losses (average loss per episode)
    best_reward = -float('inf')

    try:
        for episode in range(num_episodes):
            # Check for ROS shutdown before each episode
            if rospy.is_shutdown():
                rospy.logwarn("ROS shutdown detected, saving model and exiting...")
                save_model(agent, episode, "interrupted_models")
                break
            
            state = env.reset()
            episode_reward = 0
            episode_loss = 0
            training_steps = 0  # Count training steps in this episode

            for t in range(max_steps):
                # Check for ROS shutdown before each step
                if rospy.is_shutdown():
                    rospy.logwarn("ROS shutdown detected during episode, saving model and exiting...")
                    save_model(agent, episode, "interrupted_models")
                    break
                
                action = agent.select_action(state)
                print(t, "   ", action)

                next_state, reward, done = env.step(action)

                agent.memory.push(state, action, reward, next_state, done)
                
                # Store loss before training step
                prev_loss = agent.loss if hasattr(agent, 'loss') else 0
                
                # Train step
                agent.train_step()
                
                # Accumulate loss if it changed and is a valid tensor
                if hasattr(agent, 'loss') and agent.loss != prev_loss:
                    try:
                        # Convert loss to float if it's a tensor
                        if isinstance(agent.loss, torch.Tensor):
                            loss_value = agent.loss.item()
                        else:
                            loss_value = float(agent.loss)
                        episode_loss += loss_value
                        training_steps += 1
                    except:
                        pass  # Skip if loss is not convertible

                state = next_state
                episode_reward += reward
                step_count += 1

                if step_count % target_update_freq == 0:
                    agent.update_target_network()

                if done:
                    break
            
            # Check if we broke due to ROS shutdown
            if rospy.is_shutdown():
                break

            agent.update_epsilon()
            
            # Store episode reward
            episode_rewards.append(episode_reward)
            
            # Store episode loss (average loss per training step in this episode)
            if training_steps > 0:
                avg_episode_loss = episode_loss / training_steps
            else:
                avg_episode_loss = 0
            episode_losses.append(avg_episode_loss)
            
            # Track best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                # Optional: Save best model
                best_model_dir = "best_models"
                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)
                torch.save(agent.q_net.state_dict(), 
                          os.path.join(best_model_dir, f"best_model_episode_{episode}.pth"))
            
            # Calculate moving average (last 100 episodes)
            if len(episode_rewards) >= 100:
                avg_reward = np.mean(episode_rewards[-100:])
            else:
                avg_reward = np.mean(episode_rewards)
            
            # Calculate moving average loss (last 100 episodes)
            if len(episode_losses) >= 100:
                # Convert losses to floats before calculating mean
                losses_float = [convert_loss_to_float(loss) for loss in episode_losses[-100:]]
                avg_loss = np.mean(losses_float)
            else:
                losses_float = [convert_loss_to_float(loss) for loss in episode_losses]
                avg_loss = np.mean(losses_float) if len(losses_float) > 0 else 0
            
            rospy.loginfo(
                f"Episode {episode} | Reward: {episode_reward:.2f} | "
                f"Avg100 Reward: {avg_reward:.2f} | Best: {best_reward:.2f} | "
                f"Loss: {avg_episode_loss:.6f} | Avg100 Loss: {avg_loss:.6f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )
            
            # Save model at regular intervals
            if (episode + 1) % save_interval == 0:
                save_model(agent, episode + 1)
                
                # Optional: Plot intermediate results every 500 episodes
                if (episode + 1) % 500 == 0:
                    rospy.loginfo(f"Generating intermediate plot at episode {episode+1}")
                    plot_training_results(episode_rewards, episode_losses)

    except KeyboardInterrupt:
        rospy.loginfo("Training interrupted by user")
        # Save model on keyboard interrupt
        save_model(agent, episode if 'episode' in locals() else 0, "interrupted_models")
    
    except Exception as e:
        rospy.logerr(f"Training error: {e}")
        # Save model on error
        if 'episode' in locals():
            save_model(agent, episode, "error_models")
        raise
    
    finally:
        # Always save final model
        rospy.loginfo("Saving final model...")
        final_episode = episode if 'episode' in locals() else 0
        save_model(agent, final_episode, "final_models")
        
        # Plot final results
        if 'episode_rewards' in locals() and len(episode_rewards) > 0:
            rospy.loginfo("Generating training plots with loss...")
            plot_training_results(episode_rewards, episode_losses)
            
            # Print summary statistics
            rospy.loginfo("\n" + "="*50)
            rospy.loginfo("TRAINING SUMMARY")
            rospy.loginfo("="*50)
            rospy.loginfo(f"Total episodes: {len(episode_rewards)}")
            rospy.loginfo(f"Average reward: {np.mean(episode_rewards):.2f}")
            rospy.loginfo(f"Median reward: {np.median(episode_rewards):.2f}")
            rospy.loginfo(f"Best reward: {np.max(episode_rewards):.2f}")
            rospy.loginfo(f"Worst reward: {np.min(episode_rewards):.2f}")
            rospy.loginfo(f"Std deviation: {np.std(episode_rewards):.2f}")
            
            if len(episode_losses) > 0:
                # Convert losses to floats for statistics
                losses_float = [convert_loss_to_float(loss) for loss in episode_losses]
                rospy.loginfo(f"\nLoss Statistics:")
                rospy.loginfo(f"Average loss: {np.mean(losses_float):.6f}")
                rospy.loginfo(f"Median loss: {np.median(losses_float):.6f}")
                rospy.loginfo(f"Min loss: {np.min(losses_float):.6f}")
                rospy.loginfo(f"Max loss: {np.max(losses_float):.6f}")
            
            # Last 100 episodes stats
            if len(episode_rewards) >= 100:
                last_100 = episode_rewards[-100:]
                rospy.loginfo(f"\nLast 100 episodes (Reward):")
                rospy.loginfo(f"  Average: {np.mean(last_100):.2f}")
                rospy.loginfo(f"  Best: {np.max(last_100):.2f}")
                rospy.loginfo(f"  Worst: {np.min(last_100):.2f}")
            
            if len(episode_losses) >= 100:
                # Convert losses to floats
                last_100_losses = [convert_loss_to_float(loss) for loss in episode_losses[-100:]]
                rospy.loginfo(f"\nLast 100 episodes (Loss):")
                rospy.loginfo(f"  Average: {np.mean(last_100_losses):.6f}")
                rospy.loginfo(f"  Best: {np.min(last_100_losses):.6f}")
                rospy.loginfo(f"  Worst: {np.max(last_100_losses):.6f}")
        
        rospy.loginfo("Training finished")
        
        # Keep plot window open for a while
        try:
            if 'plt' in locals():
                plt.show(block=True)
        except:
            pass

if __name__ == "__main__":
    main()