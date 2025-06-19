import gym
from tensorboardX import SummaryWriter
import Config
from Agent import Agent
from TestProcess import TestProcess
from lunar_lander_env import UnrealLunarLanderEnv
from gymnasium.wrappers import NormalizeObservation

# --- The path to your Unreal Engine executable ---
YOUR_UNREAL_EXE_PATH = r"C:\Users\ADMIN\Desktop\lunar_lander\lunar_lander.exe"

# --------------------------------------------------- Initialization ---------------------------------------------------
# Create Unreal Engine environment and add wrappers
env = UnrealLunarLanderEnv(
    flask_port=5000,
    unreal_exe_path=YOUR_UNREAL_EXE_PATH,
    launch_unreal=True,  # Set to False if you start UE manually
    ue_launch_args=["-port=5000"]  # ,"-nullrhi"]
)
env = NormalizeObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

print("Attempting to reset the environment for the first time...")
state, _ = env.reset()
print("Initial reset successful. Starting training.\n")

# Create agent which will use DDPG to train NNs
agent = Agent(state.shape[0], env.action_space.shape[0])
# Initialize test process which will be occasionally called to test whether goal is met
test_process = TestProcess(state.shape[0], env.action_space.shape[0])
# Create writer for Tensorboard
writer = SummaryWriter(log_dir='content/runs/'+Config.writer_name) if Config.writer_flag else None
print(f"TensorBoard writer: {Config.writer_name}\n")

# ------------------------------------------------------ Training ------------------------------------------------------
episode = 0
agent.episode_reward = 0.0  # initialize per-episode reward accumulator

for n_step in range(Config.number_of_steps):
    # Check whether we should test the model
    if agent.check_test(test_process, n_step, writer, env):
        break

    # Get an action from the agent
    actions = agent.get_action(state, n_step, env)

    # Perform a step in the environment
    new_state, reward, terminated, truncated, info = env.step(actions)
    done = terminated or truncated

    # Accumulate reward for this episode
    agent.episode_reward += reward

    # Store experience in the replay buffer
    agent.add_to_buffer(state, actions, new_state, reward, done)

    # Update the agent's networks
    agent.update(n_step)

    # Move to the next state
    state = new_state

    # If the episode is over, record results, print summary, and reset
    if done:
        episode += 1
        # Print a summary at the end of the episode
        print(f"=== Episode {episode} ended at step {n_step} ===")
        print(f"    Total Reward: {agent.episode_reward:.2f}")
        print(f"    Episode Length: {env.episode_step_count}\n")

        # Log results and reset
        agent.record_results(n_step, writer, env)
        agent.episode_reward = 0.0  # reset accumulator
        state, _ = env.reset()
agent.save_model("final_model.pth")

# ------------------------------------------------------ Cleanup -------------------------------------------------------
if writer is not None:
    writer.close()
test_process.env.close()
env.close()
