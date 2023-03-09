import os
import gymnasium as gym
import torch



import sys
sys.path.append(os.getcwd() + "/..")


from src.agents.ppo_agent import PPOAgent
from src.utils.util import get_device

def get_ppo_config():
    env_config = {
        "env_name" : "Walker2d-v4",
        "max_ep_timesteps" : 1000,
    }

    training_config = {
        "max_training_timesteps" : int(1e8),
        "action_std" : 0.6,
        "action_std_decay_rate" : 0.03,
        "min_action_std" : 0.1,
        "action_std_decay_freq" : int(2.5e5),
        "eps_clip" : 0.2,
        "gamma" : 0.99,
        "K_epochs" : 80
    }

    network_config = {
        "lr_actor" : 0.0003,
        "lr_critic" : 0.001,
        "net_width" : 64
    }

    return env_config, training_config, network_config


def train():
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    print(f"Device set to : {device_name}")
    print("="*50)
    
    env_config, training_config, network_config = get_ppo_config()

    # Get environment setting
    env_name = env_config["env_name"]
    max_ep_timesteps = env_config["max_ep_timesteps"]

    # Get training setting
    max_training_timesteps = training_config["max_training_timesteps"]
    action_std = training_config["action_std"]
    action_std_decay_rate = training_config["action_std_decay_rate"]
    min_action_std = training_config["min_action_std"]
    action_std_decay_freq = training_config["action_std_decay_freq"]
    update_timestep = max_ep_timesteps * 4
    eps_clip = training_config["eps_clip"]
    gamma = training_config["gamma"]
    K_epochs = training_config["K_epochs"]

    # Get network setting
    lr_actor = network_config["lr_actor"]
    lr_critic = network_config["lr_critic"]
    net_width = network_config["net_width"]
    
    # Additional setting
    print_freq = max_ep_timesteps * 10
    log_freq = max_ep_timesteps * 2
    save_model_freq = int(1e5)
    random_seed = 0


    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    print("="*50)
    print(f"Current environment name : {env_name}")
    

    directory = "PPO_preTrained"
    os.makedirs(directory, exist_ok=True)

    directory = directory + '/' + env_name + '/'
    os.makedirs(directory, exist_ok=True)

    ckpt_path = directory + f"PPO_{env_name}.pth"
    print(f"Save checkpoint path : {ckpt_path}")

    # Training procedure
    ppo_agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                net_width=net_width,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                K_epochs=K_epochs,
                eps_clip=eps_clip,
                action_std_init=action_std)

    print_running_reward = 0
    print_running_episodes = 0
    
    log_running_reward = 0
    log_running_episodes = 0
    
    time_step = 0
    i_episode = 0

    print("start")
    # Core training loop
    while time_step <= max_training_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_timesteps+1):
            # Select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _, _ = env.step(action)

            # Saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()

            if time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(f"Episode : {i_episode} \t\t Timestep : {time_step} \t\t Average Reward : {print_avg_reward}")

                print_running_reward = 0
                print_running_episodes = 0
            
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print(f"saving model at : {ckpt_path}")
                ppo_agent.save(ckpt_path)
                print("model saved")

            if done:
                break

        
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        i_episode += 1

    env.close()


if __name__ == "__main__":
    
    train()