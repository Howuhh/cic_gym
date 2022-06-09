import gym
import torch

from cic.agent import CICAgent
from cic.trainer import CICTrainer
from cic.utils import set_seed, NormalNoise, rollout


def main():
    set_seed(seed=32)

    agent = CICAgent(
        obs_dim=17,
        action_dim=6,
        skill_dim=64,
        hidden_dim=256,
    )
    exploration = NormalNoise(
        action_dim=6,
        timesteps=1_000_000,
        max_action=1.0,
        eps_max=0.5,
        eps_min=0.0
    )
    trainer = CICTrainer(
        train_env="HalfCheetah-v3",
        checkpoints_path="cic_checkpoints"
    )
    trainer.train(
        agent=agent,
        exploration=exploration,
        timesteps=1_000_000,
        start_train=5000,
        batch_size=1024,
        buffer_size=50_000,
        update_skill_every=50,
        eval_every=5000
    )

    # agent = torch.load("cic_checkpoints/c31abd08-b9e2-48b3-8e41-e77b96f29fe5/agent_40000.pt")
    # skill = agent.get_new_skill()
    # rollout(gym.make("HalfCheetah-v3"), agent, skill, render_path="rollout.mp4")


if __name__ == "__main__":
    main()