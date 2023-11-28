import os
import logging
import torch

from pufferlib.vectorization import Serial, Multiprocessing
from pufferlib.policy_store import DirectoryPolicyStore
from pufferlib.frameworks import cleanrl
import clean_pufferl
import policy

def env_creator():
    """Create an environment."""
    env = nmmo.Env(Config(args))
    env = pufferlib.emulation.PettingZooPufferEnv(env,
        postprocessor_cls=Postprocessor,
        postprocessor_kwargs={
            'eval_mode': args.eval_mode,
            'early_stop_agent_num': args.early_stop_agent_num,
            'sqrt_achievement_rewards': args.sqrt_achievement_rewards,
            'heal_bonus_weight': args.heal_bonus_weight,
            'meander_bonus_weight': args.meander_bonus_weight,
            'explore_bonus_weight': args.explore_bonus_weight,
        },
    )
    return env

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # You can either edit the defaults in config.py or set args
    # from the commandline.
    args = config.create_config(config.Config)

    # Avoid OOMing your machine for local testing
    if args.local_mode:
        args.num_envs = 1
        args.num_buffers = 1
        args.use_serial_vecenv = True
        args.rollout_batch_size = 2**10

    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    logging.info("Training run: %s (%s)", args.run_name, run_dir)
    logging.info("Training args: %s", args)

    policy_store = None
    if args.policy_store_dir is None:
        args.policy_store_dir = os.path.join(run_dir, "policy_store")
        logging.info("Using policy store from %s", args.policy_store_dir)
        policy_store = DirectoryPolicyStore(args.policy_store_dir)

    def make_policy(envs):
        learner_policy = policy.Policy(
            envs.driver_env,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            task_size=args.task_size
        )
        return cleanrl.Policy(learner_policy)

    trainer = clean_pufferl.CleanPuffeRL(
        device=torch.device(args.device),
        seed=args.seed,
        env_creator=environment.make_env_creator(args),
        env_creator_kwargs={},
        agent_creator=make_policy,
        data_dir=run_dir,
        exp_name=args.run_name,
        policy_store=policy_store,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_extra_data=args,
        checkpoint_interval=args.checkpoint_interval,
        vectorization=Serial if args.use_serial_vecenv else Multiprocessing,
        total_timesteps=args.train_num_steps,
        num_envs=args.num_envs,
        num_cores=args.num_cores or args.num_envs,
        num_buffers=args.num_buffers,
        batch_size=args.rollout_batch_size,
        learning_rate=args.ppo_learning_rate,
        selfplay_learner_weight=args.learner_weight,
        selfplay_num_policies=args.max_opponent_policies + 1,
        #record_loss = args.record_loss,
    )

    while not trainer.done_training():
        trainer.evaluate()
        trainer.train(
            update_epochs=args.ppo_update_epochs,
            bptt_horizon=args.bptt_horizon,
            batch_rows=args.ppo_training_batch_size // args.bptt_horizon,
            clip_coef=args.clip_coef,
        )

    trainer.close()
