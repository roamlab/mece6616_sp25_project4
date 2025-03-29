import torch
import collections
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from dataset import normalize_data, unnormalize_data
from util import save_video_with_cv2



def test_cosine_beta_schedule(input_function):
    timesteps = 100
    ref_betas = torch.tensor(
        [
            6.3128e-04, 1.1169e-03, 1.6029e-03, 2.0894e-03, 2.5767e-03, 3.0650e-03,
            3.5546e-03, 4.0456e-03, 4.5385e-03, 5.0333e-03, 5.5304e-03, 6.0300e-03,
            6.5323e-03, 7.0378e-03, 7.5465e-03, 8.0589e-03, 8.5751e-03, 9.0955e-03,
            9.6205e-03, 1.0150e-02, 1.0685e-02, 1.1226e-02, 1.1772e-02, 1.2324e-02,
            1.2883e-02, 1.3449e-02, 1.4023e-02, 1.4604e-02, 1.5193e-02, 1.5791e-02,
            1.6399e-02, 1.7016e-02, 1.7643e-02, 1.8282e-02, 1.8931e-02, 1.9593e-02,
            2.0268e-02, 2.0956e-02, 2.1658e-02, 2.2376e-02, 2.3109e-02, 2.3859e-02,
            2.4627e-02, 2.5413e-02, 2.6219e-02, 2.7047e-02, 2.7897e-02, 2.8770e-02,
            2.9668e-02, 3.0593e-02, 3.1546e-02, 3.2530e-02, 3.3545e-02, 3.4594e-02,
            3.5680e-02, 3.6805e-02, 3.7971e-02, 3.9182e-02, 4.0441e-02, 4.1751e-02,
            4.3116e-02, 4.4541e-02, 4.6030e-02, 4.7588e-02, 4.9221e-02, 5.0936e-02,
            5.2740e-02, 5.4640e-02, 5.6646e-02, 5.8768e-02, 6.1018e-02, 6.3407e-02,
            6.5952e-02, 6.8669e-02, 7.1578e-02, 7.4701e-02, 7.8065e-02, 8.1700e-02,
            8.5642e-02, 8.9935e-02, 9.4629e-02, 9.9786e-02, 1.0548e-01, 1.1181e-01,
            1.1888e-01, 1.2683e-01, 1.3586e-01, 1.4620e-01, 1.5815e-01, 1.7215e-01,
            1.8875e-01, 2.0879e-01, 2.3344e-01, 2.6453e-01, 3.0494e-01, 3.5953e-01,
            4.3718e-01, 5.5538e-01, 7.4994e-01, 9.9900e-01]
    )
    test_betas = input_function(timesteps)

    # Compare values
    max_abs_diff = torch.max(torch.abs(ref_betas - test_betas)).item()
    print(f"Max absolute difference between reference and student betas: {max_abs_diff:.6e}")

    # Assert closeness
    assert torch.allclose(ref_betas, test_betas, atol=1e-3), \
        "Test failed: Your implementation differs from the reference!"

    # Plot for visual verification
    plt.figure(figsize=(8, 4))
    plt.plot(ref_betas.numpy(), label="Reference")
    plt.plot(test_betas.numpy(), linestyle="--", label="Yours")
    plt.title("Cosine Beta Schedule")
    plt.xlabel("Timestep")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Test passed!")



def score_part1(env, mlp_inference, agent, obs_horizon, action_dim, pred_horizon, action_horizon, max_steps, stats, device):
    # Rollout the policy
    score_list = list()
    combined_imgs = [env.render(mode='rgb_array')]

    for iter in range(10):
        env.seed(iter)

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Eval PushTStateEnv iter:{iter}") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # predict the next action sequence
                naction = mlp_inference(
                    model=agent,
                    batch_size=B,
                    nobs=nobs,
                    pred_horizon=pred_horizon,
                    action_dim=action_dim,
                )

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    #and reward/vis
                    rewards.append(reward)
                    combined_imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print('Score: ', max(rewards))
        # add to the score list
        score_list.append(max(rewards))

    # compute the mean of the scores
    print('Mean score: ', np.mean(score_list))

    # visualize
    save_video_with_cv2('mlp_combined_vis.mp4', combined_imgs, fps=30)
    print("Video saved successfully!")


def score_part2(env, cvae_inference, agent, obs_horizon, action_dim, pred_horizon, action_horizon, max_steps, stats, device):
    # Rollout the policy
    score_list = list()
    combined_imgs = [env.render(mode='rgb_array')]

    for iter in range(10):
        env.seed(iter)

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Eval PushTStateEnv iter:{iter}") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # predict the next action sequence
                naction = cvae_inference(
                    model=agent,
                    batch_size=B,
                    nobs=nobs,
                    pred_horizon=pred_horizon,
                    action_dim=action_dim,
                )

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    #and reward/vis
                    rewards.append(reward)
                    combined_imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print('Score: ', max(rewards))
        # add to the score list
        score_list.append(max(rewards))

    # compute the mean of the scores
    print('Mean score: ', np.mean(score_list))

    # visualize
    save_video_with_cv2('cvae_combined_vis.mp4', combined_imgs, fps=30)
    print("Video saved successfully!")


def score_part3(env, ddpm_inference, model, scheduler, num_diffusion_iters, obs_horizon, action_dim, pred_horizon, action_horizon, max_steps, stats, device):
    # Rollout the policy
    score_list = list()
    combined_imgs = [env.render(mode='rgb_array')]

    for iter in range(10):
        env.seed(iter)

        # get first observation
        obs, info = env.reset()

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc=f"Eval PushTStateEnv iter:{iter}") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = normalize_data(obs_seq, stats=stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

                # predict the next action sequence
                naction = ddpm_inference(
                    ema_model=model,
                    ddpm_scheduler=scheduler,
                    batch_size=B,
                    nobs=nobs,
                    num_diffusion_iters=num_diffusion_iters,
                    pred_horizon=pred_horizon,
                    action_dim=action_dim,
                    device=device,
                )

                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = unnormalize_data(naction, stats=stats['action'])

                # only take action_horizon number of actions
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[start:end, :]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, reward, done, _, info = env.step(action[i])
                    # save observations
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    combined_imgs.append(env.render(mode='rgb_array'))

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break

        # print out the maximum target coverage
        print('Score: ', max(rewards))
        # add to the score list
        score_list.append(max(rewards))

    # compute the mean of the scores
    print('Mean score: ', np.mean(score_list))

    # visualize
    save_video_with_cv2('ddpm_combined_vis.mp4', combined_imgs, fps=30)
    print("Video saved successfully!")