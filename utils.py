import cv2
import torch
import collections
import numpy as np
from tqdm.auto import tqdm

from dataset import normalize_data, unnormalize_data


def save_video_with_cv2(video_path, frames, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Ensure frames are converted to uint8 format for cv2
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()


def score_agent(env, agent, obs_dim, obs_horizon, action_dim, pred_horizon, action_horizon, max_steps, stats, device):
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

                # infer action
                with torch.no_grad():
                    # reshape observation to (B, obs_horizon * obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # (B, pred_horizon, action_dim)
                    naction = agent(obs_cond)
                    naction = naction.view(B, pred_horizon, action_dim)

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