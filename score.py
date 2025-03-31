import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as torch_kl_div
import collections
import numpy as np
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from dataset import normalize_data, unnormalize_data
from util import save_video_with_cv2



# Test and compare
def test_kl_divergence(new_kl_divergence, batch_size=16, latent_dim=8):
    dummy_mu = torch.tensor([[-6.5052e-01, -8.5216e-01,  3.5559e-02, -1.2526e+00,  9.2190e-01,
         -5.2134e-01,  1.4183e+00, -1.3505e-01],
        [-8.0830e-01,  1.3081e+00,  5.3316e-01, -7.5062e-01,  1.5895e+00,
          6.7036e-01,  7.5531e-01,  2.1166e-01],
        [ 1.2672e+00, -3.1864e-01,  8.4702e-01,  2.9942e-01,  1.3307e+00,
         -8.4062e-01, -9.1890e-01, -1.3935e+00],
        [ 8.9628e-01,  1.2645e-01,  6.5380e-01, -1.5828e-03,  8.5072e-01,
         -1.0819e+00, -1.8943e+00, -1.2002e+00],
        [ 6.7797e-01, -5.3697e-01,  2.3846e-01,  3.7178e-01,  1.1200e+00,
         -1.2343e-01,  2.6611e-01,  1.5457e+00],
        [-2.4914e-01,  5.4214e-02,  6.2757e-02,  1.9285e-01,  2.0073e+00,
          7.6524e-01, -7.7907e-01, -9.9272e-01],
        [-5.1301e-01,  4.6015e-01, -7.5827e-01,  7.8142e-02,  5.0385e-01,
         -1.5004e+00,  5.5766e-02,  6.3198e-01],
        [-3.3260e-01,  1.5772e+00,  1.7730e+00,  1.3027e-01, -3.0427e-01,
         -1.8743e-01, -8.3208e-02,  4.3741e-01],
        [-5.0345e-01, -1.9609e+00,  1.1573e+00, -1.8114e+00, -6.5368e-03,
         -1.6925e+00,  1.4671e-01, -1.1088e+00],
        [-1.0295e+00,  8.8771e-01, -5.6598e-01, -8.4222e-01,  1.3865e+00,
         -1.2862e+00, -1.1113e+00,  7.5804e-01],
        [ 5.6872e-01,  3.5330e-01,  3.5778e-02,  9.9565e-01, -5.7588e-01,
          3.7062e-01,  3.2796e+00,  1.2046e+00],
        [-4.7624e-01, -5.6918e-01,  2.2235e-01, -2.2803e+00,  5.3268e-01,
          1.8392e+00,  1.0280e+00,  1.5383e+00],
        [-1.0050e+00,  9.3139e-01,  5.2532e-02,  5.7852e-01,  1.1077e+00,
         -5.8846e-01,  3.5520e-01, -2.4751e-01],
        [-4.3416e-01, -2.1375e+00,  1.9551e+00, -1.9397e+00, -8.2929e-01,
         -8.8379e-02, -9.1785e-01,  3.4228e-01],
        [ 1.0281e+00,  6.6642e-01,  1.3305e-01,  4.1525e-01,  1.3444e+00,
          1.2565e+00, -1.1169e+00, -7.9272e-01],
        [ 9.5513e-01,  5.1677e-01, -5.7418e-01, -2.1315e-01, -8.0831e-01,
         -1.4306e+00,  6.1098e-02, -4.8114e-01]])

    dummy_logvar = torch.tensor([[ 1.4556, -1.9093, -0.2521, -1.4515, -0.8993, -1.1653,  0.8043, -0.2588],
        [-1.1331, -0.4876, -0.0787,  0.2391, -2.3004,  0.8704,  0.3426,  2.2723],
        [ 0.4949,  0.0155,  0.1522,  0.8320,  2.8060, -0.6160, -0.7675,  1.0423],
        [-0.6717, -1.1952,  0.2712,  0.3353, -1.0757,  0.8716, -0.6235,  0.8001],
        [ 1.8761,  0.3987,  0.2777,  0.1777,  0.5594, -1.4577, -0.2987,  0.7168],
        [-0.0486, -2.2638,  0.7679,  1.4314,  0.7456, -0.6676,  0.2054, -0.8070],
        [ 1.1840, -0.5374,  0.2416, -1.3947, -1.8673,  1.3763,  1.3384, -0.1149],
        [ 0.7007,  0.0509,  0.7410, -0.8912, -0.5321, -2.1916,  1.3749, -0.0640],
        [ 0.0148, -1.3118, -0.6544,  0.5895,  1.4669,  0.3845,  0.3249,  0.7251],
        [-0.9155, -0.9448, -1.2387,  2.5269,  0.1119,  0.9043,  0.4717, -0.1844],
        [ 0.6096,  0.7487,  0.3533, -0.5379,  1.0714,  0.9237, -1.5815,  0.4143],
        [-0.0302,  2.6269,  0.1191,  0.6085, -1.1465, -1.4847,  0.8153, -0.1900],
        [-1.4585, -0.5467,  0.0730,  0.0592,  0.7489, -0.0178, -0.1542,  0.4467],
        [-0.7486,  2.2644, -0.5240,  0.7691, -0.0505,  1.4829,  0.9635, -0.4761],
        [ 0.6130, -0.8263,  0.9772, -0.0054, -1.6389,  1.2368, -1.2867,  2.6502],
        [-1.4365,  0.2872,  1.2302,  1.2404, -0.3054,  1.9547, -0.3691,  0.7587]])

    expected_kl = torch.tensor([119.844955])
    kl_custom = new_kl_divergence(dummy_mu, dummy_logvar)
    print(f"Your KL divergence: {kl_custom.item():.6f}")
    print(f"Expected KL divergence: {expected_kl.item():.6f}")
    print(f"Absolute difference: {abs(kl_custom.item() - expected_kl.item()):.6f}")

    if not torch.allclose(kl_custom, expected_kl, atol=1e-4):
        print("Test FAILED: Outputs differ. Check the implementation of your KL divergence.")
    else:
        print("Test PASSED: Outputs match.")



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
    print(f"Max absolute difference between reference and your betas: {max_abs_diff:.6e}")

    # Assert closeness
    assert torch.allclose(ref_betas, test_betas, atol=1e-3), \
        "Test FAILED: Outputs differ. Check the implementation of your cosine beta schedule."

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

    print("Test PASSED: Outputs match.")



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
    score1 = np.mean(score_list)
    part1_bound = 0.6
    grade1 = score1 / part1_bound * 5 if score1 < part1_bound else 5

    print('\n---')
    print(f'Part I Average Score: {score1}')
    print(f'Part I Grade: {score1:.2f} / {part1_bound:.2f} * 5 = {grade1:.2f}')

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
    score2 = np.mean(score_list)
    part2_bound = 0.65
    grade2 = score2 / part2_bound * 5 if score2 < part2_bound else 5

    print('\n---')
    print(f'Part II Average Score: {score2}')
    print(f'Part II Grade: {score2:.2f} / {part2_bound:.2f} * 5 = {grade2:.2f}')

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
    score3 = np.mean(score_list)
    part3_bound = 0.85
    grade3 = score3 / part3_bound * 5 if score3 < part3_bound else 5

    print('\n---')
    print(f'Part III Average Score: {score3}')
    print(f'Part III Grade: {score3:.2f} / {part3_bound:.2f} * 5 = {grade3:.2f}')

    # visualize
    save_video_with_cv2('ddpm_combined_vis.mp4', combined_imgs, fps=30)
    print("Video saved successfully!")