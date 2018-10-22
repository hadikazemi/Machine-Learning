import gym
from ReinforcementLearning.models import DQN, Policy, Replay
import torch
import torch.nn.functional as F
from torch import optim

eps_init = 1.0
eps_min = 0.05
eps_decay = 0.995
replay_memory = 20000
batch_size = 1000
gamma = 0.999
lr = 0.0001
main_net_update = 10
episode_num = 1000

test_only = False

device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.device(0)
    device = 'cuda'

env = gym.make('LunarLander-v2')
state = env.reset()
print(state)

action_size = env.action_space.n
print(action_size, env.reward_range)
state_size = 8

layers = [state_size, 128, 64, action_size]

dqn_main = DQN(layers, device).to(device)
dqn_aux = DQN(layers, device).to(device)
dqn_aux.load_state_dict(dqn_main.state_dict())

policy = Policy(dqn_main, eps_init, eps_min, eps_decay, action_size)
replay = Replay(replay_memory, state_size=state_size, device=device)

optimizer = optim.Adam(dqn_aux.parameters(), lr=lr)

if test_only:
    dqn_main.load_state_dict(torch.load('dqn_main.pth'))
else:
    total_reward = 0
    for ep in range(episode_num):
        state = env.reset()

        done = False
        steps = 0
        while not done:
            steps += 1

            action = policy.action(state)
            new_state, reward, done, _ = env.step(action)

            temp = list(state)
            temp.append(action)
            temp.extend(list(new_state))
            temp.extend([reward, int(done)])
            replay.push(temp)
            # python 3
            # replay.push([*decode(state), action, *decode(new_state), reward, int(done)])
            total_reward += reward

            if len(replay.memory) > batch_size:
                batch = replay.sample(batch_size)
                q_hat = dqn_aux(batch.state)
                q_new_hat = dqn_main(batch.new_state)

                expected_state_action_values = batch.reward + gamma * (torch.max(q_new_hat, 1)[0]).unsqueeze(1) * (1 - batch.done)
                state_action_values = q_hat.gather(1, batch.action)

                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())

                optimizer.zero_grad()
                loss.backward()
                for param in dqn_aux.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            state = new_state

        if ep > 20:
            policy.update_eps()
        if not ep % main_net_update:
            dqn_main.load_state_dict(dqn_aux.state_dict())
            print(total_reward/float(main_net_update), ep, policy.eps)
            total_reward = 0

    torch.save(dqn_main.state_dict(), 'dqn_main.pth')

total_reward = 0
steps = 0
s = env.reset()
while True:
    a = policy.action(s, test=True)
    s, r, done, info = env.step(a)
    total_reward += r

    still_open = env.render()
    if still_open == False: break

    if steps % 20 == 0 or done:
        print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
    steps += 1
    if done: break
print(total_reward)
