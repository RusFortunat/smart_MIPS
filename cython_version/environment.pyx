# smart modification of on-lattice 2d MIPS model that is taken from: 
# https://pubs.aip.org/aip/jcp/article/148/15/154902/195348/Phase-separation-and-large-deviations-of-lattice

# In this smart version of the code, i will try controlling the size of forming droplets using reinforcement
# learning. Specifically, i will be feeding the network the system parameters and fraction of blocked particlees, 
# and on the output i would ask it to tell whether the Peclet number needs to be increased or decreased.
# input state: density, Pe number, fraction of blocked particles, ...? 
# output: +\delta forward_jump_rate, -\delta forward_jump_rate

# The overall goal for this project would be to teach network to make state-to-state transitions -- starting 
# from some random steady state, we want network to reach the target steady state by adjusting system parameters.
# Specifically, we want network to learn to be able to control the size of the droplets by adjusting the Peclet number.

import math
import numpy as np
import random
import itertools
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse):
    # pick random particle
    picked_particle = random.randint(0,N-1) 
    X = particles[picked_particle][0]
    Y = particles[picked_particle][1]
    angle = particles[picked_particle][2]
    #if lattice[X][Y] == 0:
    #    print("Error! Lattice site is empty!")

    # COUNTERCLOCKWISE ANGLE DIRECTION STARTING FROM +X DIRECTION
    # 0 -- +X, 1 -- +Y, 2 -- -X, 3 -- -Y

    dice = random.random()
    # rotate
    if dice < 2*rotation_rate: 
        counter_or_clock_rotation = random.randint(0,1)
        new_angle = 0
        if counter_or_clock_rotation == 0: # +pi/2
            new_angle = angle + 1 if angle != 3 else 0
        else: # -pi/2
            new_angle = angle - 1 if angle != 0 else 3
        particles[picked_particle][2] = new_angle
    # translate
    else: 
        newX = X
        newY = Y
        # could be a smarter way to encode all what is below
        if dice < translate_along_rate + 2*rotation_rate: # jump along the director
            if angle == 0: # jump to the right, +X
                newX = X + 1 if X < L - 1 else 0
            elif angle == 1: # jump to the top, +Y
                newY = Y + 1 if Y < L - 1 else 0
            elif angle == 2: # jump to the left, -X
                newX = X - 1 if X > 0 else L - 1
            else: # jump to the bottom, -Y
                newY = Y - 1 if Y > 0 else L - 1
        else:
            if dice < translate_along_rate + 2*rotation_rate + translate_opposite_rate: # jump against the director
                if angle == 0: # jump to the right, +X --> to the left, -X
                    newX = X - 1 if X > 0 else L - 1
                elif angle == 1: # jump to the top, +Y --> to the bottom, -Y
                    newY = Y - 1 if Y > 0 else L - 1
                elif angle == 2: # jump to the left, -X --> to the right, +X
                    newX = X + 1 if X < L - 1 else 0
                else: # jump to the bottom, -Y --> to the top, +Y
                    newY = Y + 1 if Y < L - 1 else 0
            else:
                if dice < translate_along_rate + 2*rotation_rate + translate_opposite_rate + translate_transverse: # jump to +pi/2 from the director
                    if angle == 0: # jump to the right, +X --> to the top, +Y
                        newY = Y + 1 if Y < L - 1 else 0
                    elif angle == 1: # jump to the top, +Y --> to the left, -X
                        newX = X - 1 if X > 0 else L - 1
                    elif angle == 2: # jump to the left, -X --> to the bottom, -Y
                        newY = Y - 1 if Y > 0 else L - 1
                    else: # jump to the bottom, -Y --> to the right, +X
                        newX = X + 1 if X < L - 1 else 0
                else:  # jump to -pi/2 from the director
                    if angle == 0: # jump to the right, +X --> to the bottom, -Y
                        newY = Y - 1 if Y > 0 else L - 1
                    elif angle == 1: # jump to the top, +Y --> to the right, +X
                        newX = X + 1 if X < L - 1 else 0
                    elif angle == 2: # jump to the left, -X --> to the top, +Y
                        newY = Y + 1 if Y < L - 1 else 0
                    else: # jump to the bottom, -Y --> to the left, -X
                        newX = X - 1 if X > 0 else L - 1
        
        if lattice[newX][newY] == 0:
            lattice[X][Y] = 0 # particle leaves the original lattice site
            lattice[newX][newY] = 1 # diffusion
            particles[picked_particle][0] = newX
            particles[picked_particle][1] = newY




def do_training(num_episodes, L, sim_duration, forw_rate_increment):
    # training will be done for a few target droplet sizes that is represented by a fraction of completely surrounded particles
    # in the steady state

    # we choose different initial conditions every episode
    for i_episode in range(num_episodes):
        # initial conditions
        density = random.uniform(0.1, 0.5)
        density = round(density,2)
        print("Episode ", i_episode, "; density ", density)
        target_fraction = random.uniform(0.2,0.5) # doesn't look like it's possible to go beyond 0.8
        target_fraction = round(target_fraction,2)
        N = int(L*L*density)
        particles = []
        lattice = np.zeros(shape=(L, L))
        n = 0
        while n < N:
            X = random.randint(0, L-1)
            Y = random.randint(0, L-1)
            if lattice[X][Y] == 0: 
                lattice[X][Y] = 1
                angle = random.randint(0,3)
                entry = [X,Y,angle] 
                particles.append(entry)
                n += 1       
        
        # to set the other rates, i will use the original model reference that i mention at the top
        translate_along_rate = random.uniform(0, 0.95)
        alpha = 0.035 / 0.0071 # took form paper; alpha = translate_transverse / rotation_rate, with rotation_rate=0.1, translate_transverse=1, translate_along_rate=25
        rotation_rate = (1 - translate_along_rate) / (3*alpha + 2)
        translate_transverse = alpha * rotation_rate
        translate_opposite_rate = translate_transverse
        print("starting rates: v_+ =  ", translate_along_rate, "; v_0 = ", translate_transverse, "; D_r = ", rotation_rate)

        # first, relax from the initial conditions
        for t in range(5000):
            for n in range(N):
                update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

        blocked_fraction = count_blocked_particles(lattice, L) / N
        blocked_fraction = round(blocked_fraction,2)
        starting_fraction = blocked_fraction
        #print("system relaxed from its initial conditions")

        # adjust parameters every 1000 time steps to let the system relax to its steady state
        score = 0
        for update_params in range(100):

            rate_to_state = round(translate_along_rate, 2)
            state = [density, rate_to_state, blocked_fraction, target_fraction] 
            #print("state before ", state)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = select_action_training(state, n_actions) # increase or decrease the forward jumping rate
            delta = +forw_rate_increment if action == 0 else -forw_rate_increment
            
            translate_along_rate += delta
            reward = 0
            if translate_along_rate < 0:
                translate_along_rate = 0.02
                reward += -100
            elif translate_along_rate > 1:
                translate_along_rate = 1
                reward += -100
            rotation_rate = (1 - translate_along_rate) / (3*alpha + 2)
            translate_transverse = alpha * rotation_rate
            translate_opposite_rate = translate_transverse

            # relax to the steady state; here i assume that after 1000 updates the system would reach the new steady state
            for t in range(1000):
                for n in range(N):
                    update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

            blocked_fraction = count_blocked_particles(lattice, L) / N      # new fraction of blocked particles
            blocked_fraction = round(blocked_fraction,2)

            #print("action ", action.item())
            reward += - 10*abs(blocked_fraction - target_fraction)
            reward = torch.tensor([reward], dtype=torch.float32, device=device)
            #print("reward ", reward)
            next_state = [density, translate_along_rate, blocked_fraction, target_fraction] 
            #print("state after ", next_state)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0) 
            memory.push(state, action, next_state, reward)           
            score += reward

            if len(memory) > BATCH_SIZE:
                optimize_model() # optimize both predators and prey networks
                # Soft update of the target network's weights: θ′ ← τ θ + (1 −τ)θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

        print("final rates: v_+ =  ", translate_along_rate, "; v_0 = ", translate_transverse, "; D_r = ", rotation_rate)
        print("target ", target_fraction, "; starting fraction ", starting_fraction , "; after training ", blocked_fraction, "\n")
        #record_results = [density, target_fraction, blocked_fraction, translate_along_rate]
        #results.append(record_results)
        rewards.append(score) 
        plot_score()

    torch.save(target_net.state_dict(), PATH)
    plot_score(show_result=True)
    plt.ioff()
    #plt.show()
    plt.savefig("./Training.png", format="png", dpi=600)

# plots
def plot_score(show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf() # clf -- clear current figure
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Episode Duration')
    plt.plot(rewards)
    if len(rewards) >= 20:
        rewards_mean = torch.tensor(rewards, dtype=torch.float)
        means_r = rewards_mean.unfold(0, 20, 1).mean(1).view(-1)
        means_r = torch.cat((torch.zeros(19), means_r))
        plt.plot(means_r.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            output = "./training_score.png"
            plt.savefig(output, format = "png", dpi = 300)

# Main
if __name__ == '__main__':
    Jesse_we_need_to_train_NN = True
    ############# Model parameters for Machine Learning #############
    num_episodes = 50      # number of training episodes
    BATCH_SIZE = 100        # the number of transitions sampled from the replay buffer
    GAMMA = 0.99            # the discounting factor
    EPS_START = 0.9         # EPS_START is the starting value of epsilon; determines how random our action choises are at the beginning
    EPS_END = 0.001         # EPS_END is the final value of epsilon
    EPS_DECAY = 1000        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    TAU = 0.005             # TAU is the update rate of the target network
    LR = 1e-3               # LR is the learning rate of the AdamW optimizer
    ############# Lattice simulation parameters #############
    sim_duration = 5000
    forw_rate_increment = 0.05
    L = 100
    n_observations = 4      # just give network a difference between positive and negative spins
    n_actions = 2           # the particle can jump on any neighboring lattice sites, or stay put and eat
    hidden_size = 32        # hidden size of the network
    PATH = "./NN_params.txt"

    ############# Do the training if needed ##############
    if Jesse_we_need_to_train_NN:
        policy_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net = DQN(n_observations, hidden_size, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(100*sim_duration) # the overall memory batch size 
        #memory_prey = ReplayMemory(Nt) # the overall memory batch size 
        rewards = []
        results = []
        steps_done = 0 
        do_training(num_episodes, L, sim_duration, forw_rate_increment) 

    ############# Training summary ######################
    trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
    trained_net.load_state_dict(torch.load(PATH))

    ############# Post-training simulation ##############


    

    #filename = "results_L" + str(L) + "_runs" + str(runs) + ".txt"
    #with open(filename, 'w') as f:
    #    for t in range(Nt):
    #        output_string = str(t) + "\t" + str(particles_left[t]) + "\n"
    #        f.write(output_string)

    # animation
    '''
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(24, 24))
    im = plt.imshow(memory[:, :, 1], interpolation="none", aspect="auto", vmin=0, vmax=1)

    def animate_func(i):
        im.set_array(memory[:, :, i])
        return [im]

    fps = 60

    anim = animation.FuncAnimation(
        fig,
        animate_func,
        frames=sim_duration,
        interval=1000 / fps,  # in ms
    )

    print("Saving animation...")
    filename_animation = "anim_T" + str(Temp) + "_M=" + str(total_magnetization[sim_duration-1]) + ".mp4"
    anim.save(filename_animation, fps=fps, extra_args=["-vcodec", "libx264"])
    print("Done!")
    '''
