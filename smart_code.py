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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# structure of the Q table
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
# class that defines the Q table
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # deque is a more efficient form of a list

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, hidden_size, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)    

def count_blocked_particles(lattice, L):
    sum = 0
    for X in range(L):
        for Y in range(L):
            nextX = X + 1 if X < L - 1 else 0
            prevX = X - 1 if X > 0 else L - 1
            nextY = Y + 1 if Y < L - 1 else 0
            prevY = Y - 1 if Y > 0 else L - 1
            if lattice[nextX][Y] == 1 and lattice[prevX][Y] == 1 and lattice[X][nextY] == 1 and lattice[X][prevY] == 1:
                sum += 1 # blocked

    return sum

def select_action_training(state, n_actions): 
    global steps_done # count total number of steps to go from almost random exploration to more efficient actions
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # exploitation
        with torch.no_grad():
            #if predator_or_prey == 1:
            return policy_net(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
            #else:
                #return actor_prey(state).max(1)[1].view(1, 1) # view(1,1) changes shape to [[action], dtype]
    else:
        # select a random action; 
        rand_aciton = random.randint(0, n_actions-1) # left, right, top, bottom
        return torch.tensor([[rand_aciton]], device=device, dtype=torch.long)

def select_action_post_training(state, T): 
    # interpret Q values as probabilities when simulating dynamics of the system 
    # in principle this could be easily extended to make this more general, but i am a lazy boi
    with torch.no_grad():
        #print("state ", state)
        Q_values = trained_net(state)
        #print("Q-values ", Q_values)
        #probs = torch.softmax(Q_values, dim=1) # converts logits to probabilities (torch object)
        #print("probs ", probs)
        meanQ_value = 0
        for i in range(n_actions):
            meanQ_value += Q_values[0][i]/n_actions
        #print("mean Q value", meanQ_value)
        # update Q values
        for i in range(n_actions):
            Q_values[0][i] = (Q_values[0][i] - meanQ_value) / T
        #print("new Q values", Q_values)
        new_probs = torch.softmax(Q_values, dim=1)
        #print("new probs", new_probs, "\n")
        dist = Categorical(new_probs) # feeds torch object to generate a list of probs (numpy object ?)
        action = dist.sample().numpy()[0] # sample list of probs and return the action
        #print("action ", action, "\n\n")
        
        return action

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

def optimize_model():
    if len(memory) < BATCH_SIZE: # execute 'optimize_model' only if #BATCH_SIZE number of updates have happened 
        return
    transitions = memory.sample(BATCH_SIZE) # draws a random set of transitions; the next_state for terminal transition will be NONE
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions)) # turn [transition, (args)] array into [[transitions], [states], [actions], ... ]

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s != None, batch.next_state)), device=device, dtype=torch.bool) # returns a set of booleans
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # creates a list of non-empty next states
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # My understanding (for the original explanation check tutorial): 
    # Policy_net produces [[Q1,...,QN], ...,[]] (BATCH x N)-sized matrix, where N is the size of action space, 
    # and action_batch is BATCH-sized vector whose values are the actions that have been taken. 
    # Gather tells which Q from [Q1,...,QN] row to take, using action_batch vector, and returns BATCH-sized vector of Q(s_t, a) values
    state_action_values = policy_net(state_batch).gather(1, action_batch) # input = policy_net, dim = 1, index = action_batch

    # My understanding (for the original explanation check tutorial): 
    # Compute Q^\pi(s_t,a) values of actions for non_final_next_states by using target_net (old policy_net), from which max_a{Q(s_t, a)} are selected with max(1)[0].
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0] # target_net produces a vector of Q^pi(s_t+1,a)'s and max(1)[0] takes maxQ
    # Compute the expected Q^pi(s_t,a) values for all BATCH_SIZE (default=128) transitions
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def do_training(num_episodes, L, sim_duration, forw_rate_increment):
    # training will be done for a few target droplet sizes that is represented by a fraction of completely surrounded particles
    # in the steady state
    for train in range(10): # let's train the network for 10 randomly chosen target blocked fractions
        target_fraction = random.uniform(0,0.8) # doesn't look like it's possible to go beyond 0.8
        print("target fraction of blocked particles", target_fraction)
        # we choose different initial conditions every episode
        for i_episode in range(num_episodes):
            print("Episode ", i_episode)
            # initial conditions
            density = random.uniform(0, 0.5)
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
            for t in range(1000):
                for n in range(N):
                    update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

            blocked_fraction = count_blocked_particles(lattice, L) / N
            print("system relaxed from its initial conditions; current fraction of blocked particles ", blocked_fraction)

            # adjust parameters every 1000 time steps to let the system relax to its steady state
            score = 0
            for update_params in range(100):

                state = [density, translate_along_rate, blocked_fraction, target_fraction] 
                #print("state before ", state)
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = select_action_training(state, n_actions) # increase or decrease the forward jumping rate
                delta = +forw_rate_increment if action == 0 else -forw_rate_increment
                
                translate_along_rate += delta
                rotation_rate = (1 - translate_along_rate) / (3*alpha + 2)
                translate_transverse = alpha * rotation_rate
                translate_opposite_rate = translate_transverse

                # relax to the steady state; here i assume that after 500 updates the system would reach the new steady state
                for t in range(500):
                    for n in range(N):
                        update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

                blocked_fraction = count_blocked_particles(lattice, L) / N      # new fraction of blocked particles

                #print("action ", action.item())
                reward = - abs(blocked_fraction - target_fraction)
                reward = torch.tensor([reward], device=device)
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

            print("fraction of blocked particles after training ", blocked_fraction, "; target fraction ", target_fraction)
            print("final rates: v_+ =  ", translate_along_rate, "; v_0 = ", translate_transverse, "; D_r = ", rotation_rate)
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
    num_episodes = 100      # number of training episodes
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
    n_observations = 5      # just give network a difference between positive and negative spins
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
        steps_done = 0 
        do_training(num_episodes, L, sim_duration, forw_rate_increment) 

    ############# Training summary ######################
    trained_net = DQN(n_observations, hidden_size, n_actions).to(device)
    trained_net.load_state_dict(torch.load(PATH))

    ############# Post-training simulation ##############

    

    #filename = "2d_encourage_data_L" + str(L) + "_bigL" + str(newL) + "_runs" + str(runs) + ".txt"
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
