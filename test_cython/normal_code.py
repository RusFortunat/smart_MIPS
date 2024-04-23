# MIPS on 2d lattice; the model is taken from: 
# https://pubs.aip.org/aip/jcp/article/148/15/154902/195348/Phase-separation-and-large-deviations-of-lattice
import math
import numpy as np
import random
import time
from environment import run as c_run
from environment import count_blocked as c_count_blocked

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

if __name__ == '__main__':
    # simulation parameters
    L = 100
    sim_duration = 10000
    runs = 10

    for i in range(runs):

        density = random.uniform(0.1, 0.5)
        N = int(L*L*density)
        # to set the rates, i will use the original model reference that i mention at the top
        translate_along_rate = random.uniform(0, 0.95)
        alpha = 0.035 / 0.0071 # took form paper; alpha = translate_transverse / rotation_rate, with rotation_rate=0.1, translate_transverse=1, translate_along_rate=25
        rotation_rate = (1 - translate_along_rate) / (3*alpha + 2)
        translate_transverse = alpha * rotation_rate
        translate_opposite_rate = translate_transverse
        print("density ", round(density,2), "v_+ =  ", translate_along_rate, "; v_0 = ", translate_transverse, "; D_r = ", rotation_rate)
        
        # Cython execution
        c_particles = []
        c_lattice = np.zeros(shape=(L, L))
        n = 0
        while n < N:
            X = random.randint(0, L-1)
            Y = random.randint(0, L-1)
            if c_lattice[X][Y] == 0: 
                c_lattice[X][Y] = 1
                angle = random.randint(0,3)
                entry = [X,Y,angle] 
                c_particles.append(entry)
                n += 1 
                
        start_time = time.time()
        c_run(c_lattice, c_particles, sim_duration, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)
        c_blocked = c_count_blocked(c_lattice, L)  
        print("Cython execution time %s seconds" % (time.time() - start_time), "; blocked fraction ", c_blocked)
        
        # python execution
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
        
        
        start_time = time.time()
        for t in range(sim_duration):
            for n in range(N):
                update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

        blocked = count_blocked_particles(lattice, L) / N
        print("Python execution time %s seconds" % (time.time() - start_time), "; blocked fraction ", blocked)
            
