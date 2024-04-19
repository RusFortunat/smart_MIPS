# MIPS on 2d lattice; the model is taken from: 
# https://pubs.aip.org/aip/jcp/article/148/15/154902/195348/Phase-separation-and-large-deviations-of-lattice
import math
import numpy as np
import random
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


def get_image(particles, lattice, N, L):
    # create a 
    im = np.zeros(shape=(L,L, 3))
    for n in range(N):
        X = particles[n][0]
        Y = particles[n][1]
        angle = particles[n][2]
        newX = int(X)
        newY = int(Y)
        #print("newX = ", newX, "; newY = ", newY, "; angle = ", angle)
        if angle == 0: # jump to the right
            newX = X + 1 if X < L - 1 else 0
        elif angle == 1: # jump to the left
            newY = Y + 1 if Y < L - 1 else 0
        elif angle == 2: # jump to the top
            newX = X - 1 if X > 0 else L - 1
        else: # jump to the bottom
            newY = Y - 1 if Y > 0 else L - 1

        
        if lattice[newX][newY] != 0:
            im[X][Y][0] = 255 # represents red colour; 0 -- blue
        else:
            im[X][Y][2] = 255            

    for x in range(L):
        for y in range(L):
            if lattice[x][y] == 0:
                im[x][y][0] = 255
                im[x][y][1] = 255
                im[x][y][2] = 255
    
    return im

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

# Main
if __name__ == '__main__':

    # simulation parameters
    runs = 1
    sim_duration = 20000
    L = 200
    density = 0.2
    rotation_rate = 0.0071
    translate_along_rate = 0.8808
    translate_opposite_rate = 0.035
    translate_transverse = 0.035
    sum_rates = rotation_rate*2 + translate_along_rate + translate_opposite_rate + 2*translate_transverse
    if sum_rates != 1:
        print("rates are chosen incorrectly! do not sum up to 1!")
    N = int(L*L*density)
    #print("N = ", N)
    #print("parameters chosen")

    #for run in range(runs):
    #    print("run ", run)
        
    # initial conditions
    #particles = np.zeros(shape=(N,3)) # x,y position, angle, color code
    particles = []
    lattice = np.zeros(shape=(L, L))
    n = 0
    while n < N:
        X = random.randint(0, L-1)
        Y = random.randint(0, L-1)
        if lattice[X][Y] == 0: 
            lattice[X][Y] = 1
            #particles[n][0] = int(X)
            #particles[n][1] = int(Y)
            angle = random.randint(0,3)
            #particles[n][2] = int(angle)
            entry = [X,Y,angle]
            particles.append(entry)
            n += 1
    # check if the number of particles in the list is equal to number of particles in the system
    sum_particles = 0
    for x in range(L):
        for y in range(L):
            if lattice[x][y] == 1:
                sum_particles += 1
    if sum_particles != N:
        print("Error! Number of particles in the list does not fit the number of particles in the system!")

    print("initial conditions set")

    images = []
    image = get_image(particles, lattice, N, L)
    image = Image.fromarray(np.uint8(image))
    images.append(image)
    #print("first image recorded")

    blocked_vs_time = []
    for t in range(sim_duration):
        if t % 100 == 0:
            print("timestep ", t)
        for n in range(N):
            update(particles, lattice, N, L, rotation_rate, translate_along_rate, translate_opposite_rate, translate_transverse)

        blocked = count_blocked_particles(lattice, L) / N
        blocked_vs_time.append(blocked)

        # collect system colored snapshot every 10 MCS
        if t % 100 == 0:
            image = get_image(particles, lattice, N, L)
            image = Image.fromarray(np.uint8(image))
            images.append(image)

    # check if the number of particles haven't changed
    sum_particles = 0
    for x in range(L):
        for y in range(L):
            if lattice[x][y] == 1:
                sum_particles += 1
    if sum_particles != N:
        print("Number of particles changed!")

    # animation
    filename = "./output_normal/density_" + str(density) + "_L_" + str(L) + "_duration_" + str(sim_duration) + "_p_" + str(translate_along_rate) + "_rot_" + str(rotation_rate) + ".gif"
    images[0].save(filename,
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)

    filename2 = "./output_normal/blocked_vs_t_den_" + str(density) + "_L_" + str(L) + "_T_" + str(sim_duration) + "_p_" + str(translate_along_rate) + "_rot_" + str(rotation_rate) + ".txt"
    with open(filename2, 'w') as f:
        for t in range(sim_duration):
            output_string = str(t) + "\t" + str(blocked_vs_time[t]) + "\n"
            f.write(output_string)

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