# 1d coalescence with DQN
import math
import numpy as np
import random
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def update(particles, lattice, N, L, rotation_rate, translate_along_rate):
    # pick random particle
    picked_particle = random.randint(0,N-1) 
    X = particles[picked_particle][0]
    Y = particles[picked_particle][1]
    if lattice[X][Y] == 0:
        print("Error! Lattice site is empty!")

    rotate_or_translate = random.randint(0,1)
    # rotate
    if rotate_or_translate == 0: 
        dice = random.random(0,1)
        if dice < rotation_rate:
            angle = particles[picked_particle][2]
            left_or_right = random.randint(0,1)
            if left_or_right == 0: # left
                new_angle = angle - 1 if angle != 0 else 3
            else:
                new_angle = angle + 1 if angle != 3 else 0
            particles[picked_particle][2] = new_angle

    # translate
    else: 
        newX = X
        newY = Y
        angle = particles[picked_particle][2]
        dice = random.random(0,1)
        # could be a smarter way to encode all what is below
        if dice < translate_along_rate: # jump along the director
            if angle == 0: # jump to the right
                newX = X + 1 if X < L - 1 else 0
            elif angle == 1: # jump to the left
                newX = X - 1 if X > 0 else L - 1
            elif angle == 2: # jump to the top
                newY = Y + 1 if Y < L - 1 else 0
            else: # jump to the bottom
                newY = Y - 1 if Y > 0 else L - 1
        elif dice < 0.5:                # jump against the director
            if angle == 0: # jump to the right
                newX = X - 1 if X > 0 else L - 1
            elif angle == 1: # jump to the left
                newX = X + 1 if X < L - 1 else 0
            elif angle == 2: # jump to the top
                newY = Y - 1 if Y > 0 else L - 1
            else: # jump to the bottom
                newY = Y + 1 if Y < L - 1 else 0
        elif dice < 0.75:               # jump to the left of the director
            if angle == 0: # jump to the right
                newY = Y + 1 if Y < L - 1 else 0
            elif angle == 1: # jump to the left
                newY = Y - 1 if Y > 0 else L - 1
            elif angle == 2: # jump to the top
                newX = X - 1 if X > 0 else L - 1
            else: # jump to the bottom
                newX = X + 1 if X < L - 1 else 0
        else:                           # jump to the right of the director
            if angle == 0: # jump to the right
                newY = Y - 1 if Y > 0 else L - 1
            elif angle == 1: # jump to the left
                newY = Y + 1 if Y < L - 1 else 0
            elif angle == 2: # jump to the top
                newX = X + 1 if X < L - 1 else 0
            else: # jump to the bottom
                newX = X - 1 if X > 0 else L - 1
        
        if lattice[newX][newY] == 0:
            lattice[X][Y] = 0 # particle leaves the original lattice site
            lattice[newX][newY] = 1 # diffusion; otherwise coalescence
            particles[picked_particle][0] = newX
            particles[picked_particle][1] = newY


def get_image(particles, lattice, N, L):
    # create a 
    im = np.zeros(shape=(L,L, 3), np.uint8)
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
            newX = X - 1 if X > 0 else L - 1
        elif angle == 2: # jump to the top
            newY = Y + 1 if Y < L - 1 else 0
        else: # jump to the bottom
            newY = Y - 1 if Y > 0 else L - 1

        
        if lattice[newX][newY] != 0:
            im[X][Y][0] = 255 # represents red colour; 0 -- blue
        else:
            im[X][Y][2] = 255            

    for x in range(L):
        for y in range(L):
            if lattice[x][y] == 0:
                im[x][y][0] == 255
                im[x][y][1] == 255
                im[x][y][2] == 255
    
    return im

# Main
if __name__ == '__main__':

    # simulation parameters
    runs = 1
    sim_duration = 500
    L = 20
    density = 0.1
    rotation_rate = 0.1
    translate_along_rate = 0.4
    translate_opposite_rate = 0.5 - translate_along_rate
    translate_transverse = 0.25
    N = int(L*L*density)
    print("N = ", N)
    print("parameters chosen")

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
    #print("particles positions and angles")
    #print(particles)

    print("initial conditions set")

    images = []
    image = get_image(particles, lattice, N, L)
    image = Image.fromarray(image)
    images.append(image)
    print("first image recorded")

    for t in range(sim_duration):
        print("timestep ", t)
        for n in range(N):
            update(particles, lattice, N, L, rotation_rate, translate_along_rate)

        # collect system colored snapshot every 10 MCS
        if t % 10 == 0:
            image = get_image(particles, lattice, N, L)
            image = Image.fromarray(image)
            images.append(image)

    # animation
    images[0].save('Users/Ruslan.Mukhamadiarov/Work/smart_MIPS/output.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=40, loop=0)


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



    filename = "2d_random_decay_L" + str(L) + "_runs" + str(runs) + ".txt"
    with open(filename, 'w') as f:
        for t in range(Nt):
            output_string = str(t) + "\t" + str(particles_left[t]) + "\n"
            f.write(output_string)
'''