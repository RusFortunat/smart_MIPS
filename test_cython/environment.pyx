#cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: language = c++
# distutils: extra_compile_args = -std=c++11

from libc.stdio cimport printf

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed
    
    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(mt19937 gen) # ignore the possibility of using other classes for "gen"

def my_rng():
    cdef:
        mt19937 gen = mt19937(5)
        uniform_real_distribution[double] dist = uniform_real_distribution[double](0.0,1.0)
    return dist(gen)


cdef int count_blocked_c(int [:, :] lattice, int L):
    cdef int sum = 0
    cdef int nextX, nextY, prevX, prevY
    for X in range(L):
        for Y in range(L):
            nextX = X + 1 if X < L - 1 else 0
            prevX = X - 1 if X > 0 else L - 1
            nextY = Y + 1 if Y < L - 1 else 0
            prevY = Y - 1 if Y > 0 else L - 1
            if lattice[nextX][Y] == 1 and lattice[prevX][Y] == 1 and lattice[X][nextY] == 1 and lattice[X][prevY] == 1:
                sum += 1 # blocked

    return sum

def count_blocked(int [:, :] lattice, int L):
    return count_blocked_c(lattice, L)


def run(int [:, :] lattice, int [:, :] particles, int simulation_time, int N, int L, double rotation_rate, double translate_along_rate, double translate_opposite_rate, double translate_transverse):
    
    printf("we are inside ;)\n")
    cdef int picked_particle, X, Y, angle, new_angle, newX, newY
    cdef double dice, counter_or_clock_rotation

    for t in range (simulation_time):

        # pick random particle
        picked_particle = <int>((N-1) * my_rng())
        printf("picked particke %d\n", picked_particle)
        X = particles[picked_particle][0]
        Y = particles[picked_particle][1]
        angle = particles[picked_particle][2]

        # COUNTERCLOCKWISE ANGLE DIRECTION STARTING FROM +X DIRECTION
        # 0 -- +X, 1 -- +Y, 2 -- -X, 3 -- -Y

        dice = my_rng()
        printf("dice %f\n", dice)
        # rotate
        if dice < 2*rotation_rate: 
            counter_or_clock_rotation = my_rng()
            new_angle = 0
            if counter_or_clock_rotation < 0.5: # +pi/2
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

