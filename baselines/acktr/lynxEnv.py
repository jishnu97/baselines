'''
Creates a simulated environment for Lynx
Author: Jishnu Renugopal

'''
import numpy as np
from gym import spaces
from random import randint

zeroPose = [263.5250, -0.0000, 222.2500]
validTarget = [304.8000, 0, 222.2500]

# Reachable goal coordinates for the end effector
goal_coords = np.array(
    [[263.5250, -0.0000, 222.2500],
     [2.389962e+02, 1.201340e+02, -1.796832e+02],
     [2.827689e+01, -1.316843e+02, 1.425919e+02],
     [-1.127699e+02, -8.564412e+01, 4.013418e+02],
     [6.984863e+01, -2.578414e+01, -4.989683e+01],
     [5.036049e+01, 1.749105e+01, 4.607722e+02],
     [1.019483e+02, -6.706507e+01, -1.722048e+02],
     [-3.469942e+02, 1.252062e+02, 2.121298e+02],
     [1.052465e+02, 2.239595e+02, 3.399449e+02],
     [-1.230087e+02, -2.323047e+02, 2.524574e+02],
     [6.227686e+01, -1.232393e+02, 2.313400e+02],
     [4.039428e+01, -7.111365e+01, 4.366448e+01],
     [-9.278331e-01, 3.944950e+02, 1.857928e+02],
     [5.130823e+00, -4.017664e+02, 1.438592e+01],
     [-2.057003e+02, 1.924921e+02, 2.970967e+02],
     [-9.008321e+00, 1.646394e+02, 4.443873e+02],
     [-1.417013e+02, -7.271685e-01, 1.801735e+01],
     [9.006752e-01, -1.287634e+02, 2.853707e+02],
     [1.643138e+02, 1.886724e+02, 1.586476e+02],
     [1.294388e+01, 1.708157e+01, -5.429041e+01],
     [1.842947e+01, 7.874060e+01, 2.016529e+02],
     [2.323795e+02, -1.872660e+02, 1.589517e+02],
     [-9.460154e+01, 2.897565e+00, 2.513568e+01],
     [2.250571e+02, 3.189362e+02, 9.583698e+01],
     [8.155004e+01, -9.745030e+01, 1.687526e+02],
     [-2.783849e+02, -2.080288e+02, 2.047963e+02],
     [2.031335e+02, 2.360369e+02, -7.642675e+01],
     [-1.400978e+01, -6.152099e+01, 3.368385e+02],
     [6.200556e+01, 9.377905e-01, -2.287349e+01],
     [-2.088530e+01, 7.173197e+01, 9.774447e+01],
     [3.210790e+02, 1.156620e+02, 1.162568e+02]
     ])
delta = 3

# Limits for co-ordinates
LOW_coor = -500
HIGH_coor = 500

# Limits for joints
LOW_Joint = -2
HIGH_Joint = 2


class Lynx:
    def __init__(self):
        self.t = 0
        self.goal = goal_coords[0]
        self.observation_space = spaces.Box(low=LOW_coor, high=HIGH_coor, shape=(6,))
        self.action_space = spaces.Box(low=LOW_Joint, high=HIGH_Joint, shape=(5,))
        self.metadata = []

    def close(self):
        return

    def dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def getEE(self, q):
        # Constants to improve readability
        L1 = 3 * 25.4
        L2 = 5.75 * 25.4
        L3 = 7.375 * 25.4
        L4 = 1.75 * 25.4
        L5 = 1.25 * 25.4
        L6 = 1.125 * 25.4
        PI = np.pi

        A1 = [[np.cos(q[0]), -np.sin(q[0]) * np.cos(-1 * PI / 2), np.sin(q[0]) * np.sin(-PI / 2), 0],
              [np.sin(q[0]), np.cos(q[0]) * np.cos(-1 * PI / 2), -np.cos(q[0]) * np.sin(-PI / 2), 0],
              [0, np.sin(-PI / 2), np.cos(-PI / 2), L1],
              [0, 0, 0, 1]]

        A2 = [[np.cos(q[1] - (PI / 2)), -np.sin(q[1] - (PI / 2)), 0, L2 * np.cos(q[1] - (PI / 2))],
              [np.sin(q[1] - (PI / 2)), np.cos(q[1] - (PI / 2)), 0, L2 * np.sin(q[1] - (PI / 2))],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

        A3 = [[np.cos(q[2] + (PI / 2)), -np.sin(q[2] + (PI / 2)), 0, L3 * np.cos(q[2] + (PI / 2))],
              [np.sin(q[2] + (PI / 2)), np.cos(q[2] + (PI / 2)), 0, L3 * np.sin(q[2] + (PI / 2))],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]

        A4 = [
            [np.cos(q[3] - (PI / 2)), -np.sin(q[3] - (PI / 2)) * np.cos(-PI / 2),
             np.sin(q[3] - (PI / 2)) * np.sin(-PI / 2),
             0],
            [np.sin(q[3] - (PI / 2)), np.cos(q[3] - (PI / 2)) * np.cos(-PI / 2),
             -np.cos(q[3] - (PI / 2)) * np.sin(-PI / 2),
             0],
            [0, np.sin(-PI / 2), np.cos(-PI / 2), 0],
            [0, 0, 0, 1]]
        A5 = [[np.cos(q[4]), -np.sin(q[4]), 0, 0],
              [np.sin(q[4]), np.cos(q[4]), 0, 0],
              [0, 0, 1, L4 + L5],
              [0, 0, 0, 1]]

        # Convert to numpy arrays
        A1 = np.array(A1)
        A2 = np.array(A2)
        A3 = np.array(A3)

        A4 = np.array(A4)
        A5 = np.array(A5)

        # Get transformation matrix for ee frame
        A1A2 = np.matmul(A1, A2)
        A3A4 = np.matmul(A3, A4)
        A1A4 = np.matmul(A1A2, A3A4)
        eeTransformationMat = np.matmul(A1A4, A5)
        endEffector = np.matmul(eeTransformationMat, np.transpose([0, 0, 0, 1]))

        return endEffector[0:3]

    def step(self, a):
        self.t += 1
        a = np.append(a, 0)
        state = self.getEE(a)
        error = self.dist(state, self.goal)
        state = np.append(state, self.goal)

        if error < delta:
            reward = 1000
            done = True
        else:
            reward = -1 * error if error < 1000 else -1000
            done = False
        if reward:
            with open('reward_vals.txt', 'a') as the_file:
                s = str(self.t)+", "+ str(reward)+", "+ str(reward / self.t)+"\n"
                the_file.write(s)
        return state, reward, done, 0

    def reset(self):
        ind = randint(0, len(goal_coords) - 1)
        self.goal = goal_coords[ind]
        state = self.getEE([0, 0, 0, 0, 0, 0])
        state = np.append(state, self.goal)
        return state
