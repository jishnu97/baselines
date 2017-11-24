'''
Creates a simulated environment for Lynx
Author: Jishnu Renugopal

'''
import numpy as np

#Reachable goal coordinates for the end effector
coords = []

def dist(p1, p2):
    return np.linalg.norm(p1-p2)


def getEE(q):
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
        [np.cos(q[3] - (PI / 2)), -np.sin(q[3] - (PI / 2)) * np.cos(-PI / 2), np.sin(q[3] - (PI / 2)) * np.sin(-PI / 2),
         0],
        [np.sin(q[3] - (PI / 2)), np.cos(q[3] - (PI / 2)) * np.cos(-PI / 2), -np.cos(q[3] - (PI / 2)) * np.sin(-PI / 2),
         0],
        [0, np.sin(-PI / 2), np.cos(-PI / 2), 0],
        [0, 0, 0, 1]]
    A5 = [[np.cos(q[4]), -np.sin(q[4]), 0, 0],
          [np.sin(q[4]), np.cos(q[4]), 0, 0],
          [0, 0, 1, L4 + L5],
          [0, 0, 0, 1]]

    A6 = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, L6],
          [0, 0, 0, 1]]

    # Convert to numpy arrays
    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)
    A4 = np.array(A4)
    A5 = np.array(A5)
    A6 = np.array(A6)

    # Get transformation matrix for ee frame
    A1A2 = np.matmul(A1, A2)
    A3A4 = np.matmul(A3, A4)
    A1A4 = np.matmul(A1A2, A3A4)
    eeTransformationMat = np.matmul(A1A4, A5)
    endEffector = np.matmul(eeTransformationMat, np.transpose([0, 0, 0, 1]))

    return endEffector[0:3]


print(getEE([1,2,2,4,5,6]))
