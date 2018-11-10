import numpy as np
import pyrosetta
import matplotlib.pyplot as plt
from pyrosetta import PyMOLMover

pyrosetta.init()

def simple_pose():
    # Create a dummy pose with certain initial angles
    pose = pyrosetta.pose_from_sequence('A'*50)
    for res in range(1, pose.total_residue() +1):
        pose.set_phi(res, -150)
        pose.set_psi(res, 150)
        pose.set_omega(res, 180)
    return pose

def modular_angle(n):
    if np.abs(n) > 180: n = -np.sign(n) * 360 + n
    return n 

new_angle = lambda a, da: modular_angle(np.random.randint(a-da, a+da))

metropolis_crit = lambda dE, kT=1: np.exp(-dE / kT)

def limited_move(pose, res, score_fn, max_add=25):
    E1 = score_fn(pose)
    # create a candidate and modify it
    candidate = pose.clone()
    r1 = new_angle(candidate.phi(res), max_add)
    r2 = new_angle(candidate.psi(res), max_add)
    candidate.set_phi(res, r1)
    candidate.set_psi(res, r2)
    E2 = score_fn(candidate)
    # decide with energy difference + Met criterion
    dE = E2 - E1
    if dE < 0: return candidate, E2
    if np.random.random() < metropolis_crit(dE, kT=1): return candidate, E2
    return pose, E1


def metropolis(pose, score_fn, pmover, niter=10):
    # Get initial score
    c = 0
    E = score_fn(pose)
    Es = []
    for i in range(niter):
        residue = np.random.randint(1, pose.total_residue() + 1)
        new_pose, new_E = limited_move(pose, residue, score_fn)
        if E != new_E:
            # If there is a change in energy, accept changes
            pose = new_pose
            pmover.apply(pose)
            E = new_E
            c += 1
            Es.append(E)
    return pose, Es

pmover = PyMOLMover()
score3 = pyrosetta.create_score_function("score3")

pose = simple_pose()
print("Initial energy: {}".format(score3(pose)))
pmover.apply(pose)
pose, Es = metropolis(pose, score3, pmover, 1000)

print("Final energy: {}".format(score3(pose)))
plt.figure()
plt.plot(Es)
plt.xlabel("Iteration")
plt.ylabel("Score 3")
plt.show()

