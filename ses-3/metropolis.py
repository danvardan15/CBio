import numpy as np
import pyrosetta
import matplotlib.pyplot as plt
from pyrosetta import PyMOLMover
from pyrosetta import SwitchResidueTypeSetMover

pyrosetta.init()
SCORE3 = pyrosetta.create_score_function("score3")

def simple_pose():
    # Create a dummy pose with certain initial angles
    pose = pyrosetta.pose_from_sequence('A'*10)
    for res in range(1, pose.total_residue() +1):
        pose.set_phi(res, -150)
        pose.set_psi(res, 150)
        pose.set_omega(res, 180)
    return pose

def pose_ex2():
    return pyrosetta.pose_from_sequence('DAYAQWLKDGGPSSGRPPPS')

def modular_angle(n):
    if np.abs(n) > 180: n = -np.sign(n) * 360 + n
    return n 

new_angle = lambda a, da: modular_angle(np.random.randint(a-da, a+da))

metropolis_crit = lambda dE, kT=1: np.exp(-dE / kT)


def centroid_score(pose, movers):
    movers["centroid"].apply(pose)
    score = SCORE3(pose)
    movers["fa"].apply(pose)
    return score

def limited_move(pose, res, score_fn, E1, movers, met_flag=False, max_add=25):
    # create a candidate and modify it
    candidate = pose.clone()
    r1 = new_angle(candidate.phi(res), max_add)
    r2 = new_angle(candidate.psi(res), max_add)
    candidate.set_phi(res, r1)
    candidate.set_psi(res, r2)
    E2 = score_fn(candidate, movers)
    # decide with energy difference + Met criterion
    dE = E2 - E1
    if dE < 0: return candidate, E2
    if met_flag and np.random.random() < metropolis_crit(dE, kT=1): return candidate, E2
    return pose, E1

def metropolis(pose, score_fn, movers, niter=10):
    # Get initial score
    E = score_fn(pose, movers)
    Es = []
    for i in range(niter):
        residue = np.random.randint(1, pose.total_residue() + 1)
        new_pose, new_E = limited_move(pose, residue, score_fn, E, movers)
        if E != new_E:
            # If there is a change in energy, accept changes
            pose = new_pose
            movers["pymol"].apply(pose)
            E = new_E
        Es.append(E)
    return pose, Es

movers = {
 "pymol":PyMOLMover(),
 "centroid": SwitchResidueTypeSetMover("centroid"),
 "fa": SwitchResidueTypeSetMover("fa_standard")
 }

pose = pose_ex2()
print("Initial energy: {}".format(centroid_score(pose, movers)))
movers["pymol"].apply(pose)
pose, Es = metropolis(pose, centroid_score, movers, 1000)

print("Final energy: {}".format(centroid_score(pose, movers)))
plt.figure()
plt.plot(Es)
plt.xlabel("Iteration")
plt.ylabel("Score 3")
plt.show()

