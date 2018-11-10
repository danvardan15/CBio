#thoughts: move phi or psi, or both?
#going uphill ends up not in optima, restore to best value?
#are trajectories decoys?

import numpy as np
import pyrosetta
import matplotlib.pyplot as plt
from pyrosetta import PyMOLMover
from pyrosetta import SwitchResidueTypeSetMover

pyrosetta.init()
SCORE3 = pyrosetta.create_score_function("score3")

# methods to initialise a pose
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


#methods to manipulate angles
def modular_angle(n):
    if np.abs(n) > 180: n = -np.sign(n) * 360 + n
    return n 

new_angle = lambda a, da: modular_angle(np.random.randint(a-da, a+da))

#class to fold the poses
class Folding:
    def __init__(self, pose_init, niter, use_criterion, annealing=False, max_add=25, kT0=100):
        self.pose = pose_init()
        self.pmover = PyMOLMover()
        self.centroid = SwitchResidueTypeSetMover("centroid")
        self.fa_mover = SwitchResidueTypeSetMover("fa_standard")
        self.niter = niter
        self.use_criterion = use_criterion
        self.max_add = max_add
        self.energy_course = []
        # variables for simulated annealing
        self.annealing = annealing and use_criterion #ensure annealing if criterion is used
        self.kT0 = kT0
        self.L = np.log(1000) / niter
        self.kT = 1 #default value for no annealing

    def set_score_fn(self, score_fn):
        self.score_fn = score_fn
        self.E = self.score_fn(self.pose)

    def metropolis_crit(self, dE):
        return np.exp(-dE / self.kT)

    def centroid_score(self, pose):
        self.centroid.apply(pose)
        score = SCORE3(pose)
        self.fa_mover.apply(pose)
        return score

    def anneal(self, curr_iter):
        self.kT = self.kT0 * np.exp(-self.L * curr_iter)

    def limited_move(self, res, curr_iter):
        # create a candidate and modify it
        candidate = self.pose.clone()
        candidate.set_phi(res, new_angle(candidate.phi(res), self.max_add))
        candidate.set_psi(res, new_angle(candidate.psi(res), self.max_add))
        E2 = self.score_fn(candidate)
        # decide with energy difference + Met criterion
        dE = E2 - self.E
        if dE < 0:
            self.pose, self.E = candidate, E2
            self.pmover.apply(self.pose)
        if self.annealing: self.anneal(curr_iter)
        if self.use_criterion and np.random.random() < self.metropolis_crit(dE):
            self.pose, self.E = candidate, E2
            self.pmover.apply(self.pose)

    def metropolis(self):
        # Get initial score
        print("Initial energy: {}".format(self.E))
        self.energy_course = []
        for i in range(self.niter):
            residue = np.random.randint(1, self.pose.total_residue() + 1)
            self.limited_move(residue, i)
            self.energy_course.append(self.E)
        print("Final energy: {}".format(self.E))

    def plot_energy_course(self):
        plt.plot(self.energy_course)
        plt.xlabel("Iteration")
        plt.ylabel("Score")


f1 = Folding(pose_ex2, 1000, False)
f1.set_score_fn(f1.centroid_score)
f1.metropolis()

f2 = Folding(pose_ex2, 1000, True)
f2.set_score_fn(f2.centroid_score)
f2.metropolis()

f3 = Folding(pose_ex2, 1000, True, True)
f3.set_score_fn(f3.centroid_score)
f3.metropolis()

plt.figure()
f1.plot_energy_course()
f2.plot_energy_course()
f3.plot_energy_course()
plt.show()