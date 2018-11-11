#thoughts: move phi or psi, or both?
#going uphill ends up not in optima, restore to best value?

#some attribs of traj could be in folding
#plot mean should be improved

import numpy as np
import pyrosetta
import matplotlib.pyplot as plt
from pyrosetta import PyMOLMover
from pyrosetta import SwitchResidueTypeSetMover

pyrosetta.init()
SCORE3 = pyrosetta.create_score_function("score3")

# function to initialise a pose
def pose_ex2(seq='DAYAQWLKDGGPSSGRPPPS'):
    pose = pyrosetta.pose_from_sequence(seq)
    for res in range(1, pose.total_residue() +1):
        pose.set_phi(res, -150)
        pose.set_psi(res, 150)
        pose.set_omega(res, 180)
    return pose


# functions to manipulate angles
new_angle = lambda a, da: modular_angle(np.random.randint(a-da, a+da))

def modular_angle(n):
    """Keep angle in [-180,180] range"""
    if np.abs(n) > 180: n = -np.sign(n) * 360 + n
    return n 

class Trajectory:
    def __init__(self, tr_id, pose, movers, niter, use_criterion, annealing, max_add, kT0):
        """
        Start with: pose (Pose), movers (dictionary of Movers), 
                    number of iterations (int), use_criterion (boolean), annealing (boolean),
                    maximum angles to move a residue (float), k*T_0 (float)
        Note:   Simulated annealing is only performed if
                both the use_criterion, and the annealing flags are True
        For annealing parameters refer to: func:`~Trajectory.anneal`
        kT0 and L were calculated to fit:
        kT(0) = 100 and kT(n) = 0.1; kT(x) = kT0 * exp(-Lx)
        """
        self.tr_id = tr_id
        self.pose = pose
        self.pmover = movers["pmover"]
        self.centroid = movers["centroid"]
        self.fa_mover = movers["fa_mover"]
        self.niter = niter
        self.use_criterion = use_criterion
        self.max_add = max_add
        self.energy_course = np.zeros(niter)
        
        self.annealing = annealing and use_criterion
        self.kT0 = kT0
        self.L = np.log(1000) / niter
        self.kT = 1 #default value for no annealing

    def set_score_fn(self, score_fn):
        """Set a score function and calculate initial energy"""
        self.score_fn = score_fn
        self.E = self.score_fn(self.pose)

    def centroid_score(self, pose):
        """
        change to centroid representation for scoring
        recover full atom representation and return score3
        """
        self.centroid.apply(pose)
        score = SCORE3(pose)
        self.fa_mover.apply(pose)
        return score

    def anneal(self, curr_iter):
        """
        Exponential decay of annealing.
        """
        self.kT = self.kT0 * np.exp(-self.L * curr_iter)

    def metropolis_crit(self, dE):
        return np.exp(-dE / self.kT)

    def alter_pose(self, res, curr_iter):
        """
        Phi and psi angles of a pose's residue are modified.
        Changes persist its scored is accepted according to
        energy difference and metropolis criterion.
        """
        candidate = self.pose.clone()
        candidate.set_phi(res, new_angle(candidate.phi(res), self.max_add))
        candidate.set_psi(res, new_angle(candidate.psi(res), self.max_add))
        E2 = self.score_fn(candidate)
        # acceptance criteria
        dE = E2 - self.E
        if dE < 0:
            self.pose, self.E = candidate, E2
            self.pmover.apply(self.pose)
        if self.annealing: self.anneal(curr_iter)
        if self.use_criterion and np.random.random() < self.metropolis_crit(dE):
            self.pose, self.E = candidate, E2
            self.pmover.apply(self.pose)

    def metropolis(self):
        """
        Monte Carlo metropolis method.
        Choosing a random residue, the pose is altered and accepted according to the energy.
        Energy evolution is saved for plotting reasons.
        """
        for i in range(self.niter):
            residue = np.random.randint(1, self.pose.total_residue() + 1)
            self.alter_pose(residue, i)
            self.energy_course[i] = self.E

    def plot_energy_course(self):
        """plot energy evolution"""
        plt.plot(self.energy_course, label=self.tr_id + 1)
        plt.xlabel("Iteration")
        plt.ylabel("Score")


class Folding:
    """
    Class that contains general parameters and the trajectories calculated.
    It also stores results, and computes statistical information.
    """
    def __init__(self, pose_init, niter, ntraj, use_criterion, annealing=False, max_add=25, kT0=100):
        self.trajectories =[]
        self.final_energies = np.zeros(ntraj)
        self.ntraj = ntraj
        self.niter = niter
        pose = pose_init()
        movers = {
        "pmover": PyMOLMover(),
        "centroid": SwitchResidueTypeSetMover("centroid"),
        "fa_mover": SwitchResidueTypeSetMover("fa_standard")
        }
        for i in range(ntraj):
            tr = Trajectory(i, pose.clone(), movers, niter, use_criterion, annealing, max_add, kT0)
            tr.set_score_fn(tr.centroid_score)
            self.trajectories.append(tr)

    def run_metropolis(self):
        """
        Run MC metropolis for every trajectory, keeping track of final energy
        """
        for i, traj in enumerate(self.trajectories):
            print("Trajectory {}:\n".format(i + 1))
            traj.metropolis()
            self.final_energies[i] = traj.E

    def plot_energy_courses(self):
        plt.figure()
        for traj in self.trajectories:
            traj.plot_energy_course()
        plt.legend()
        plt.show()

    def plot_mean_course(self):
        energy_matrix = np.zeros((self.ntraj, self.niter))
        for i, traj in enumerate(self.trajectories):
            energy_matrix[i] = traj.energy_course
        plt.figure()
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Mean energy evolution")
        plt.plot(np.mean(energy_matrix, axis=0))
        plt.show()

    def plot_final_energies(self):
        plt.figure()
        plt.xlabel("Trajectory")
        plt.ylabel("Final energy")
        plt.title("Comparison of trajectories")
        plt.plot(self.final_energies)
        plt.show()

    def print_stats(self):
        argbest = np.argmin(self.final_energies)
        argworse = np.argmax(self.final_energies)
        best = self.final_energies[argbest]
        worse = self.final_energies[argworse]
        print("Best energy: {} from trajectory {}".format(best, argbest + 1))
        print("Worse energy: {} from trajectory {}".format(worse, argworse + 1))
        print("Median energy: {}".format(np.median(self.final_energies)))





f = Folding(pose_ex2, 75, 4, True, True)
f.run_metropolis()
f.print_stats()
f.plot_energy_courses()
f.plot_final_energies()
f.plot_mean_course()
