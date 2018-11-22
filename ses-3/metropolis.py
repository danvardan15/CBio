import numpy as np
import pyrosetta
import matplotlib.pyplot as plt
from pyrosetta import PyMOLMover
from pyrosetta import SwitchResidueTypeSetMover

# init pyrosetta and load score3 function
pyrosetta.init()
SCORE3 = pyrosetta.create_score_function("score3")

# function to initialise a pose
def create_pose(seq='DAYAQWLKDGGPSSGRPPPS'):
    pose = pyrosetta.pose_from_sequence(seq)
    # set given initial conditions
    for res in range(1, pose.total_residue() +1):
        pose.set_phi(res, -150)
        pose.set_psi(res, 150)
        pose.set_omega(res, 180)
    return pose


# functions to create an angle in range [a-da, a+da)
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
        self.q = np.zeros((self.pose.total_residue(), 2))
        self.annealing = annealing and use_criterion
        self.kT0 = kT0
        self.L = np.log(1000) / niter
        self.kT = 1 #default value for no annealing

    def set_score_fn(self, score_fn):
        """Set a score function and calculate initial energy"""
        self.score_fn = score_fn
        self.save_angles(self.pose)
        self.E = self.score_fn(self.pose)
        self.best_energy = self.E

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
        Phi or psi angles of a pose's residue are modified.
        Changes persist if its score is accepted, according to
        energy difference and metropolis criterion.
        """
        candidate = self.pose.clone()
        if np.random.randint(0, 2):
            candidate.set_phi(res, new_angle(candidate.phi(res), self.max_add))
        else:
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

        if E2 < self.best_energy:
            self.save_angles(self.pose, res)
            self.best_energy = E2


    def metropolis(self):
        """
        Monte Carlo metropolis method.
        Choosing a random residue, the pose is altered and accepted according to the energy.
        Energy evolution is saved for plotting reasons.
        """
        for i in range(self.niter):
            if not (i % 100 ): print("iter {}".format(i))
            residue = np.random.randint(1, self.pose.total_residue() + 1)
            self.alter_pose(residue, i)
            self.energy_course[i] = self.E
        self.pmover.apply(self.pose)

    def save_angles(self, pose, res=None):
        """
        Store angles of pose, either for one res or for all pose
        """
        if res: self.q[res - 1] = (pose.phi(res), pose.psi(res))
        for res in range(1, pose.total_residue() +1):
            self.q[res - 1] = (pose.phi(res), pose.psi(res))
        return pose

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
        # movers needed to convert to centroid and full atom
        movers = {
        "pmover": PyMOLMover(),
        "centroid": SwitchResidueTypeSetMover("centroid"),
        "fa_mover": SwitchResidueTypeSetMover("fa_standard")
        }
        # initialise trajectories with clone of initial pose
        for i in range(ntraj):
            tr = Trajectory(i, pose.clone(), movers, niter, use_criterion, annealing, max_add, kT0)
            tr.set_score_fn(tr.centroid_score)
            self.trajectories.append(tr)

    def run_metropolis(self):
        """
        Run MC metropolis for every trajectory, keeping track of final energy,
        and showing the best pose with the configuration that yielded the least energy
        """
        for i, traj in enumerate(self.trajectories):
            print("Trajectory {}:\n".format(i + 1))
            traj.metropolis()
            self.final_energies[i] = traj.best_energy

        argbest = np.argmin(self.final_energies)
        best = self.trajectories[argbest]
        for i in range(1, best.pose.total_residue() + 1):
            phi, psi = best.q[i - 1]
            best.pose.set_phi(i, phi)
            best.pose.set_psi(i, psi)
        best.pmover.apply(best.pose)

    def plot_energy_courses(self):
        """plot evolution of energies of every trajectory"""
        plt.figure()
        for traj in self.trajectories:
            traj.plot_energy_course()
        plt.title("Energy evolution")
        plt.legend()
        plt.show()

    def plot_mean_course(self):
        """plot evolution of energies averaged over trajectories """
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
        """plot final energies for every trajectory"""
        plt.figure()
        plt.xlabel("Trajectory")
        plt.ylabel("Final energy")
        plt.title("Comparison of trajectories")
        plt.plot(self.final_energies)
        plt.show()

    def print_stats(self):
        """print best worse and median of solutions"""
        argbest = np.argmin(self.final_energies)
        argworse = np.argmax(self.final_energies)
        best = self.final_energies[argbest]
        worse = self.final_energies[argworse]
        print("Best energy: {} from trajectory {}".format(best, argbest + 1))
        print("Worse energy: {} from trajectory {}".format(worse, argworse + 1))
        print("Median energy: {}".format(np.median(self.final_energies)))


if __name__ == "__main__":
    # MC with metropolis criterion
    f1 = Folding(create_pose, 1000, 10, True)
    f1.run_metropolis()
    f1.print_stats()
    # MC without metropolis criterion
    f2 = Folding(create_pose, 1000, 10, False)
    f2.run_metropolis()
    f2.print_stats()
    # Simulated annealing
    f3 = Folding(create_pose, 1000, 10, True, True)
    f3.run_metropolis()
    f3.print_stats()
    # MC with metropolis criterion, more iterations
    f4 = Folding(create_pose, 5000, 5, True)
    f4.run_metropolis()
    f4.print_stats()

    #plots
    f1.plot_energy_courses()
    f1.plot_final_energies()
    f1.plot_mean_course()

    f2.plot_energy_courses()
    f2.plot_final_energies()
    f2.plot_mean_course()

    f3.plot_energy_courses()
    f3.plot_final_energies()
    f3.plot_mean_course()

    f4.plot_energy_courses()
    f4.plot_final_energies()
    f4.plot_mean_course()