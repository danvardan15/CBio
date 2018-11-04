import sys
import matplotlib.pylab as plt
import numpy as np
import math
from PymolLauncher import *
import pymol
from pymol import cmd

def read_scores(score_path, skip_header):
    data = np.genfromtxt(score_path, dtype=object, skip_header=skip_header)
    mask = np.logical_or(data[0]==b'score', data[0]==b'gdtmm')
    mask[-1] = True
    mat = data[:, mask]
    energy, gdt_ts, name = mat[1:,0], mat[1:,1], mat[1:,2] 
    energy = energy.astype('float')
    gdt_ts = gdt_ts.astype('float')
    name = name.astype('str')
    return gdt_ts, energy, name

def plot_scores(x, y):
    f, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("GDT_TS vs. Energy of protein") #specify
    ax.set_xlabel("GDT_TS [gdt_ts units]") # units?
    ax.set_ylabel("Energy [pJ / mol]") # units?
    ax.set_xlim(0.0, 1.0)
    return ax

def load_native(pl1):
    cmd.load(pl1.native)

def stats(x, y, annote):
    # What is the structure with the highest GDT_TS
    # among the 25 lowest energy structures?
    # What is the median GDT_TS
    lowest_25 = sorted([(z[1],z[2]) for z in sorted(zip(y, x, annote))][:25])
    s_best = lowest_25[-1][1]
    s_median = lowest_25[12][0]
    return s_best, s_median

@cmd.extend
def analyse(protein='1N0U', skip_header=0):
    score_path = '{}/score.fsc'.format(protein)
    x, y, annote = read_scores(score_path, skip_header)
    ax = plot_scores(x, y)
    s_best, s_median = stats(x, y, annote)
    print("among the 25 lowest energy structures:")
    print("best structrure: {}".format(s_best))
    print("Median gdt_ts: {}".format(s_median))
    pl1 = PymolLauncher(x, y, annote, ax)
    pl1.set_native("{0}/{0}.pdb".format(protein))
    load_native(pl1)
    plt.connect('button_press_event', pl1)
    plt.show()

