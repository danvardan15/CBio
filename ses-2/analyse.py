import sys
import matplotlib.pylab as plt
import numpy as np
import math
from PymolLauncher import *
import pymol
from pymol import cmd

def read_scores(score_path):
    data = np.genfromtxt(score_path, dtype=object)
    mask = np.logical_or(data[0]==b'score', data[0]==b'gdtmm')
    mask[-1] = True
    mat = data[:, mask]
    energy, gdt_ts, name = mat[1:,0], mat[1:,1], mat[1:,2] 
    energy = energy.astype('float')
    gdt_ts = gdt_ts.astype('float')
    name = name.astype('str')
    return gdt_ts, energy, name

def plot_read_scores(score_path):
    x, y, annote = read_scores(score_path)
    f, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_title("GDT_TS vs. Energy of protein") #specify
    ax.set_xlabel("GDT_TS [gdt_ts units]") # units?
    ax.set_ylabel("Energy [pJ / mol]") # units?
    ax.set_xlim(0.0, 1.0)
    return x, y, annote, ax

def load_native(pl1):
    cmd.load(pl1.native)

@cmd.extend
def analyse(protein='1N0U'):
    score_path = '{}/score.fsc'.format(protein)
    x, y, annote, ax = plot_read_scores(score_path)
    pl1 = PymolLauncher(x, y, annote, ax)
    pl1.set_native("{0}/{0}.pdb".format(protein))
    load_native(pl1)
    plt.connect('button_press_event', pl1)
    plt.show()