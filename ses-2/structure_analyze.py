
import sys
import os
import numpy as np
from optparse import OptionParser
import matplotlib
#matplotlib.use('Qt4Agg') #uncomment this line, if you have problems to rotate the proteins in pymol
import matplotlib.pylab as plt
from AnnoteFinder import *
from PymolLauncher import *
import pymol
from pymol import cmd

#Add options for the script here!
def add_options( parser ):
    #Add options one at a time
    parser.add_option("--score_file", type="string", dest="score_file", help="Name of the score file")
    parser.add_option("--native", type="string", dest="native", help="Name of the native .pdb file")
    parser.add_option("--skipheader", type="int", dest="skipheader", help="number of rows to skip")

    #Parse args and get options
    options, args = parser.parse_args()
    
    if options.score_file is None or options.score_file is None:
        parser.print_help()
        sys.exit(1)
    
    return options, args

def read_scores(score_path, skip_header=0):
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
    """ What is the structure with the highest GDT_TS
    among the 25 lowest energy structures?
    What is the median GDT_TS"""
    lowest_25 = sorted([(z[1],z[2]) for z in sorted(zip(y, x, annote))][:25])
    s_best = lowest_25[-1]
    s_median = lowest_25[12]
    return s_best, s_median

def main():
    # create a OptionParser object
    parser = OptionParser()
    options, args = add_options(parser)
    # open a score file with the argument from OptionParser object
    gdts, scores, pdbs = read_scores(options.score_file, options.skipheader)
    ax = plot_scores(gdts, scores)
    s_best, s_median = stats(gdts, scores, pdbs)
    print("among the 25 lowest energy structures:")
    print("best structrure: {1} with gdt_ts {0}".format(*s_best))
    print("Median gdt_ts: {0} from structure {1}".format(*s_median))
  
    pl1 = PymolLauncher( gdts, scores, pdbs, ax )
    pl1.set_native(options.native)
    plt.connect('button_press_event', pl1)
    plt.gca().set_autoscale_on(False)
 
    pymol.finish_launching()
    load_native(pl1)
    plt.show()
    


if __name__ == '__main__':
    sys.exit(main())
else:
    print("Loaded as a module!")   





