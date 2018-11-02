
import sys
import os
import numpy
from optparse import OptionParser
import matplotlib
#matplotlib.use('Qt4Agg') #uncomment this line, if you have problems to rotate the proteins in pymol
import matplotlib.pylab as plt
from AnnoteFinder import *
from PymolLauncher import *
from pymol import *


#Add options for the script here!
def add_options( parser ):
    #Add options one at a time
    parser.add_option("--score_file", type="string", dest="score_file", help="Name of the score file")
    parser.add_option("--native", type="string", dest="native", help="Name of the native .pdb file")

    #Parse args and get options
    options, args = parser.parse_args()
    
    if options.score_file is None or options.score_file is None:
        parser.print_help()
        sys.exit(1)
    
    return options, args


def main():
    # create a OptionParser object
    parser = OptionParser()
    options, args = add_options(parser)
    # open a score file with the argument from OptionParser object
    with open(options.score_file, 'r') as sf:
        print('\nTODO: implement code to read the file, \ngenerate the scatterplot, and \nvisualize predictions in pymol')
    
    """
    # Your code here!
    # Implement code that reads the score file and produces a energy vs. GDT_TS scatter plot. Don't forget to label the axes!
    """
   
    
    """ Code that links the data to the PymolLauncher Object
        Uncomment this if you have edited this file and are ready 
        to link the PymolLauncher to the scatter plot"""
    #pl1 = PymolLauncher( gdts, scores, pdbs )
    #pl1.set_native(options.native)
    #plt.connect('button_press_event', pl1)
    #plt.gca().set_autoscale_on(False)
 
    #import pymol
 
    #pymol.finish_launching()
    #from pymol import cmd
    #plt.show()
    """/end Code that links the data to PymolLauncher"""
    


if __name__ == '__main__':
    sys.exit(main())
else:
    print("Loaded as a module!")   





