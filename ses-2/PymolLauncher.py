import math

import pylab
import matplotlib
from AnnoteFinder import * 
import os 
from pymol import cmd

"""
Inherents from AnnoteFinder
"""
class PymolLauncher(AnnoteFinder):
    def set_native(self, native):
        self.native = native
        self.native_name = native[-8:-4]
    """
    This is a derived class from AnnoteFinder! 
  
    Overwritten draw annote, extended to start pymol and show predicted/native structures
    
    Edit this file! Implement a functionallity that shows the superimposed native structure and the prediction in pymol
    you can use PyMol commands you already know in python style ( cmd.your_command(arguments) ) to achieve this! 
    
    Make sure that your structures are differently colored, in cartoon representation and zoomed in!
    
    """
    def drawAnnote(self, axis, x, y, annote):
        
        if (x,y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x,y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.axis.figure.canvas.draw()
        
        else:
           """
           Mark data point and show data
           """
           t = axis.text(x,y, "(%3.2f, %3.2f) - %s"%(x,y,annote), )
           m = axis.scatter([x],[y], marker='d', c='r', zorder=100)
           self.drawnAnnotations[(x,y)] =(t,m)
           self.axis.figure.canvas.draw()

        # hide previous structures and load decoy
        # since native was loaded already, colour is always different
        cmd.hide("all")
        cmd.load(self.native_name + '/' + annote + '.pdb')
        # show, align, and zoom
        cmd.show("cartoon", self.native_name + ", " + annote)
        result = cmd.align(self.native_name, annote)
        cmd.zoom(self.native_name)
        print("Align of {} and {}, rmsd: {}".format(self.native_name, annote, result[0]))
