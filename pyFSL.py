#!/usr/bin/env python

#  This python script implements a 1-D version of the FSL9 code, and reads in FSL9-compatible files. 

#   Derived from FiPy-2.0.2/examples/diffusion/mesh1d.py and other codes

#from fipy import *
from fipy import TransientTerm,ImplicitDiffusionTerm,ExplicitDiffusionTerm,CellVariable
from fipy import FaceVariable, PeriodicGrid1D, DiffusionTerm, ExponentialConvectionTerm
from fipy import DefaultAsymmetricSolver, Viewer
from fipy.tools.numerix import cos,pi
from fipy.tools import numerix
import matplotlib.pyplot as plt
plt.ion()

import os
#from time import process_time, time, sleep
from sys import exit

from pyFSLutils import runtime, FSLpars

#====================================


class FSL1Dsim():
    """A class to faciliate executing simulations of Frost-Strathmann-Lessard (FSL) consumer-resource
       dynamics in spatially and temporally heterogeneous 1D landscapes.
    """
    def __init__(self,pars=None,input_file=None,num_method=2,figsize=(12,9)):
        """Create an FSLsim instance.
        """
        self.num_method = num_method
        # If requested, record the parameters object or load it from a file
        if pars is not None:
            self.pars = pars
            self.input_file = self.pars.input_file
        elif input_file is not None:
            self.input_file = input_file
            self.pars = FSLpars(input_file=input_file)
        # Create a graphics window and add axes for separate plots
        self.Ffig = plt.figure(figsize=(12,9))
        #self.Ffig.tight_layout()
        #self.Qax = self.Ffig.add_subplot(311)
        #self.Qax2 = self.Qax.twinx()
        #self.Cax = self.Ffig.add_subplot(313)
        #self.Hax = self.Ffig.add_subplot(323)
        #self.Vax = self.Ffig.add_subplot(324)
                
      
    def setup(self):
        """Set up variables and fields for starting a simulation using current parameters.
        """
        # Create a flag to interupt execution
        self.halt = False
        # Create a flag to indicate completion of the run
        self.completed = False
        # Make a shortcut for the timestep
        self.steps = int(self.pars.endTime/self.pars.dt)
        #  Resolution and domain size
        self.dx = self.pars.Lx/self.pars.Nx
        # Initialize time variables
        self.t = 0.           # simulation timekeeping
        self.time_plot = 0.   # plot timekeeping
        self.time_series = 0. # stats timekeeping
        self.Ffig.clf()
        self.ax = self.Ffig.add_subplot(212)
        self.ax2 = self.Ffig.add_subplot(211)
        #  Create mesh
        self.mesh = PeriodicGrid1D(dx = self.dx, nx = self.pars.Nx)
        self.x = self.mesh.faceCenters()
        self.X = self.mesh.cellCenters()
        #  Create resource and consumer variables
        self.R = CellVariable(name="resource", 
                         mesh=self.mesh,
                         value= self.pars.phyto5, hasOld=1)
        self.Z = CellVariable(name="consumer1", 
                         mesh=self.mesh,
                         value=self.pars.N, hasOld=1)
        self.rrr = CellVariable(name="tempr", 
                           mesh=self.mesh,
                           value= 0., hasOld=1)
        self.sss = CellVariable(name="temps", 
                           mesh=self.mesh,
                           value= 0., hasOld=1)
        self.UconvCoeff = FaceVariable(mesh=self.mesh, value=0.0, rank=1)
        self.sourceCoeff = CellVariable(mesh=self.mesh, value=1.0)
        #  Lists for time series of averages
        self.mean_t = []
        self.meanZ = []
        self.meanR = []
        self.meanRR = []
        self.meanZZ = []
        self.meanRZ = []
        #self.mean_t = [0.]
        #self.meanZ = [1.]
        #self.meanR = [1.]
        #self.meanRR = [1.]
        #self.meanZZ = [1.]
        #self.meanRZ = [1.]

        self.tmp_output_file = self.pars.directory+'/'+'py3fsl.ser'
        self.out_file = open(self.tmp_output_file,"w")
    
    def stats(self):
        """A method to streamline graphical output.
        """
        # set time for next stats
        self.time_series += self.pars.seriesInterval
        #  Collect statistics
        self.mean_t.append(self.t)
        self.meanR.append(numerix.array(self.R.cellVolumeAverage()))
        self.meanZ.append(numerix.array(self.Z.cellVolumeAverage()))
        # Calculate and record spatial correlations
        self.rrr.setValue(self.R*self.R)
        self.meanRR.append(numerix.array(self.rrr.cellVolumeAverage()))
        self.rrr.setValue(self.Z*self.Z)
        self.meanZZ.append(numerix.array(self.rrr.cellVolumeAverage()))
        self.rrr.setValue(self.Z*self.R)
        self.meanRZ.append(numerix.array(self.rrr.cellVolumeAverage()))
        # Output to the screen
        print('time = ',self.t, ' Z_avg = ',self.meanZ[-1], ' R_avg = ',self.meanR[-1],
              ' RR_avg = ',self.meanRR[-1],' ZZ_avg = ',self.meanZZ[-1], ' RZ_avg = ',self.meanRZ[-1])
        #  Output to file
        out_str = f'{self.t},{self.meanR[-1]},{self.meanZ[-1]},{self.meanRR[-1]},{self.meanZZ[-1]},{self.meanRZ[-1]}'
        self.out_file.write(out_str)

    
    def plot(self):
        """A method to streamline graphical output.
        """
        # set time for next plots
        self.time_plot += self.pars.plotInterval
        #  Create plot viewers for spatial distributions and time series of averages
        #self.ax = self.Ffig.add_subplot(212)
        self.ax.cla()
        self.lineZ, = self.ax.plot(self.mean_t,self.meanZ,color='b')
        self.lineR, = self.ax.plot(self.mean_t,self.meanR,color='g')
        self.lineRZ, = self.ax.plot(self.mean_t,self.meanRZ,color='r')
        self.ax.axhline(y=1.,color='k')
        self.ax.set_xlim(self.mean_t[0],self.mean_t[-1])
        #self.ax.axhline(y=0.,color='k')
        #self.ax.relim()            # reset axes limits
        #self.ax.autoscale_view()
        #plt.pause(0.01)
        #self.ax.set_ylim(auto=True)
        self.ax.set_ylim(bottom=0.)
        self.ax.set_ylabel('Spatial averages')
        self.ax.set_xlabel('Time (Z - blue; R - green; R*Z - red)')

        self.ax2.cla()
        self.linexz, = self.ax2.plot(self.X[0],self.Z.value,color='b')
        self.linexr, = self.ax2.plot(self.X[0],self.R.value,color='g')

        self.ax2.relim()            # reset axes limits
        self.ax2.autoscale_view()   # rescale axes
        #        self.Ffig.canvas.draw()                 # redraw the canvas
        self.ax2.set_ylabel('Population')
        self.ax2.set_xlabel('Position, x')
        self.ax2.legend((self.linexz,self.linexr,), ('Z','R',),loc='upper right')
        #ax2.legend((linexz,linexr,), ('Z','R',),0)
        self.ax2.set_xlim(0,self.pars.Lx)
        self.ax2.set_ylim(0,8)
        
        self.Ffig.suptitle(f'file: {self.pars.filename}   Time: {self.t:.2f}   endTime: {self.pars.endTime:.2f}')
        plt.pause(0.05)
        self.Ffig.canvas.draw()                 # redraw the canvas
        
    def run(self):
        """Execute a simulation using current parameters.
        """
        # Begin run timekeeping
        runtime(init=True)
        # shortcuts to improve legibility
        Fr = self.pars.Fr
        Str = self.pars.Str
        Le = self.pars.Le
        alpha4 = self.pars.alpha4
        sigma_hat = self.pars.sigma_hat
        phyto3 = self.pars.phyto3
        phyto5 = self.pars.phyto5
        psi = self.pars.psi
        N = self.pars.N
        Z = self.Z
        R = self.R
        rrr = self.rrr
        #  Main loop
        for step in range(self.steps):
            #  Calculate taxis/advection terms
            self.UconvCoeff.setValue((self.pars.Fr*self.pars.alpha4/2.)*self.R.faceGrad()[0])
            self.rrr.setValue(self.R/(self.R+self.pars.sigma_hat-1.) )
            self.sss.setValue((0.5*(1.-cos(2.*pi*(self.X/self.pars.Lx-self.pars.x_src))))**self.pars.phyto4 )
            sss_avg = self.sss.cellVolumeAverage()
            self.sourceCoeff.setValue( (self.pars.phyto1/sss_avg)*self.sss )
            # Implement the numerical method: 1 - explicit, 2 - implicit, 3 - Crank-Nicholson
            eqz = 0
            eqr = 0
            eqs = [[Z, 0.],[R, 0.]]
            match self.num_method:
                case 1 | 3:
                    eqXz = TransientTerm() == (ExplicitDiffusionTerm(coeff=self.pars.Dphyto + Fr/(2.*(1.-psi))) -
                                               ExponentialConvectionTerm(coeff=self.UconvCoeff) + (Str*(rrr - 1./sigma_hat))*Z)
                    eqXr = TransientTerm() == (ExplicitDiffusionTerm(coeff=self.pars.Dphyto) - Le*rrr*Z - phyto3*R + self.sourceCoeff)
                    eqs[0][1] += eqXz
                    eqs[1][1] += eqXr
                    #eqs = ((Z, eqXz),(R, eqXr))
                case 2 | 3:
                    eqIz = TransientTerm() == (ImplicitDiffusionTerm(coeff=self.pars.Dphyto + Fr/(2.*(1.-psi))) -
                                               ExponentialConvectionTerm(coeff=self.UconvCoeff) + (Str*(rrr - 1./sigma_hat))*Z)
                    eqIr = TransientTerm() == (ImplicitDiffusionTerm(coeff=self.pars.Dphyto) - Le*rrr*Z - phyto3*R + self.sourceCoeff)
                    eqs[0][1] += eqIz
                    eqs[1][1] += eqIr
                    #eqs = ((Z, eqIz),(R, eqIr))

            for var, eqn in eqs:
                eqn.sweep(var=var,dt=self.pars.dt)
            for var, eqn in eqs:
                var.updateOld()
            #for var, eqn in eqs:
            #    eqn.solve(var, dt = dt)

            # Advance time and call plot/stats are requested
            self.t += self.pars.dt
            if self.t >= self.time_series:
                self.stats()
            if self.t >= self.time_plot:
                self.plot()
            #if R.min() < 0 or Z.min() < 0:
            #    abort_flag = 1
            #    print("******************ABORTING RUN BECAUSE OF NEGATIVE POPULATION******************")
            #    r = R.getValue()
            #    z = Z.getValue()
            #    print("z = ",z)
            #    print("r = ",r)
            #    print("xx = ",xx)
            if self.t >= self.pars.t_src_next/self.pars.phyto2:
                self.pars.load_source(verbose=True)
                #  Terminate run if abort flag is set
                if self.halt:
                    break
            
        self.completed = True
        self.pars.src_file.close()  # Close the source file
        self.out_file.close()  # Close the output file
        # Output run time statistics
        runtime()


