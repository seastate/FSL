#!/usr/bin/env python

#  This python script implements a 1-D version of the FSL9 code, and reads in modified FSL9-compatible files. 

#   Derived from FiPy-2.0.2/examples/diffusion/mesh1d.py and other codes

#from fipy import *
from fipy import TransientTerm,ImplicitDiffusionTerm,ExplicitDiffusionTerm,CellVariable
from fipy import FaceVariable, PeriodicGrid2D, DiffusionTerm, ExponentialConvectionTerm
from fipy import DefaultAsymmetricSolver#, Viewer, multiViewer
from fipy.tools.numerix import cos,pi
from fipy.tools import numerix
from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.ion()

import os
from sys import exit
from pyFSLm_utils import runtime, FSLpars

#====================================

class Resource():
    """A class to facilitate simulation of FSL resources.
    """
    def __init__(self,pars=None,mesh=None,verbose=False,color='g'):
        """Create a Resource instance.
        """
        self.verbose = verbose
        self.color = color
        # Create the cell variable instance
        self.R = CellVariable(name="resource", mesh=mesh, value= 0., hasOld=1)
        # set parameters, if provided
        if pars:
            self.setpars(pars=pars,init=True)

    def setpars(self,pars=None,init=False,
                 phyto1=None,phyto2=None,phyto3=None,phyto4=None,phyto5=None,Damb=None):
        """Initialize a resource. Parameters can be submitted directly as arguments and/or
           within an FLSparams instance. If there are duplicates, the directly submitted
           parameter is used.
        """
        # set attributes from FSLpars, if provided
        if pars:
            self.phyto1 = pars.phyto1
            self.phyto2 = pars.phyto2
            self.phyto3 = pars.phyto3
            self.phyto4 = pars.phyto4
            self.phyto5 = pars.phyto5
            self.Damb = pars.Damb
        # if provided, keyword arguments supercede FSLpars parameters
        if phyto1:
            self.phyto1 = phyto1
        if phyto2:
            self.phyto2 = phyto2
        if phyto3:
            self.phyto3 = phyto3
        if phyto4:
            self.phyto4 = phyto4
        if phyto5:
            self.phyto5 = phyto5
        if Damb:
            self.Damb = Damb
        if init:
            # Initialize values in R
            self.R.setValue(self.phyto5)


class Consumer():
    """A class to facilitate simulation of FSL consumers.
    """
    def __init__(self,m=0,pars=None,mesh=None,verbose=False,color='b',name='consumer'):
        """Create an FSLsim instance.
        """
        self.m = m
        self.verbose = verbose
        self.color = color
        # Create the cell variable instance
        self.Z = CellVariable(name=name,mesh=mesh, value=0., hasOld=1)
        # set parameters, if provided
        if pars:
            self.setpars(pars=pars,init=True)
    
    def setpars(self,m=None,pars=None,init=False,
                 Fr=None,alpha1=None,alpha4=None,Le=None,Str=None,sigma_hat=None,N=None,psi=None,Damb=None):
        """Initialize a consumer. Parameters can be submitted directly as arguments and/or
           within an FLSparams instance. If there are duplicates, the directly submitted
           parameter is used.
        """
        if m:
            self.m = m
        # set attributes from FSLpars, if provided
        if pars:
            self.Fr = pars.Fr[m]
            self.alpha1 = pars.alpha1[m]
            self.alpha4 = pars.alpha4[m]
            self.Le = pars.Le[m]
            self.Str = pars.Str[m]
            self.sigma_hat = pars.sigma_hat[m]
            self.N = pars.N[m]
            self.psi = pars.psi[m]
            self.Damb = pars.Damb
        # if provided, keyword arguments supercede FSLpars parameters
        if Fr:
            self.Fr = Fr
        if alpha1:
            self.alpha1 = alpha1
        if alpha4:
            self.alpha4 = alpha4
        if Le:
            self.Le = Le
        if Str:
            self.Str = Str
        if sigma_hat:
            self.sigma_hat = sigma_hat
        if Damb:
            self.Damb = Damb
        if N:
            self.N = N
        if psi:
            self.psi = psi
        if init:
            # Initialize values in R
            self.Z.setValue(self.N)


class FSL2Dsim():
    """A class to faciliate executing simulations of Frost-Strathmann-Lessard (FSL) consumer-resource
       dynamics in spatially and temporally heterogeneous 2D landscapes.
    """
    def __init__(self,pars=None,input_file=None,num_method=2,figsize=(12,9),verbose=False,plot_mode='term'):
        """Create an FSLsim instance. If the FSLpars instance (pars) is provided, its parameters
           are used to initialize the simulation. If not, and if input_file is provided, it is
           parsed to obtain pars, which is subsequently used for initialization.
        """
        self.num_method = num_method
        self.verbose = verbose
        # If requested, record the parameters object or load it from a file
        if pars:
            self.pars = pars
            #self.input_file = self.pars.input_file
        elif input_file:
            self.input_file = input_file
            self.pars = FSLpars(input_file=input_file)
        modes = ['term','ipynb']
        if plot_mode in modes:
            self.plot_mode = plot_mode
        else:
            print(f'Unknown plot_mode! Currently supported modes are {modes}')
       # Create a graphics window and add axes for separate plots
        self.Ffig = plt.figure(figsize=(12,9))
        #self.Ffig.tight_layout()
        # set up a list of default colors for multiple consumers variants
        self.colors = ['b','c','gray']
      
    def setup(self):
        """Set up variables and fields for starting a simulation using current parameters, stored
           in the FSLpars instance, pars.
        """
        # Create a flag to interupt execution
        self.halt = False
        # Create a flag to indicate completion of the run
        self.completed = False
        # Make a shortcut for the timestep
        self.steps = int(self.pars.endTime/self.pars.dt)
        #  Resolution and domain size
        self.dx = self.pars.Lx/self.pars.Nx
        self.dy = self.pars.Ly/self.pars.Ny
        # Initialize time variables
        self.t = 0.           # simulation timekeeping
        self.time_plot = 0.   # plot timekeeping
        self.time_series = 0. # stats timekeeping
        #  Create mesh
        self.mesh = PeriodicGrid2D(dx=self.dx,nx=self.pars.Nx,dy=self.dy,ny=self.pars.Ny)
        self.x = self.mesh.faceCenters()
        self.X = self.mesh.cellCenters()
        self.M = self.pars.M
        # set up figure axes
        self.Ffig.clf()
        # axes for time series
        self.ax = self.Ffig.add_subplot(212)
        # axes for the resource
        self.ax2s = [self.Ffig.add_subplot(2,self.M+1,1,projection='3d')]
        plt.title('r')
        for m in range(self.M):
            self.ax2s += [self.Ffig.add_subplot(2,self.M+1,m+2,projection='3d')]
            plt.title(f' c  {m}')
        #self.ax2 = self.Ffig.add_subplot(211,projection='3d')
        #  Create resource and consumer variables
        self.Res = Resource(mesh=self.mesh)  #init=self.pars.phyto5)
        self.Res.setpars(pars=self.pars,init=True)  #init=self.pars.phyto5)
        self.Cons = [Consumer(mesh=self.mesh,name=f'consumer{i}') for i in range(self.M)]
        for m,C in enumerate(self.Cons):
            C.color=self.colors[m]
            C.setpars(m=m,pars=self.pars,init=True)
        # Some cellvariables to use in calculating rates and statistics
        self.rrr = CellVariable(name="tempr", 
                           mesh=self.mesh,
                           value= 0., hasOld=1)
        self.sss = CellVariable(name="temps", 
                           mesh=self.mesh,
                           value= 0., hasOld=1)
        self.UconvCoeff = FaceVariable(mesh=self.mesh, rank=1)
        self.UconvCoeff[0,:] = 0
        self.UconvCoeff[1,:] = 0
        self.sourceCoeff = CellVariable(mesh=self.mesh, value=1.0)
        #  Lists for time series of averages
        self.mean_t = []
        self.meanZ = [[] for m in range(self.M)]
        self.meanR = []
        self.meanRR = []
        self.meanZZ = [[] for m in range(self.M)]
        self.meanRZ = [[] for m in range(self.M)]

        #self.tmp_output_file = self.pars.directory+'/'+'py3fsl.ser'
        #self.out_file = open(self.tmp_output_file,"w")
    
    def stats(self):
        """A method to streamline graphical output.
        """
        # set time for next stats
        self.time_series += self.pars.seriesInterval
        #  Collect statistics
        self.mean_t.append(self.t)
        R = self.Res.R  # a shortcut
        self.meanR.append(numerix.array(R.cellVolumeAverage()))
        self.rrr.setValue(R*R)
        self.meanRR.append(numerix.array(self.rrr.cellVolumeAverage()))
        for m in range(self.M):
            Z = self.Cons[m].Z  # a shortcut
            self.meanZ[m].append(numerix.array(Z.cellVolumeAverage()))
            # Calculate and record spatial correlations
            self.rrr.setValue(Z*Z)
            self.meanZZ[m].append(numerix.array(self.rrr.cellVolumeAverage()))
            self.rrr.setValue(Z*R)
            self.meanRZ[m].append(numerix.array(self.rrr.cellVolumeAverage()))
        # Output to the screen
        if self.verbose:
            print('time = ',self.t, ' R_avg = ',self.meanR[-1], ' RR_avg = ',self.meanRR[-1])
            for m in range(self.M):
                print(f'Cons {m}: Z_avg = {self.meanZ[m][-1]}, ZZ_avg = {self.meanZZ[m][-1]}, RZ_avg = {self.meanRZ[m][-1]}')
        #  Output to file
        #out_str = f'{self.t},{self.meanR[-1]},{self.meanZ[-1]},{self.meanRR[-1]},{self.meanZZ[-1]},{self.meanRZ[-1]}'
        #self.out_file.write(out_str)

    
    def plot(self):
        """A method to streamline graphical output.
        """
        # set time for next plots
        self.time_plot += self.pars.plotInterval
        #  Create plot viewers for spatial distributions and time series of averages
        self.ax.cla()
        self.lineR, = self.ax.plot(self.mean_t,self.meanR,color=self.Res.color)
        # a create buckets for consumer lines
        self.linesZ = [None for m in range(self.M)]
        self.linesRZ = [None for m in range(self.M)]
        for m in range(self.M):
            self.linesZ[m], = self.ax.plot(self.mean_t,self.meanZ[m],color=self.Cons[m].color,linestyle='-')
            self.linesRZ[m], = self.ax.plot(self.mean_t,self.meanRZ[m],color=self.Cons[m].color,linestyle=':')
        self.ax.axhline(y=1.,color='k')
        self.ax.set_xlim(self.mean_t[0],self.mean_t[-1])
        self.ax.set_ylim(bottom=0.)
        self.ax.set_ylabel('Spatial averages')
        self.ax.set_xlabel('Time (Z - blue; R - green; R*Z - red)')
        # Plot resource and consumer distributions
        for i,V in enumerate([self.Res.R]+[self.Cons[m].Z for m in range(self.M)]):
            #print(i)
            plt.subplot(2,self.M+1,i+1)
            plt.cla()
            #plt.pause(0.5)
            vax = plt.gca()
            vax.plot_surface(self.X[0].reshape([128,128]),self.X[1].reshape([128,128]),V.value.reshape([128,128]), cmap='viridis')
            #plt.pause(0.5)
            vax.set_xlabel('Position, x')
            vax.set_ylabel('Position, y')
            if i == 0:
                vax.set_title(f'Resource')
            else:
                vax.set_title(f'Consumer{i-0}')
            vax.set_xlim(0,self.pars.Lx)
            vax.set_ylim(0,self.pars.Ly)
            vax.set_zlim(0,8)
            #plt.pause(0.1)
        #self.ax2.set_ylim(0,8)
            
                             
        #self.ax2.relim()            # reset axes limits
        #self.ax2.autoscale_view()   # rescale axes
        #self.ax2.set_zlabel('Population')
        #self.ax2.set_xlabel('Position, x')
        #self.ax2.set_ylabel('Position, y')
        #self.ax2.legend((self.linexz,self.linexr,), ('Z','R',),loc='upper right')
        #ax2.legend((linexz,linexr,), ('Z','R',),0)
        #self.ax2.set_xlim(0,self.pars.Lx)
        #self.ax2.set_ylim(0,8)
        #
        self.Ffig.suptitle(f'file: {self.pars.filename}   Time: {self.t:.2f}   endTime: {self.pars.endTime:.2f}')
        #plt.pause(0.05)
        #self.Ffig.canvas.draw()                 # redraw the canvas

        # this sometimes fails if the plots are empty
        try:
            pass #self.Ffig.tight_layout()
        except:
            pass
        if self.plot_mode == 'term':
            plt.draw()
            plt.pause(0.25)
            #plt.pause(0.05)
            self.Ffig.canvas.draw()
            self.Ffig.canvas.flush_events()
        elif self.plot_mode == 'ipynb': # plot mode for Jupyter notebooks
            display(self.Ffig)
            clear_output(wait=True)

        
        
    def run(self):
        """Execute a simulation using current parameters.
        """
        # Begin run timekeeping
        runtime(init=True)
        # shortcuts to improve legibility
        phyto3 = self.Res.phyto3
        phyto5 = self.Res.phyto5
        R = self.Res.R   
        rrr = self.rrr
        self.stats()
        #  Main loop
        for step in range(self.steps):
            # Calculate resource generation rate term
            #self.sss.setValue((0.5*(1.-cos(2.*pi*((self.X[0]/self.pars.Lx-self.pars.x_src)*
            #                                      (self.X[1]/self.pars.Ly-self.pars.y_src)))))**self.pars.phyto4 )
            #self.sss.setValue((0.5*(1.-cos(2.*pi*(self.X/self.pars.Lx-self.pars.x_src))))**self.pars.phyto4 )
            self.sss.setValue(((0.5*(1.-cos(2.*pi*(self.X[0]/self.pars.Lx-self.pars.x_src)))) *
                               (0.5*(1.-cos(2.*pi*(self.X[1]/self.pars.Ly-self.pars.y_src)))))**self.pars.phyto4 )
            sss_avg = self.sss.cellVolumeAverage()
            self.sourceCoeff.setValue( (self.pars.phyto1/sss_avg)*self.sss )
            # set up elements of vars, eqs
            eqs = [[self.Cons[m].Z,None] for m in range(self.M)] +  [[R,None]]
            eqs_rhs = [0. for m in range(self.M)] +  [-phyto3*R+self.sourceCoeff]
            # Add resource diffusion term(s) for the requested numerical method
            match self.num_method:
                case 1:
                    eqs_rhs[-1] += ExplicitDiffusionTerm(coeff=self.pars.Damb)
                case 2:
                    eqs_rhs[-1] += ImplicitDiffusionTerm(coeff=self.pars.Damb)
                case 3:
                    eqs_rhs[-1] += 0.5 * (ExplicitDiffusionTerm(coeff=self.pars.Damb) +
                                          ImplicitDiffusionTerm(coeff=self.pars.Damb))
            # Add rates for each of the M consumer variants
            for m in range(self.M):
                # some shortcuts
                Fr = self.Cons[m].Fr
                Str = self.Cons[m].Str
                Le = self.Cons[m].Le
                alpha4 = self.Cons[m].alpha4
                sigma_hat = self.Cons[m].sigma_hat
                psi = self.Cons[m].psi
                Z = self.Cons[m].Z  
                #  Calculate taxis/advection terms
                self.UconvCoeff.setValue((Fr*alpha4/2.)*R.faceGrad())
                #self.UconvCoeff.setValue((Fr*alpha4/2.)*R.faceGrad()[0])
                self.rrr.setValue(R/(R+sigma_hat-1.) )
                # Add the resource consumption term for the mth consumer
                eqs_rhs[-1] += - Le*rrr*Z
                eqs_rhs[m] +=  -ExponentialConvectionTerm(coeff=self.UconvCoeff) + (Str*(rrr - 1./sigma_hat))*Z
                # Implement the consumer diffusion numerical method: 1 - explicit, 2 - implicit, 3 - Crank-Nicholson
                match self.num_method:
                    case 1:
                        eqs_rhs[m] += ExplicitDiffusionTerm(coeff=self.pars.Damb + Fr/(2.*(1.-psi)))
                    case 2:
                        eqs_rhs[m] += ImplicitDiffusionTerm(coeff=self.pars.Damb + Fr/(2.*(1.-psi)))
                    case 3:
                        eqs_rhs[m] += 0.5 * (ExplicitDiffusionTerm(coeff=self.pars.Damb + Fr/(2.*(1.-psi))) +
                                             ImplicitDiffusionTerm(coeff=self.pars.Damb + Fr/(2.*(1.-psi))))
            # fill out equations using the assembled righthand sides
            for m,eq in enumerate(eqs):
                eq[1] = TransientTerm() == eqs_rhs[m]
            # perform a single sweep to advance the solution, then update
            for var, eqn in eqs:
                var.updateOld()
            for var, eqn in eqs:
                eqn.sweep(var=var,dt=self.pars.dt)
            # Advance time and call plot/stats are requested
            self.t += self.pars.dt
            if self.t >= self.time_series:
                self.stats()
            if self.t >= self.time_plot:
                self.plot()
            if self.t >= self.pars.t_src_next/self.pars.phyto2:
                self.pars.load_source(verbose=False)
                #self.pars.load_source(verbose=True)
            #  Terminate run if abort flag is set
            if self.halt:
                break
        # wrap up files and status flags
        self.completed = True
        self.pars.src_file.close()  # Close the source file
        #self.out_file.close()  # Close the output file
        # Output run time statistics
        runtime()
        # insure final state is plotted
        self.plot()


