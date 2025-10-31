"""Classes and modules to faciliate Frost-Strathmann-Lessard (FSL) simulations.
"""
from time import process_time, time, sleep
global process_time0, wall_time0


def runtime(init=False):
    """Function for timing runs
    """
    global process_time0, wall_time0
    # Set up timing of run
    if init:  # Record start time
        process_time0 = process_time()
        wall_time0 = time()
    else:  # Print run times
        print(f'Simulation process time = {process_time()-process_time0} seconds')
        print(f'Simulation wall time = {time()-wall_time0} seconds')


class FSLpars():
    """A class to facilitate handling parameters for FSL simulations.
    """
    def __init__(self,input_file=None):
        self.input_file = input_file
        if input_file is not None:
            self.load()

    def load(self,input_file=None):
        """A method to faciliate parsing of input parameter files. Input files are compatible between
           1D and 2D FSL simulations, so all include parameters for 2D simulations. These are ignored
           in 1D simulations.
        """
        
        if input_file is not None:
            self.input_file = input_file
        # Extract parameters from input file:
        self.inp_file = open(self.input_file,"r")
        #  filename
        line1 = self.inp_file.readline()
        splt_line1 = line1.split()
        self.filename = splt_line1[0]
        print("Filename = ",self.filename) 
        
        #if not os.path.exists(directory):
        #    os.mkdir(directory)
        #output_file = directory+'/'+filename+'.ser'
        #print("Initializing output file ",output_file)

        #if os.path.exists(output_file):
        #    print("*********************************")
        #    print("*********************************")
        #    print("*********************************")
        #    print("Found existing file: "+output_file)
        #    print("Skipping run")
        #    print("*********************************")
        #    print("*********************************")
        #    print("*********************************")
        #    exit()

        #  directory
        line2 = self.inp_file.readline()
        splt_line2 = line2.split()
        self.directory = splt_line2[0]
        self.directory = '.'
        print("Directory = ",self.directory)
        
        #  source file
        line3 = self.inp_file.readline()
        splt_line3 = line3.split()
        self.source_file = splt_line3[0]
        print("Source file = ",self.source_file)
        
        self.src_file = open(self.source_file,"r")
        self.load_source(init=True,verbose=True)
        #src_line = src_file.readline()
        #splt_src = src_line.split()
        #t_src = float( splt_src[0] )
        #x_src = float( splt_src[1] )
        #y_src = float( splt_src[2] )

        #src_line = src_file.readline()
        #splt_src = src_line.split()
        #t_src_next = float( splt_src[0] )
        #x_src_next = float( splt_src[1] )
        #y_src_next = float( splt_src[2] )

        #  simulation parameters
        line5 = self.inp_file.readline()
        line5 = self.inp_file.readline()
        splt_line5 = line5.split(',')
        self.M = float( splt_line5[0] )
        self.Nx = int( splt_line5[1] )
        self.Ny = int( splt_line5[2] )
        self.Lx = float( splt_line5[3] )
        self.Ly = float( splt_line5[4] )
        print("M,Nx,Lx = ",self.M,self.Nx,self.Lx)
        #print("M,Nx,Ny,Lx,Ly = ",self.M,self.Nx,self.Ny,self.Lx,self.Ly)

        #  time parameters
        line6 = self.inp_file.readline()
        line6 = self.inp_file.readline()
        splt_line6 = line6.split(',')
        self.endTime = float( splt_line6[0] )
        self.plotInterval = float( splt_line6[1] )
        self.seriesInterval = float( splt_line6[2] )
        self.dt = float( splt_line6[3] )
        print("endTime, plotInterval, seriesInterval, dt = ",
              self.endTime,self.plotInterval,self.seriesInterval,self.dt)

        #  consumer parameters
        line7 = self.inp_file.readline()
        line7 = self.inp_file.readline()
        splt_line7 = line7.split(',')
        self.Fr = float( splt_line7[0] )
        self.alpha1 = float( splt_line7[1] )
        self.alpha4 = float( splt_line7[2] )
        self.Le = float( splt_line7[3] )
        self.Str = float( splt_line7[4] )
        self.sigma_hat = float( splt_line7[5] )
        self.N = float( splt_line7[6] )
        self.psi = float( splt_line7[7] )
        print("Fr,alpha1,alpha4,Le,Str,sigma_hat,N,psi = ",
              self.Fr,self.alpha1,self.alpha4,self.Le,self.Str,self.sigma_hat,self.N,self.psi)

        #  resource parameters
        line8 = self.inp_file.readline()
        line8 = self.inp_file.readline()
        splt_line8 = line8.split(',')
        self.phyto1 = float( splt_line8[0] )
        self.phyto2 = float( splt_line8[1] )
        self.phyto3 = float( splt_line8[2] )
        self.phyto4 = float( splt_line8[3] )
        self.phyto5 = float( splt_line8[4] )
        self.Dphyto = float( splt_line8[5] )
        self.rho_plot_flag = float( splt_line8[6] )
        print("phyto1,phyto2,phyto3,phyto4,phyto5,Dphyto,rho_plot_flag = ",
              self.phyto1,self.phyto2,self.phyto3,self.phyto4,self.phyto5,self.Dphyto,self.rho_plot_flag)

        self.inp_file.close  # Close the input file


    def load_source(self,init=False,verbose=False):
        """Load the next time and position of resource sources. If init==True, then sources
           for both the current (initial) time and the next time are loaded. Otherwise,
           the previously "next" source is made copied as the current source, and a new
           next source is loaded.
        """
        if init: # On initialization, load the source for the current time
            src_line = self.src_file.readline()
            splt_src = src_line.split()
            self.t_src = float( splt_src[0] )
            self.x_src = float( splt_src[1] )
            self.y_src = float( splt_src[2] )
        else:   # Subsequently, copy the "next" source to be current
            self.t_src = self.t_src_next
            self.x_src = self.x_src_next
            self.y_src = self.y_src_next
        # Load the next source time and location
        src_line = self.src_file.readline()
        splt_src = src_line.split()
        self.t_src_next = float( splt_src[0] )
        self.x_src_next = float( splt_src[1] )
        self.y_src_next = float( splt_src[2] )
        if verbose:
            print("New source at ",self.t_src,self.x_src)
        
