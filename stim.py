class Stim:
    """
    A class for adding current injections into model neurons or networks.
    Takes in a maximum time (T), sampling rate (Fs), and steady-state current (I0) as initial parameters.

    Once initialized, it is easy to add varying currents at different times using
    "Stim.addcurrent()", which takes in a start time (t_on), end time (t_off), start value (I_start),
    and optionally an ending value (I_end) plus the number of steps to get there (I_step).

    To visualize the stimulus protocol, you can use "Stim.plotcurrent()", which plots all traces in the current protocol
    on top of one another in a new figure.

    By default, the current values are in Picoamps (pA). Thus, no need to input small values (1e-12), as the
    class will convert your values into picoamps.

    Note that the class assumes numpy, scipy, and matplotlib are all installed on your machine and available in your python path!

    Writen by Jordan Sorokin, 8/4/2016
    """

    # imports
    import numpy as np
    import matplotlib.pyplot as plt
    global np
    global plt

    def __init__(self, T, Fs, I0):
        """
        Initialize the current protocol with a max time and sampling rate.
        Additionally, set the number of trials = 1, which will change if we include stepped currents
        in the protocol to match the # of steps.
        """
        global np

        self.Tmax = float(T)
        self.Fs = float(Fs)
        self.timevec = np.arange(0,self.Tmax,1/self.Fs) # time vector, discretized by 1/Fs steps
        self.I0 = float(I0) # converted to pico amps
        self.ntrials = 1

        # now initialize the stimulus parameters that will be updated with "addcurrent()"
        self.ton = []
        self.toff = []
        self.Istart = []
        self.Iend = []
        self.nstep = []
        self.Istep = []
        self.I = dict() # will be updated with added currents


    def add_current(self, t_on, t_off, I_start, I_end=0, nstep=1):
        """
        Add a current protocol with starting time t_on, ending time t_off, picoamp value I_start,
        ending value I_end, and number of steps to reach I_end by num_step
        """

        global np

        self.ton.append(t_on)
        self.toff.append(t_off)
        self.Istart.append(I_start)
        self.Iend.append(I_end)
        self.nstep.append(nstep)
        self.ntrials = max(self.nstep) # update the current number of trials

        # cmake a series of currents for the number of steps. If nstep =1, the list will hold a single value
        self.Istep.append(list(np.linspace(I_start,I_end,nstep))); # append a list of current steps

        # update the running current protocol separately for this added current
        index = len(self.I)
        current = np.zeros( (len(self.timevec),nstep) ) + self.I0 # T x nstep matrix of zeros + constant current I0, to be updated in the loop

        # loop through the number of steps of this added current, and update each current trace
        for i in xrange(self.nstep[-1]):
            val = self.Istep[-1][i] # pull out the i-th current step and update the i-th current trace
            pulsetime = (self.timevec >= t_on) & (self.timevec <= t_off) # get logical array indicating where the current step occurs
            current[pulsetime,i] = self.Istep[-1][i] # populate the i-th current pulse with the value of the i-th current step

        self.I['I'+str(index+1)] = current # add this current step(s) to the dictionary of individual currents
        self.Itot = sum(self.I.values()) # now sum (i.e. broadcasting) all of the individual currents in "self.I" into a total current protocol "self.Itot"


    def remove_current(self,index):
        '''
        remove specific current from the total protocol using the "index" of that current
        '''
        k = self.I.keys()
        k.sort() # sort the I1 --> IN by name
        self.I.pop(k[index],0)
        self.ton.pop(index)
        self.toff.pop(index)
        self.Istart.pop(index)
        self.Iend.pop(index)
        self.nstep.pop(index)
        self.Istep.pop(index)

        # now re-calculate Itot and ntrials
        if len(self.nstep) > 0:
            self.ntrials = max(self.nstep)
            self.Itot = sum(self.I.values())


    def plot_protocol(self):
        """
        plot the current stim protocol in a new figure
        """
        global plt

        # initialize matrix of I0 for storing currents
        plt.figure()
        plt.plot(self.timevec,self.Itot,'k')
        plt.show()
