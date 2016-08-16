class Neuron:
    """
    The parent class for developing and simulating biophysical model neurons.

    By default, any instantiation of this class will contain hodgkin-huxley biophysical parameters,
    including fast-activating sodium channel, delayed-rectifying potassium channel, and corresponding m, h, and n gates.

    These parameters are contained in the HH differential equations:

        C dV/dt = - [Gna * m^3*h * (V-Ena)] - [Gk * n^4 * (V-Ek)] - [Gl * (V-El)] + Iext + noise <--- voltage kinetics

        dm/dt = alpha_m * (1-m) - beta_m * m   <--- sodium activation kinetics
        dh/dt = alpha_h * (1-h) - beta_h * h   <--- sodium inactivation kinetics
        dn/dt = alpha_n * (1-n) - beta_n * n   <--- potassium activation kinetics

    with parameters as follows:

        C = 1uF/cm^2
        Gk = 36 uS/cm^2  <--- maximally open
        Gna = 120 uS/cm^2    <--- maximally open
        Gl = 0.3 uS/cm^2   <--- maximally open
        Ena = 50 mV
        Ek = -77 mV
        El = -55 mV

        alpha_m = [0.1 * (25 + V)] / [1 - exp(-(25 + V)/10))]
        beta_m = 4 * exp(-V/18)

        alpha_h = 0.07 * exp(-V/20)
        beta_h = 1 / [exp((-v + 30)/10) + 1]

        alpha_n =
        beta_n =


    The model can accept an instance of the Stim class that can be used for simulation.
    To add new currents to the model, use the "add_channel()" method, which accepts a Channel class with appropriate conductances/equations.

    For each instantiation of the Neuron class, you must supply the following parameters:
        Tmax    =   the maximum time for simulation (in seconds)
        dt      =   the time step (due to the very fast kinetics of the m gate, must be less than or equal to 0.0001 sec to adequately sample the AP!)
        V0      =   the initial resting voltage (typically -65 to -70 mV)

    Writen by Jordan Sorokin, 8/4/2016
    """

    # imports
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint

    global np
    global sp
    global odeint
    global plt

    # initialize
    def __init__(self, T, Fs, V0):
        global np

        # set up common parameters
        self.Tmax = float(T)
        self.Fs = float(Fs)
        self.dt = 1.0/self.Fs
        self.timevec = np.arange(0,self.Tmax,self.dt)
        self.V0 = float(V0)
        self.C = 1.0
        self.noise = 0.0 # initialize noise as 0
        self.stimulus = 0
        self.params = dict()

        # add sodium, potassium, and leak channels using the "add_channel" method
        Leak_params = {
            'Erev'      :   -55.0,
            'G'         :   0.3
        }

        Na_params = {
            'G'         :   120.0, # was 2.0
            'Erev'      :   50.0,
            'mK'        :   3.0,
            'm0'        :   0.05,
            'hK'        :   1.0,
            'h0'        :   0.6,
            'Am_alpha'  :   0.1,
            'Bm_alpha'  :   40.0,
            'Cm_alpha'  :   10.0,
            'Am_beta'   :   4.0,
            'Bm_beta'   :   65.0,
            'Cm_beta'   :   18.0,
            'Ah_alpha'  :   0.07,
            'Bh_alpha'  :   65.0,
            'Ch_alpha'  :   20.0,
            'Ah_beta'   :   1.0,
            'Bh_beta'   :   35.0,
            'Ch_beta'   :   10.0,
            'mAlpha'    :   lambda V,A,B,C: A * (V + B) / (1 - np.exp(-(V + B) / C)),
            'mBeta'     :   lambda V,A,B,C: A * np.exp(-(V + B) / C),
            'hAlpha'    :   lambda V,A,B,C: A * np.exp(-(V + B) / C),
            'hBeta'     :   lambda V,A,B,C: A / (1 + np.exp(-(V + B) / C))
        }

        K_params = {
            'G'         :   36.0, # was 1.44,
            'Erev'      :   -77.0,
            'mK'        :   4.0,
            'm0'        :   0.3,
            'Am_alpha'  :   0.01,
            'Bm_alpha'  :   55.0,
            'Cm_alpha'  :   10.0,
            'Am_beta'   :   0.125,
            'Bm_beta'   :   65.0,
            'Cm_beta'   :   80.0,
            'mAlpha'    :   lambda V,A,B,C: A * (V + B) / (1 - np.exp(-(V + B) / C)),
            'mBeta'     :   lambda V,A,B,C: A * np.exp(-(V + B) / C)
        }

        # add the sodum, potassium, and common parameters to "self.params" dictionary
        self.add_channel('Leak_params',Leak_params)
        self.add_channel('Na_params',Na_params)
        self.add_channel('K_params',K_params)


    def add_channel(self,chan_name,chan_params):
        """
        Add a channel with properties specified in "chan_params"
        Look at the parameters in self.params as a reference for how to structure the "chan_params" and "chan_name"
        """
        self.params[chan_name] = chan_params


    def remove_channel(self,chan_name):
        """
        Remove a specified channel from "self.params"
        """
        if self.params.has_key(chan_name):
            self.params.pop(chan_name)


    def add_stim(self,Stim):
        """
        Accept an instance of the Stim class which will be used for simulation.
        The values Stim.I and Stim.Itot will be re-populated if self.timevec does not match self.Stim.timevec
        """

        global np

        self.stimulus = Stim

        # loop and re-establish the stimulus protocol
        if not self.Tmax == self.stimulus.Tmax or not self.Fs == self.stimulus.Fs:
            self.stimulus.timevec = np.arange(0,self.Tmax,self.dt)

            for j in xrange(len(self.stimulus.I)):
                current = np.zeros( (len(self.stimulus.timevec),self.stimulus.nstep[j]) ) + self.stimulus.I0 # T x nstep matrix of zeros + constant current I0, to be updated in the loop

                # loop through the number of steps of this added current, and update each current trace
                for i in xrange(self.stimulus.nstep[j]):
                    val = self.stimulus.Istep[j][i] # pull out the i-th current step and update the i-th current trace
                    pulse = (self.stimulus.timevec >= self.stimulus.ton[j]) & (self.stimulus.timevec <= self.stimulus.toff[j]) # get logical array indicating where the current step occurs
                    current[pulse,i] = val # populate the i-th current pulse with the value of the i-th current step

                self.stimulus.I['I'+str(j+1)] = current # add this current step(s) to the dictionary of individual currents

            self.stimulus.Itot = sum(self.I.values()) # now sum (i.e. broadcasting) all of the individual currents in "self.I" into a total current protocol "self.Itot"


    def add_noise(self,sigma):
        """
        Add zero-mean gaussian white noise with variance sigma.
        """
        self.noise = sigma # converted to mV


    def Iinj(self,t):
        """
        Pull out the injected current value from self.stimulus.Itot
        If self.stimulus == 0, the set an arbitrary stimulus protocol
        """
        global np

        if not self.stimulus == 0:
            Iext = 0
            I0 = self.stimulus.I0
            for j in xrange(len(self.stimulus.ton)):
                ton = self.stimulus.ton[j]
                toff = self.stimulus.toff[j]
                Ion = self.stimulus.Istart[j]
                Ioff = self.stimulus.Iend[j]

                Iext += Ion*(t>ton) - Ion*(t>toff) + Ioff*(t>toff)

            #Iext = Ion*(t>ton) - Ion*(t>toff) + Ioff*(t>toff) + I0
            Iext += I0
        else:
            Iext = 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)

        if not self.noise == 0:
            Iext += self.noise * (t>0) * np.random.standard_normal()

        return Iext


    def dGate(self,alpha,beta,g):
        """
        Return the new gating value
        """
        return (alpha * (1.0-g)) - (beta * g)


    def dCurrent(self,V,G,E,m,h,k,l):
        """
        calculate the current from a channel give current voltage
        """
        return G * (m**k) * (h**l) * (V-E)


    # ode function for extracting dV/dt, and dk/dt gating variables
    #===========================================================
    @staticmethod
    def derivate(X,t,self):
        """
        For each k-th gating variable, calculate dk/dt
        Then find dV/dt by summing the currents from each channel, the external current, and any noise terms
        """
        global np

        V = X[0] # pull out the voltage
        Xnew = list(np.zeros(len(X)))
        Ich = []
        P = self.params.keys();
        varkey = self.Results['Names']

        # check for stim class
        Iext = self.Iinj(t)

        # now loop through the available channels and get the dk/dt and Ik for each
        for i in xrange(len(P)):
            # initialize m,h,l,k to = 1, which will be overriden if these gate variables exist in this channel i
            m = 1; h = 1; l = 1; k = 1;
            dm = 1; dh = 1;

            # extract the channel i using the name from P
            Ch = self.params[P[i]]
            allgates = [(ind,x) for ind,x in enumerate(varkey) if P[i] in x] # pull out the names and indices of the gating variables for channel i

            # loop over any gate variables that exist in this channel, update the new gating values
            if any(allgates):
                for j in xrange(len(allgates)):
                    gate_name = allgates[j][1].split('-')[1]
                    gate_ind = allgates[j][0]

                    if gate_name == 'm':
                        alpha = Ch['mAlpha'](V,Ch['Am_alpha'],Ch['Bm_alpha'],Ch['Cm_alpha'])
                        beta = Ch['mBeta'](V,Ch['Am_beta'],Ch['Bm_beta'],Ch['Cm_beta'])

                        # now compute the derivative dk/dt for the k-th activating gate
                        m = X[gate_ind] # extract old activation gate value
                        dm = self.dGate(alpha,beta,m) # calculate new activation gate value
                        k = Ch['mK']
                        Xnew[gate_ind] = dm # store the new activation gate value

                    if gate_name == 'h':
                        alpha = Ch['hAlpha'](V,Ch['Ah_alpha'],Ch['Bh_alpha'],Ch['Ch_alpha'])
                        beta = Ch['hBeta'](V,Ch['Ah_beta'],Ch['Bh_beta'],Ch['Ch_beta'])

                        # append this inactivation gate
                        h = X[gate_ind] # take old inactivation gate state
                        dh = self.dGate(alpha,beta,h) # find the new inactivation value
                        l = Ch['hK']
                        Xnew[gate_ind] = dh # store the new value

            # now calculate the current for this channel
            dI = self.dCurrent(V, Ch['G'], Ch['Erev'], m, h, k, l)
            Ich.append(dI) # find the current for this channel using the old activation/inactivation values

        # finally, calculate the new dV/dt
        dV = (Iext - sum(Ich)) / self.C
        Xnew[0] = dV

        return Xnew
    #===========================================================

    def simulate(self):
        """
        Main runtime for simulating V(t) given current channels and stimulus.
        Results of the simulation are stored in self.Results, which is a dictionary containing:
            'Values'    :   time series of V(t), and gating variables k(t) for k = 1:K gates
            'Names'     :   key for the 'Values, which contains a list of names for each time series
        """
        # define globals
        global odeint

        # Pull out the channels in the model, and create the initialization vector for the ode solver
        P = self.params.keys() # will be an ordered list of names!
        init = [self.V0]
        varkey = ['V']

        # loop for the number of channels in P
        for j in xrange(len(P)):
            Ch = self.params[P[j]]
            if Ch.has_key('mK'):
                init.append(Ch['m0'])
                varkey.append(P[j]+'-m')
            if Ch.has_key('hK'):
                init.append(Ch['h0'])
                varkey.append(P[j]+'-h')

        self.Results = {
            'Names' : varkey
        }

        # ====================================
        print('Running simulation . . .')
        X = odeint(self.derivate, init, self.timevec, args=(self,))
        # ====================================

        # store the simulation results into self.Results dictionary
        self.Results['Values'] = X


    def plot_results(self):
        """
        Plots the results from running "self.simulate()" in new figures.
        """
        global plt

        R = self.Results['Values']
        N = self.Results['Names']

        numVals = len(N)

        plt.figure()
        for j in xrange(numVals):
            plt.subplot(numVals,1,j+1)
            plt.plot(self.timevec,R[:,j],'k')
            plt.title(N[j])

        plt.show()
