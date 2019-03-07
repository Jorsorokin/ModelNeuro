class Neuron():
    """
    The parent class for developing and simulating biophysical model point neurons.

    By default, any instantiation of this class will contain hodgkin-huxley biophysical parameters,
    e.g. fast-activating sodium channel, delayed-rectifying potassium channel, leak, and corresponding m, h, and n gates.

    These parameters are contained in the HH differential equations:

        C dV/dt = - [Gna * m^3*h * (V-Ena)] - [Gk * n^4 * (V-Ek)] - [Gl * (V-El)] + I(t) + e(t)
            I(t)    =   external input
            e(t)    =   gaussian white noise
            Gion(t) =   conductance
            Eion    =   reversal potential
            V(t)    =   neuron voltage
            C       =   neuron capacitance

        dm/dt = alpha_m * (1-m) - beta_m * m   <--- sodium activation kinetics
        dh/dt = alpha_h * (1-h) - beta_h * h   <--- sodium inactivation kinetics
        dn/dt = alpha_n * (1-n) - beta_n * n   <--- potassium activation kinetics

    with parameters as follows:

        C   =   1uF/cm^2
        Gk  =   36 mS/cm^2    <--- maximally open
        Gna =   120 mS/cm^2   <--- maximally open
        Gl  =   0.3 mS/cm^2   <--- maximally open
        Ena =   50 mV
        Ek  =   -80 mV
        El  =   -60 mV

        alpha_m = [0.3 * (25 + V)] / [1 - exp((-V - 25) / 10))]
        beta_m = 4 * exp(-(V + 65)/18)

        alpha_h = 0.07 * exp(-V/20)
        beta_h = 1 / [exp((-V - 30)/10) + 1]

        alpha_n = [0.01 * (V + 55)] / [1 - exp((-V - 55) / 10)]
        beta_n = 0.125 * exp((-V - 65) / 80)

    * For details/explanations of these values, see Pospischil et al. 2008, Biol. Cybernetics *

    The model accepts an instance of the Stim class that can be used for simulation.
    You can easily add or remove stimuli directly through the neuron instance.

    To add new ionic currents to the model, use the "add_channel()" method. The channel
    parameters should be set up in a dictionary with the same parameter/function names and
    structure as those for the pre-defined Na, K, and Leak parameters.
        * See the __init__ method for details on how to construct these parameters *

    For each instantiation of the Neuron class, you must supply the following parameters:
        T       =   the maximum time for simulation (in miliseconds)
        Fs      =   the sampling rate (in samples / milisecond...values of 50-100 work well)
        V0      =   the initial resting voltage (typically -65 to -70 mV)

    To simulate the current model, use the "simulate" method, which will loop over all stim
    protocols contained in the model and output the Voltage and m/h gates for each channel for each stim.
    These will be stored in the "Results" attribute. You can then view the results with the "plot_results" method.

    Writen by Jordan Sorokin, 8/4/2016
    """

    # imports
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    from numba import jit

    global np
    global sp
    global plt

    # initialize
    def __init__(self, T, Fs, V0):
        global np, jit

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
        Leak = {
            'Erev'      :   -60.0,
            'G'         :   0.3
        }

        Na = {
            'G'         :   120.0, # was 2.0
            'Erev'      :   50.0,
            'mK'        :   3.0,
            'm0'        :   0.05,
            'hK'        :   1.0,
            'h0'        :   0.6,
            'Am_alpha'  :   0.3,
            'Bm_alpha'  :   25.0,
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
            'hBeta'     :   lambda V,A,B,C: A / (1 + np.exp(-(V + B) / C)),
            'deltaG'    :   lambda alpha,beta,g: (alpha * (1.0 - g)) - (beta * g)
        }

        K = {
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
            'mBeta'     :   lambda V,A,B,C: A * np.exp(-(V + B) / C),
            'deltaG'    :   lambda alpha,beta,g: (alpha * (1.0 - g)) - (beta * g)
        }

        # add the sodum, potassium, and common parameters to "self.params" dictionary
        self.add_channel('Leak_params',Leak)
        self.add_channel('Na_params',Na)
        self.add_channel('K_params',K)


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
        self.noise = sigma


    @jit(cache=True)
    def Iinj(self,t,trial):
        """
        Pull out the injected current value from self.stimulus.Itot
        If self.stimulus == 0, the set an arbitrary stimulus protocol
        """
        global np

        if not self.stimulus == 0:
            Itot = self.stimulus.Itot[:,trial]
            Iext = Itot[t] + self.stimulus.I0
        else:
            #Iext = 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)
            Iext = 0

        if not self.noise == 0:
            Iext += self.noise * np.random.standard_normal() # add gaussian noise weighted by the noise level in "self.noise"

        return Iext


    # ode function for extracting dV/dt, and dk/dt gating variables
    #===========================================================
    def _derivate(self,X,trial):
        """
        For each k-th gating variable, calculate dk/dt
        Then find dV/dt by summing the currents from each channel, the external current, and any noise terms
        """
        global np

        # pull out the relevant variables for looping
        params = self.params
        P = params.keys()
        varkey = self.Results['Names']

        # initialize vectors/matrices for storage
        nchan = len(P)
        nvars = len(X)
        ntime = len(self.timevec)
        C = self.C
        dt = self.dt

        R = np.zeros((ntime,nvars))
        Ich = [0]*nchan
        
        #==========================
        R = self._derivateloop(R,X,P,params,varkey,ntime,nchan,nvars,C,dt,Ich,trial)
        
        return R
        #==========================

    #@jit
    def _derivateloop(self,R,X,P,Params,varkey,ntime,nchan,nvars,C,dt,Ich,trial):
        """ Grunt-work of the _derivate method belongs here. It performs the tripple-nested 
        for-loop for updating voltages given the equations specified in the main Neuron-class 
        comments. Putting it in its own function allows for numba-optimized JIT
        """

        # LOOP !
        for tind in xrange(ntime):
            V = X[0] # pull out the voltage
            Iext = self.Iinj(tind,trial) # get the external voltage command

            # now loop through the available channels and get the dk/dt and Ik for each
            for i in xrange(nchan):
                # initialize m,h,l,k to = 1, which will be overriden if these gate variables exist in this channel i
                m = 1; h = 1; l = 1; k = 1;
                dm = 1; dh = 1;

                # extract the channel i using the name from P
                Ch = Params[P[i]]
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
                            k = Ch['mK']
                            X[gate_ind] = m + Ch['deltaG'](alpha,beta,m) * dt # store the new activation gate value

                        if gate_name == 'h':
                            alpha = Ch['hAlpha'](V,Ch['Ah_alpha'],Ch['Bh_alpha'],Ch['Ch_alpha'])
                            beta = Ch['hBeta'](V,Ch['Ah_beta'],Ch['Bh_beta'],Ch['Ch_beta'])

                            # append this inactivation gate
                            h = X[gate_ind] # take old inactivation gate state
                            l = Ch['hK']
                            X[gate_ind] = h + Ch['deltaG'](alpha,beta,h) * dt # store the new value

                # now calculate the current for this channel
                Ich[i] = Ch['G'] * (m**k) * (h**l) * (V-Ch['Erev'])

            # finally, calculate the new dV/dt
            X[0] = V + ((Iext - sum(Ich)) / C * dt)
            R[tind,:] = X

        return R
    #===========================================================
    

    @jit
    def simulate(self):
        """
        Main runtime for simulating V(t) given current channels and stimulus.
        Results of the simulation are stored in self.Results, which is a dictionary containing:
            'Values'    :   time series of V(t), and gating variables k(t) for k = 1:K gates
            'Names'     :   key for the 'Values, which contains a list of names for each time series
        """

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

        self.Results = {'Names' : varkey}

        # ====================================
        print('Running simulation . . .')
        trials = self.stimulus.ntrials
        if trials > 1:
            self.Results['Values'] = np.zeros((len(self.timevec),len(init),trials))
            # loop over number of trials, store into 3rd dimension of "Values" for each
            for s in xrange(trials):
                R = self._derivate(init,s)
                self.Results['Values'][:,:,s] = R
        else:
            R = self._derivate(init,0)
            self.Results['Values'] = R
        # ====================================


    @jit
    def plot_results(self):
        """
        Plots the results from running "self.simulate()" in new figures.
        """
        global plt

        if not any(self.Results):
            print('No simulation results found...run "self.simulate()" first!')
            return

        numVals = len(self.Results['Names'])
        trials = self.stimulus.ntrials;

        # plot the results of all gates, voltage
        plt.figure()
        for j in xrange(numVals):
            plt.subplot(numVals,1,j+1)
            plt.plot(self.timevec,self.Results['Values'][:,j],'k') # plot the time series
            plt.title(self.Results['Names'][j])
            if not j==numVals:
                plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off') # turn of x-axis ticks unless last plot
        plt.show()


    def print_model(self):
        """
        Print the equations and channels currently used in the model
        """
        P = self.params
        names = P.keys()
        current = str(self.C) + ' * dV/dT = '
        for i in xrange(len(names)):
            x = names[i]
            ch = x.split('_')[0]
            Erev = str(P[x]['Erev'])
            if P[x].has_key('m0'):
                y = 'm' + '^' + str(int(P[x]['mK']))
            if P[x].has_key('h0'):
                y += 'h' + '^' + str(int(P[x]['hK']))

            current += ('[g' + ch + ' * ' + y + ' * (V(t) - ' + Erev + ')] + ')

        if not self.stimulus == 0:
            current += 'I(t)'

        print('Current model:')
        print(current)
