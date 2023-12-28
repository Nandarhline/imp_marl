""" Defines the offshore wind farm (owf) class."""

import numpy as np
import scipy.stats as stats
import os
import numpy as np
from imp_env.imp_env import ImpEnv

class Owf_twin(ImpEnv):

    def __init__(self, config=None):
        """ offshore wind farm (owf) class. 

    Attributes:
        n_owt: Integer indicating the number of wind turbines.
        lev: Integer indicating the number of components considered in each wind turbine.
        discount_reward: Float indicating the discount factor.
        campaign_cost: Boolean indicating whether a global campaign cost is considered in the reward model.
        n_comp: Integer indicating the number of components.
        ep_length: Integer indicating the number of time steps in the finite horizon.
        proba_size: Integer indicating the number of bins considered in the discretisation of the damage probability.
        n_obs_inspection: Integer indicating the number of potential outcomes resulting from an inspection.
        actions_per_agent: Integer indicating the number of actions that an agent can take.
        initial_damage_proba: Numpy array containing the initial damage probability.
        transition_model: Numpy array containing the transition model that drives the environment dynamics.
        inspection_model: Numpy array containing the inspection model.
        agent_list: Dictionary categorising the number of agents.
        time_step: Integer indicating the current time step.
        damage_proba: Numpy array contatining the current damage probability.
        d_rate: Numpy array contatining the current deterioration rate.
        observations: Dictionary listing the observations received by the agents in the Dec-POMDP.

    Methods: 
        reset
        step
        pf_sys
        immediate_cost
        belief_update_uncorrelated
    """
        if config is None:
            config = {"n_owt": 2,
                      "comps": [2, 1, 4],
                      "discount_reward": 1,
                      "campaign_cost": False,
                      "virtual_sensor": True}
        assert "n_owt" in config and \
               "comps" in config and \
               "discount_reward" in config and \
               "campaign_cost" in config and \
               "virtual_sensor" in config, \
            "Missing env config"

        self.n_owt = config["n_owt"]  
        self.n_awcomp = config["comps"][0]
        self.n_bwcomp = config["comps"][1] 
        self.n_mdcomp = config["comps"][2] 
        self.lev = self.n_awcomp + self.n_bwcomp + self.n_mdcomp
        self.discount_reward = config["discount_reward"]
        self.campaign_cost = config["campaign_cost"]
        self.virutal_sensor = config["virtual_sensor"]
        self.n_comp = self.n_owt*self.lev
        self.n_agents = self.n_owt*(self.lev-self.n_mdcomp)
        self.ep_length = 20 

        # Loading the underlying transition and inspection models
        if self.virutal_sensor is True:
            drmodel = np.load(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'pomdp_models/Owf203020_WithVM.npz'))
        else:
            drmodel = np.load(os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'pomdp_models/Owf203020_WithoutVM.npz'))

        self.d_interv = drmodel['d_interv']
        self.q_interv = drmodel['q_interv'] 
        self.proba_size_d = len(self.d_interv[0])-1 
        self.proba_size_q = len(self.q_interv[0])-1 
        self.proba_size = self.proba_size_d * self.proba_size_q

        self.n_obs_inspection = 2*self.proba_size_q
        self.actions_per_agent = 6


        # To build oservation model of digital twin 
        self.q_ref = self.q_interv[:-1,:].copy()
        self.q_ref[:,0] = -1e100
        self.q_ref[:,-1] = 1e100
        

        # (n_owt, 3 levels, nstcomp cracks)
        self.initial_damage_proba = np.zeros((self.n_owt, self.lev, self.proba_size))
        self.initial_damage_proba[:,0:self.n_awcomp,:] = drmodel['belief0'][0,:]
        self.initial_damage_proba[:,self.n_awcomp:self.n_awcomp+self.n_bwcomp,:] = drmodel['belief0'][1,:]
        self.initial_damage_proba[:,-self.n_mdcomp:,:] = drmodel['belief0'][2,:]
        self.initial_twin_state = np.tile(drmodel['belief0_twin'],(self.n_owt, self.lev-self.n_mdcomp, 1))          
        self.initial_eps = np.tile(drmodel['belief0_eps'],(self.n_owt, self.lev-self.n_mdcomp, 1))

        # Transition models
        self.T0 = drmodel['T0'] 
        self.Tr = drmodel['Tr']
        self.T0_twin = drmodel['T0_twin']
        self.Tr_twin = drmodel['Tr_twin']
        self.Ts_twin = drmodel['Ts_twin']

        # Observation models for inspection and physical sensor   
        self.O_ins = drmodel['O_ins']
        self.O_monitor = drmodel['O_monitor']
        self.O_ins_monitor = drmodel['O_ins_monitor']

        self.agent_list = ["agent_" + str(i) for i in range(self.n_agents)]

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.twin_state = self.initial_twin_state
        self.epsilon_proba = self.initial_eps
        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        self.observations = None 

        self.reset()

    def reset(self):
        """ Resets the environment to its initial step.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
        """        
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self.time_step = 0
        self.damage_proba = self.initial_damage_proba
        self.twin_state = self.initial_twin_state
        self.epsilon_proba = self.initial_eps
        self.d_rate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        damage_proba_comp = np.reshape(self.damage_proba[:, :-self.n_mdcomp, :], (self.n_agents, -1))
        # epsilon_proba_comp = np.reshape(self.epsilon_proba[:, :, :], (self.n_agents, -1))
        self.observations = {}

        for i in range(self.n_agents): # Shall we also add pf_sys here?
            dq_proba = np.reshape(damage_proba_comp[i,:], [self.proba_size_d, self.proba_size_q])
            d_proba = np.sum(dq_proba, axis=1)
            q_proba = np.sum(dq_proba, axis=0)
            self.observations[self.agent_list[i]] = np.concatenate(
                (d_proba, q_proba, [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        """ Transitions the environment by one time step based on the selected actions. 

        Args:
            action: Dictionary containing the actions assigned by each agent.

        Returns:
            observations: Dictionary with the damage probability received by the agents.
            rewards: Dictionary with the rewards received by the agents.
            done: Boolean indicating whether the final time step in the horizon has been reached.
        """
        action_list = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            action_list[i] = action[self.agent_list[i]]

        observation_, next_damage_proba, next_twin_state, next_epsilon_proba, next_drate = \
            self.belief_update_uncorrelated(self.damage_proba, self.twin_state, self.epsilon_proba,
                                            action_list, self.d_rate)

        reward_ = self.immediate_cost(self.damage_proba, action_list, next_damage_proba,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()

        rewards = {}
        for i in range(self.n_agents):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1
        damage_proba_comp = np.reshape(next_damage_proba[:,:-self.n_mdcomp,:], (self.n_agents, -1))
        # epsilon_proba_comp = np.reshape(self.epsilon_proba[:, :, :], (self.n_agents, -1))

        self.observations = {} 
        for i in range(self.n_agents):
            dq_proba = np.reshape(damage_proba_comp[i,:], [self.proba_size_d, self.proba_size_q])
            d_proba = np.sum(dq_proba, axis=1)
            q_proba = np.sum(dq_proba, axis=0)
            self.observations[self.agent_list[i]] = np.concatenate(
                (d_proba, q_proba, [self.time_step / self.ep_length]))

        self.damage_proba = next_damage_proba
        self.twin_state = next_twin_state
        self.epsilon_proba = next_epsilon_proba
        self.d_rate = next_drate

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        return self.observations, rewards, done, observation_

    def pf_sys1(self, pf): 
        """ Computes the system failure probability as the sum of the failure risk of all wind tubines.
            Each wind turbine fails if any component fails.
        
        Args:
            pf: Numpy array with components' failure probability.
        
        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        pfSys = np.zeros(self.n_owt)
        surv = 1 - pf.copy()
        #failsys = np.zeros((nwtb,2))
        for i in range(self.n_owt):
            survC = np.prod(surv[i,:])
            pfSys[i] = 1 - survC
        return pfSys
    
    def pf_sys(self, pf):
        n = 2 # 1-out-of-2 system (number of components per level)
        n_pairedlev = int(self.lev/n)
        k = 1
        nk = n - k
        m = k + 1
        pfSys = np.zeros(self.n_owt)
        for wt in range(self.n_owt):
            pfSubsys = np.zeros(n_pairedlev)
            # First compute pf of each level
            for l in range(n_pairedlev):
                pf_lev = pf[wt,l*n:(l+1)*n]
                A = np.zeros(m + 1)
                A[1] = 1
                L = 1
                for j in range(1, n + 1):
                    h = j + 1
                    Rel = 1 - pf_lev[j - 1]
                    if nk < j:
                        L = h - nk
                    if k < j:
                        A[m] = A[m] + A[k] * Rel
                        h = k
                    for i in range(h, L - 1, -1):
                        A[i] = A[i] + (A[i - 1] - A[i]) * Rel
                pfSubsys[l] = 1 - A[m]
            surv = 1-pfSubsys.copy()
            survC = np.prod(surv)
            pfSys[wt] = 1 - survC
        return pfSys

    def immediate_cost(self, B, a, B_, drate):
        """ Computes the immediate reward (negative cost) based on current (and next) damage probability and action selected
        
            Args:
                B: Numpy array with current damage probability.
                a: Numpy array with actions selected.
                B_: Numpy array with the next time step damage probability.
                d_rate: Numpy array with current deterioration rates.
            
            Returns:
                cost_system: Float indicating the reward received.
        """
        cost_system = 0
        PF = np.sum(B[:, :, -self.proba_size_q:], axis=2)
        PF_ = np.sum(B_[:, :, -self.proba_size_q:], axis=2).copy()
        for i in range(self.n_owt):
            for j in range(self.lev-self.n_mdcomp): 
                # Identify which component is this
                if j<self.n_awcomp: comp_ind = 0 # Above-water component
                elif j<(self.n_awcomp+self.n_bwcomp): comp_ind = 1 # Below-water component
                else: comp_ind = 2 # Mudline component
                # Perfect repair 
                if a[(self.lev-self.n_mdcomp)*i+j] == 4 and comp_ind == 0: # atomospheric component
                    cost_system += -10 
                elif a[(self.lev-self.n_mdcomp)*i+j] == 4 and comp_ind == 1: # splash zone component
                    cost_system += -30
                # Perfect repair - install sensor    
                elif a[(self.lev-self.n_mdcomp)*i+j] == 5 and comp_ind == 0: # atomospheric component
                    cost_system += -12 if self.campaign_cost else  -16 
                elif a[(self.lev-self.n_mdcomp)*i+j] == 5 and comp_ind == 1: # splash zone component
                    cost_system +=  -36 if self.campaign_cost else -48
                # Do nothing
                else:
                    Bplus = B[i, j, :].dot(self.T0[comp_ind, drate[i, j, 0]]) 
                    PF_[i,j] = np.sum(Bplus[-self.proba_size_q:])
                    # Do nothing - Inspection
                    if a[(self.lev-self.n_mdcomp)*i+j] == 1 and comp_ind == 0: # atomospheric component
                        cost_system += -1 if self.campaign_cost else -3
                    elif a[(self.lev-self.n_mdcomp)*i+j] == 1 and comp_ind == 1: # splash zone component
                        cost_system += -3 if self.campaign_cost else  -9 
                    # Do nothing - Install sensor
                    elif a[(self.lev-self.n_mdcomp)*i+j] == 2 and comp_ind == 0: # atomospheric component
                        cost_system += -2 if self.campaign_cost else  -6
                    elif a[(self.lev-self.n_mdcomp)*i+j] == 2 and comp_ind == 1: # splash zone component
                        cost_system += -6 if self.campaign_cost else  -18     
                    # Do nothing - Inspection - Install sensor
                    elif a[(self.lev-self.n_mdcomp)*i+j] == 3 and comp_ind == 0: # atomospheric component
                        cost_system += -3 if self.campaign_cost else  -9
                    elif a[(self.lev-self.n_mdcomp)*i+j] == 3 and comp_ind == 1: # splash zone component
                        cost_system += -9 if self.campaign_cost else  -27
                        
        if self.lev ==3:
            PfSyS = self.pf_sys1(PF)
            PfSyS_ = self.pf_sys1(PF_)
        else:
            PfSyS = self.pf_sys(PF)
            PfSyS_ = self.pf_sys(PF_)   
        for i in range(self.n_owt):
            if PfSyS_[i] < PfSyS[i]:
                cost_system += PfSyS_[i] * (-5000)
            else:
                cost_system += (PfSyS_[i] - PfSyS[i]) * (-5000)
        if self.campaign_cost and np.sum(a)>0:  # There is at least one inspection or repair
            cost_system += -10
        return cost_system

    def belief_update_uncorrelated(self, damage_proba, twin_state, epsilon_proba, action, drate):
        """ Transitions the environment based on the current damage prob, actions selected, and current deterioration rate
            In this case, the initial damage prob are not correlated among components.
        
        Args:
            proba: Numpy array with current damage probability.
            action: Numpy array with actions selected.
            drate: Numpy array with current deterioration rates.

        Returns:
            inspection: Integers indicating which inspection outcomes have been collected.
            new_proba: Numpy array with the next time step damage probability.
            new_drate: Numpy array with the next time step deterioration rate.
        """
        next_damage_proba = np.zeros(damage_proba.shape)
        next_twin_state = np.zeros(twin_state.shape)
        next_epsilon_proba = epsilon_proba.copy()

        ob = np.ones((self.n_owt, self.lev))*2*self.proba_size_q
        next_drate = np.zeros((self.n_owt, self.lev, 1), dtype=int)
        for i in range(self.n_owt):
            for j in range(self.lev-self.n_mdcomp):
                if j<self.n_awcomp: comp_ind = 0 # Above-water component
                elif j<(self.n_awcomp+self.n_bwcomp): comp_ind = 1 # Below-water component
                else: comp_ind = 2 # Mudline component
                # print(twin_state[i,j,])
                twin_presence = np.nonzero(twin_state[i, j, :])[0][0]
                # TRANSITION THE PHYSICAL TWIN
                if action[(self.lev - self.n_mdcomp) * i + j] == 4 or action[(self.lev - self.n_mdcomp ) * i + j] == 5:
                    next_damage_proba[i, j, :] = damage_proba[i, j, :].dot(self.Tr[comp_ind, drate[i, j, 0]]) 
                    next_drate[i, j, 0] = 0
                
                else:
                    p1 = damage_proba[i, j, :].dot(self.T0[comp_ind, drate[i, j, 0]])
                    next_drate[i, j, 0] = drate[i, j, 0] + 1
                    if action[(self.lev - self.n_mdcomp) * i + j] == 0 or action[(self.lev - self.n_mdcomp) * i + j] == 2: # No-inspection
                        if twin_presence == 0: # Belief update with load observation from physical sensor
                            prob_obs = self.O_monitor[comp_ind,:].T.dot(p1)
                            s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                            next_damage_proba[i, j, :] = p1*self.O_monitor[comp_ind,:,s1]/sum(p1*self.O_monitor[comp_ind,:,s1])    
                            ob[i, j] = s1
                        elif twin_presence == 1: # Belief update with load observation from virtual sensor
                            # Built the observation model on the go
                            qobs, epsilon = self.dtwin_observation_matrix(epsilon_proba[i, j, :], comp_ind)
                            O = np.zeros(self.O_monitor[comp_ind, :].shape)
                            O[:,0:self.proba_size_q] = qobs 
                            prob_obs = O.T.dot(p1)
                            s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                            next_damage_proba[i, j, :] = p1*O[:,s1]/sum(p1*O[:,s1])    
                            next_epsilon_proba[i, j, 0] = epsilon # The second parameter controls how uncertain the turbine will evolve
                            ob[i, j] = s1
                        else: # No belief update
                            next_damage_proba[i, j, :] = p1

                    if action[(self.lev - self.n_mdcomp) * i + j] == 1 or action[(self.lev - self.n_mdcomp) * i + j] == 3: # Inspection
                        if twin_presence == 0:# Belief update with load observation from physical sensor and inspection
                            prob_obs = self.O_ins_monitor[comp_ind,:].T.dot(p1)
                            s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                            next_damage_proba[i, j, :] = p1*self.O_ins_monitor[comp_ind,:,s1]/sum(p1*self.O_ins_monitor[comp_ind,:,s1]) 
                            ob[i, j ] = s1 
                        elif twin_presence == 1: # Belief update with load observation from virtual sensor and inspection
                            # Built the observation model on the go
                            qobs, epsilon = self.dtwin_observation_matrix(epsilon_proba[i, j, :], comp_ind)
                            O = np.zeros(self.O_ins_monitor[comp_ind,:].shape)
                            O = np.concatenate((qobs.T*(self.O_ins[comp_ind,:,0]), qobs.T*(self.O_ins[comp_ind,:,self.proba_size_q])),axis=0).T
                            prob_obs = O.T.dot(p1)
                            s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                            next_damage_proba[i, j, :] = p1*O[:,s1]/sum(p1*O[:,s1]) 
                            next_epsilon_proba[i, j, 0] = epsilon # The second parameter controls how uncertain the turbine will evolve
                            ob[i, j] = s1 
                        else: # Belief update with inspection
                            prob_obs = self.O_ins[comp_ind,:].T.dot(p1)
                            s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                            next_damage_proba[i, j, :] = p1*self.O_ins[comp_ind,:,s1]/sum(p1*self.O_ins[comp_ind,:,s1]) 
                            ob[i, j] = s1

               # TRANSITION THE PHYSICAL TWIN  
                if action[(self.lev - self.n_mdcomp) * i + j] == 4: # Perfect repair (Go back to no physical/virtual sensor)
                    next_twin_state[i, j ,:] = twin_state[i, j ,:].dot(self.Tr_twin)
                    next_epsilon_proba[i, j, 0] = 0.1
                elif action[(self.lev - self.n_mdcomp) * i + j] == 0 or action[(self.lev - self.n_mdcomp) * i + j] == 1:  
                    next_twin_state[i, j, :] = twin_state[i, j ,:].dot(self.T0_twin) 
                else:  # Install sensor
                    next_twin_state[i, j, :] = twin_state[i, j ,:].dot(self.Ts_twin)
                    next_epsilon_proba[i, j, 0] = 0.1
            
            ## Turbine level that cannot be inspected nor repaired
            for j in range(self.n_mdcomp):
                p1 = damage_proba[i, self.n_awcomp+self.n_bwcomp+j, :].dot(self.T0[-1, drate[i, self.n_awcomp+self.n_bwcomp+j, 0]])
                next_damage_proba[i, self.n_awcomp+self.n_bwcomp+j, :] = p1
                ob[i, self.n_awcomp+self.n_bwcomp+j] = 2*self.proba_size_q
                # if do nothing, you update your damage prob without new evidence
                next_drate[i, self.n_awcomp+self.n_bwcomp+j, 0] = drate[i, self.n_awcomp+self.n_bwcomp+j, 0] + 1
                # At every timestep, the deterioration rate increases
            
        return ob, next_damage_proba, next_twin_state, next_epsilon_proba, next_drate

    def dtwin_observation_matrix(self, beps, comp_ind):
        epsilon = np.random.normal(beps[0],beps[0]*beps[1],(1,1))
        qint = np.tile(self.q_ref[comp_ind],(100,1)).T
        while epsilon < 0:
            epsilon = np.random.normal(beps[0],beps[0]*beps[1],(1,1))
        # qobs_std = np.ones((100,))*epsilon          
        qobs = np.zeros((self.proba_size_q, self.proba_size_q))
        for k in range(self.proba_size_q):
            qobs_mean = np.linspace(self.q_interv[comp_ind, k],self.q_interv[comp_ind, k+1],100)
            #Negative samples are taken as first bin q_ref[0] = -1e100
            qobs_cdf = stats.norm.cdf(qint, qobs_mean, (0.07+epsilon)*qobs_mean).T
            qobs_pdf = np.diff(qobs_cdf)/100
            qobs[k,:] += np.sum(qobs_pdf, axis=0) 
        qobs = np.tile(qobs,(self.proba_size_d,1))
        return qobs, epsilon