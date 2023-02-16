import numpy as np


class Struct_twin:

    def __init__(self, config=None):
        if config is None:
            config = {"n_comp": 2,
                      "discount_reward": 1,
                      "k_comp": None,
                      "env_correlation": False,
                      "campaign_cost": False}
        assert "n_comp" in config and \
               "discount_reward" in config and \
               "k_comp" in config and \
               "env_correlation" in config and \
               "campaign_cost" in config, \
            "Missing env config"

        self.n_comp = config["n_comp"]
        self.discount_reward = config["discount_reward"]
        self.k_comp = self.n_comp - 1 if config["k_comp"] is None \
            else config["k_comp"]
        self.env_correlation = config["env_correlation"]
        self.campaign_cost = config["campaign_cost"]
        self.time = 0
        self.ep_length = 20  # Horizon length
        self.n_st_comp = 30  # Crack states (fatigue hotspot damage states)
        self.n_st_stress = 20  # Stress states (fatigue hotspot damage states)
        self.n_st_hyperp = 80 if self.env_correlation else None
        # Gaussian hyperparemeter states
        self.n_obs = 40 
        # Total number of observations (crack detected * n_st_stress + crack not detected * n_st_stress)
        self.actions_per_agent = 6

        # Uncorrelated obs = 30 per agent + 1 timestep
        # Correlated obs = 30 per agent + 1 timestep +
        #                   80 hyperparameter states = 111
        self.obs_per_agent_multi = None  # Todo: check
        self.obs_total_single = None  # Todo: check used in gym env

        ### Loading the underlying POMDP model ###
        if not self.env_correlation:
            drmodel = np.load('pomdp_models/Dr203020_Tw07.npz')
        else:
            drmodel = np.load('pomdp_models/Dr3031_H08.npz')

        
        if not self.env_correlation:
            self.belief0 = np.tile(drmodel['belief0'],(self.n_comp,1))
            self.belief0_twin = np.tile(drmodel['belief0_twin'],(self.n_comp,1))
            
            self.T0 = drmodel['T0'] 
            self.Tr = drmodel['Tr']
            self.T0_twin = drmodel['T0_twin']
            self.Tr_twin = drmodel['Tr_twin']
            
            self.O_no_obs = drmodel['O_no_obs']
            self.O_ins = drmodel['O_ins']
            self.O_monitor = drmodel['O_monitor']
            self.O_ins_monitor = drmodel['O_ins_monitor']

            self.belief0c = None
            self.b0cR = None
            self.alpha0 = None

        else:
            self.belief0[:, :] = drmodel['belief0']
            # (3 actions, 31 det rates, 30 cracks, 30 cracks)
            self.P = drmodel['P']

            # (3 actions, 30 cracks, 2 observations)
            self.O = drmodel['O']

            self.belief0c = \
                np.zeros((self.n_comp, self.n_st_hyperp, self.n_st_comp))
            self.belief0c[:, :, :] = drmodel['belief0c']

            # conditional beliefs associated with a repair action
            # (80 hyperparameter states, 30 crack states)
            self.b0cR = drmodel['b0cR']

            # hyperparameter marginal states
            self.alpha0 = drmodel['alpha0']

        self.agent_list = ["agent_" + str(i) for i in range(self.n_comp)]

        self.time_step = 0
        self.beliefs = self.belief0
        self.beliefs_twin = self.belief0_twin
        self.beliefsc = self.belief0c
        self.alphas = self.alpha0
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = None

        # Reset struct_env.
        self.reset()

    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's belief
        self.time_step = 0
        self.beliefs = self.belief0
        self.beliefs_twin = self.belief0_twin
        self.beliefsc = self.belief0c
        self.alphas = self.alpha0
        self.d_rate = np.zeros((self.n_comp, 1), dtype=int)
        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (self.beliefs[i], [self.time_step / self.ep_length]))

        return self.observations

    def step(self, action: dict):
        action_ = np.zeros(self.n_comp, dtype=int)
        for i in range(self.n_comp):
            action_[i] = action[self.agent_list[i]]

        if not self.env_correlation:
            observation_, belief_prime, belief_twin_prime, drate_prime = \
                self.belief_update_uncorrelated(self.beliefs, self.beliefs_twin, action_,
                                                self.d_rate)

        else:
            observation_, belief_prime, drate_prime, bc_prime, alpha_prime = \
                self.belief_update_correlated(self.beliefsc, action_,
                                              self.d_rate, self.alphas)

        reward_ = self.immediate_cost(self.beliefs, action_, belief_prime,
                                      self.d_rate)
        reward = self.discount_reward ** self.time_step * reward_.item()  # Convert float64 to float

        rewards = {}
        for i in range(self.n_comp):
            rewards[self.agent_list[i]] = reward

        self.time_step += 1

        self.observations = {}
        for i in range(self.n_comp):
            self.observations[self.agent_list[i]] = np.concatenate(
                (belief_prime[i], [self.time_step / self.ep_length]))

        if self.env_correlation:
            self.beliefsc = bc_prime
            self.alphas = alpha_prime

        self.beliefs = belief_prime
        self.beliefs_twin = belief_twin_prime
        self.d_rate = drate_prime

        # An episode is done if the agent has reached the target
        done = self.time_step >= self.ep_length

        # info = {"belief": self.beliefs}
        return self.observations, rewards, done, observation_

    def pf_sys(self, pf, k):
        """compute pf_sys for k-out-of-n components"""
        n = pf.size
        # k = ncomp-1
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def immediate_cost(self, B, a, B_, drate):
        """ immediate reward (-cost),
         based on current damage state and action """
        cost_system = 0
        PF = np.sum(B[:, -self.n_st_stress:], axis=1)
        PF_ = np.sum(B_[:, -self.n_st_stress:], axis=1).copy()
        campaign_executed = False
        for i in range(self.n_comp):
            if a[i] == 4 or a[i] == 5: # Perfect repair
                cost_system += - 10
                if a[i] == 5: # install sensor
                    cost_system += - 3
                if self.campaign_cost and not campaign_executed:
                    campaign_executed = True # Campaign executed
            else: # Do-nothing
                Bplus = B[i, :].dot(self.T0[drate[i, 0]]) 
                PF_[i] = np.sum(Bplus[-self.n_st_stress:])
                if a[i] == 1:
                    cost_system += -1 #if self.campaign_cost else -1 # Individual inspection costs 
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
                if a[i] == 2:
                    cost_system += -3 #if self.campaign_cost else -3 # Individual install-sensor costs 
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
                if a[i] == 3:
                    cost_system += -4 #if self.campaign_cost else -4 # Individual inspection/install-sensor costs 
                    if self.campaign_cost and not campaign_executed:
                        campaign_executed = True # Campaign executed
        if self.n_comp < 2:  # single component setting
            PfSyS_ = PF_
            PfSyS = PF
        else:
            PfSyS_ = self.pf_sys(PF_, self.k_comp)
            PfSyS = self.pf_sys(PF, self.k_comp)
        if PfSyS_ < PfSyS:
            cost_system += PfSyS_ * (-10000)
        else:
            cost_system += (PfSyS_ - PfSyS) * (-10000)
        if campaign_executed: # Assign campaign cost
            cost_system += -5
        return cost_system

    def belief_update_uncorrelated(self, b, btwin, a, drate):
        """Bayesian belief update based on
         previous belief, current observation, and action taken"""
        b_prime = np.zeros(b.shape)
        btwin_prime = np.zeros(btwin.shape) 
        ob = np.zeros(self.n_comp)
        drate_prime = np.zeros((self.n_comp, 1), dtype=int)
        for i in range(self.n_comp):
            ob[i] = 41
            twin_state = np.nonzero(np.random.multinomial(1, btwin[i,:]))[0][0]
            # TRANSITION THE PHYSICAL TWIN
            if a[i] == 4 or a[i] == 5: # Perfect-repair                
                p1 = b[i, :].dot(self.Tr[drate[i, 0]]) 
                drate_prime[i, 0] = 0 
                if twin_state<len(btwin[i,:])-1: # Virtual observation
                    prob_obs = self.O_monitor.T.dot(p1)
                    s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                    b_prime[i, :] = p1*self.O_monitor[:,s1]/sum(p1*self.O_monitor[:,s1]) # belief update  
                    ob[i] = s1 
                else: 
                    b_prime[i, :] = p1 
                          
            else:   # Do-nothing
                p1 = b[i, :].dot(self.T0[drate[i, 0]])
                drate_prime[i, 0] = drate[i, 0] + 1     
                if a[i] == 0 or a[i] == 2: # No-inspection
                    if twin_state<len(btwin[i,:])-1: # Belief update with virtual observation
                        prob_obs = self.O_monitor.T.dot(p1)
                        s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                        b_prime[i, :] = p1*self.O_monitor[:,s1]/sum(p1*self.O_monitor[:,s1])    
                        ob[i] = s1 
                    else: # No belief update
                        b_prime[i, :] = p1
                    
                elif a[i] == 1 or a[i] == 3: # Inspection
                    if twin_state<len(btwin[i,:])-1: # Belief update with virtual observation and inspection
                        prob_obs = self.O_ins_monitor.T.dot(p1)
                        s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                        b_prime[i, :] = p1*self.O_ins_monitor[:,s1]/sum(p1*self.O_ins_monitor[:,s1]) 
                        ob[i] = s1 
                    else: # Belief update with inspection
                        prob_obs = self.O_ins.T.dot(p1)
                        s1 = np.nonzero(np.random.multinomial(1, prob_obs))[0][0]
                        b_prime[i, :] = p1*self.O_ins[:,s1]/sum(p1*self.O_ins[:,s1]) 
                        ob[i] = s1 
           
            # TRANSITION THE DIGITAL  TWIN  
            if a[i] == 2 or a[i] == 3 or a[i] == 5: # Install sensor 
                btwin_prime[i,:] = btwin[i,:].dot(self.Tr_twin)
            else:  
                btwin_prime[i,twin_state] = 1 # The digital twin belief becomes the fully-observed state
                btwin_prime[i,:] = btwin_prime[i,:].dot(self.T0_twin) # Transition the digital twin belief stochastically
                         
        return ob, b_prime, btwin_prime, drate_prime

    def belief_update_correlated(self, bc, a, drate, alpha):
        """Bayesian belief update based on previous belief,
         current observation, and action taken"""

        b_prime = np.zeros((self.n_comp, self.n_st_comp))
        bc_prime = np.zeros((self.n_comp, self.n_st_hyperp, self.n_st_comp))
        drate_prime = np.zeros((self.n_comp, 1), dtype=int)
        alpha_prime = np.zeros((self.n_st_hyperp))
        alpha_prime = alpha.copy()
        ob = np.zeros(self.n_comp)
        for i in range(self.n_comp):
            p1 = bc[i, :, :].dot(
                self.P[a[i], drate[i, 0]])  # environment transition
            bc_prime[i, :, :] = p1
            drate_prime[i, 0] = drate[i, 0] + 1

            if a[i] == 1:
                Obs0 = np.sum(alpha[:].dot(p1) * self.O[a[i], :, 0])
                Obs1 = 1 - Obs0
                if Obs1 < 1e-5:
                    ob[i] = 0
                else:
                    ob_dist = np.array([Obs0, Obs1])
                    ob[i] = np.random.choice(range(0, self.n_obs), size=None,
                                             replace=True, p=ob_dist)
                pInsp = p1 * self.O[a[i], :, int(ob[i])]  # belief update
                likAlpha = np.sum(pInsp, axis=1)  # likelihood insp alpha
                normBel = np.tile(likAlpha, self.n_st_comp).reshape(
                    self.n_st_hyperp, self.n_st_comp,
                    order='F')  # normalization constant
                bc_prime[i, :, :] = pInsp / normBel
                alpha_curr = likAlpha * alpha_prime
                alpha_prime = alpha_curr / np.sum(alpha_curr)

            if a[i] == 2:
                bc_prime[i, :, :] = self.b0cR
                drate_prime[i, 0] = 0

        for i in range(self.n_comp):
            b_prime[i, :] = alpha_prime.dot(
                bc_prime[i, :, :])  # Belief (marginalize out alpha)
        return ob, b_prime, drate_prime, bc_prime, alpha_prime
