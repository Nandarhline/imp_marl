import numpy as np
from datetime import datetime
from os import path, makedirs
from imp_env.owf_envtwin import Owf_twin

class Heuristics():
    def __init__(self,
                 n_owt: int = 1,
                 # Number of structure
                 lev: int = 3,
                 discount_reward: float = 0.95,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        self.n_owt = n_owt
        self.lev = lev
        self.discount_reward = discount_reward
        self.campaign_cost = campaign_cost
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.config = {"n_owt": n_owt,
                       "lev": lev,
                       "discount_reward": discount_reward,
                       "campaign_cost": campaign_cost}
        self.owf_env = Owf_twin(self.config)
        self.date_record = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    def search_insinterv(self, eval_size):
        insp_interval = np.arange(1, self.owf_env.ep_length)
        comp_inspection = np.arange(1, self.owf_env.n_agents+1)
        heur = np.meshgrid(insp_interval, comp_inspection)
        insp_list = heur[0].reshape(-1)
        comp_list = heur[1].reshape(-1)
        ret_opt = -10000
        ind_opt = 0
        ret_total = []
        for ind in range(len(insp_list)):
            return_heur = 0
            for _ in range(eval_size):
                return_heur += self.episode_insinterv(insp_list[ind], comp_list[ind])
            return_heur /= eval_size
            ret_total.append(return_heur)
            if return_heur > ret_opt:
                ret_opt = return_heur
                ind_opt = ind
            print('Heur_results', return_heur, 'insp_int', insp_list[ind], 'n_comp', comp_list[ind])
        self.opt_heur = {"opt_reward_mean": ret_opt,
                         "insp_interv": insp_list[ind_opt],
                         "insp_comp": comp_list[ind_opt]}
        
        if not self.campaign_cost:
            camp_file = 'ref'
        else:
            camp_file = 'camp'
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez('heuristics/Owf_Results/insinterv_heuristics_'+ str(self.n_owt) + '_' + str(self.lev) + camp_file + '_' + self.date_record,
                  ret_total = ret_total, opt_heur = self.opt_heur, config=self.config, seed_test=self._seed)
        return self.opt_heur

    def eval_insinterv(self, eval_size, insp_int, comp_insp):
        self.return_heur = 0
        for ep in range(eval_size):
            self.return_heur += self.episode_insinterv(insp_int, comp_insp)
            disp_cost = self.return_heur/(ep+1)
            if (ep+1)%500==0:
                print('Reward:', disp_cost)
        self.return_heur /= eval_size
        return self.return_heur
    
    def episode_insinterv(self, insp_int, comp_insp):
        rew_total_ = 0
        done_ = False
        obs = np.ones((self.owf_env.n_owt,self.owf_env.lev))*self.owf_env.proba_size_q*2
        self.owf_env.reset()
        action = {}
        for agent in self.owf_env.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.owf_env.time_step%insp_int)==0 and self.owf_env.time_step>0:
                beliefs_comp = np.reshape(self.owf_env.damage_proba[:,:-1,:], (self.owf_env.n_agents, -1))
                pf = np.sum(beliefs_comp[:,-self.owf_env.proba_size_q:],axis=1)
                inspection_index = (-pf).argsort()[:comp_insp]
                for index in inspection_index:
                    action_[self.owf_env.agent_list[index]] = 1
            if np.sum(obs) < self.owf_env.n_owt*self.owf_env.lev*self.owf_env.proba_size_q*2:
                obs_ag = np.reshape(obs[:,:-1], (self.owf_env.n_agents, -1) )
                index_repair = np.where(obs_ag<self.owf_env.proba_size_q)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.owf_env.agent_list[index]] = 4
            [bel_, rew_, done_, obs] = self.owf_env.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_

    def search_sensinterv(self, eval_size):
        sens_interval = np.arange(1, self.owf_env.ep_length)
        comp_sensing = np.arange(1, self.owf_env.n_agents+1)
        heur = np.meshgrid(sens_interval, comp_sensing)
        sens_list = heur[0].reshape(-1)
        comp_list = heur[1].reshape(-1)
        ret_opt = -10000
        ind_opt = 0
        ret_total = []
        for ind in range(len(sens_list)):
            return_heur = 0
            for _ in range(eval_size):
                return_heur += self.episode_sensinterv(sens_list[ind], comp_list[ind])
            return_heur /= eval_size
            ret_total.append(return_heur)
            if return_heur > ret_opt:
                ret_opt = return_heur
                ind_opt = ind
            print('Heur_results', return_heur, 'sens_int', sens_list[ind], 'n_comp', comp_list[ind])
        self.opt_heur = {"opt_reward_mean": ret_opt,
                         "sens_interv": sens_list[ind_opt],
                         "sens_comp": comp_list[ind_opt]}
        
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez('heuristics/Owf_Results/sensinterv_heuristics_'+ str(self.n_owt) + '_' + str(self.lev) + camp_file + '_' + self.date_record,
                  ret_total = ret_total, opt_heur = self.opt_heur, config=self.config, seed_test=self._seed)
        return self.opt_heur
    
    def eval_sensinterv(self, eval_size, sens_int, comp_insp):
        self.return_heur = 0
        for ep in range(eval_size):
            self.return_heur += self.episode_sensinterv(sens_int, comp_insp)
            disp_cost = self.return_heur/(ep+1)
            if (ep+1)%500==0:
                print('Reward:', disp_cost)
        self.return_heur /= eval_size
        return self.return_heur    
    
    def episode_sensinterv(self, sens_int, comp_insp):
        rew_total_ = 0
        done_ = False
        obs = np.ones((self.owf_env.n_owt,self.owf_env.lev))*self.owf_env.proba_size_q*2
        self.owf_env.reset()
        action = {}
        for agent in self.owf_env.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.owf_env.time_step%sens_int)==0 and self.owf_env.time_step>0:
                beliefs_comp = np.reshape(self.owf_env.damage_proba[:,:-1,:], (self.owf_env.n_agents, -1))
                pf = np.sum(beliefs_comp[:, -self.owf_env.proba_size_q:], axis=1)
                sensor_index = (-pf).argsort()[:comp_insp]
                for index in sensor_index:
                    action_[self.owf_env.agent_list[index]] = 2
            if np.sum(obs) < self.owf_env.n_owt*self.owf_env.lev*self.owf_env.proba_size_q*2:
                obs_ag = np.reshape(obs[:,:-1], (self.owf_env.n_agents, -1) )
                index_repair = np.where(obs_ag%self.owf_env.proba_size_q>4)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.owf_env.agent_list[index]] = 4
            [bel_, rew_, done_, obs] = self.owf_env.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_
    
   
