import numpy as np
from datetime import datetime
from os import path, makedirs

from imp_env.struct_envtwin import Struct_twin


class Heuristics():
    def __init__(self,
                 n_comp: int = 3,
                 # Number of structure
                 discount_reward: float = 1.,
                 # float [0,1] importance of
                 # short-time reward vs long-time reward
                 k_comp: int = 3,
                 # Number of structure required (k_comp out of n_comp)
                 campaign_cost: bool = False,
                 # campaign_cost = True=campaign cost taken into account
                 seed=None):

        self.n_comp = n_comp
        self.k_comp = k_comp
        self.discount_reward = discount_reward
        self.campaign_cost = campaign_cost
        self._seed = seed

        self.config = {"n_comp": n_comp,
                       "discount_reward": discount_reward,
                       "k_comp": k_comp,
                       "campaign_cost": campaign_cost}
        self.struct_envtwin = Struct_twin(self.config)
        self.date_record = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    def search_insinterv(self, eval_size):
        insp_interval = np.arange(1, self.struct_envtwin.ep_length)
        comp_inspection = np.arange(1, self.struct_envtwin.n_comp+1)
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
        
        path_results = "heuristics/Results"
        isExist = path.exists(path_results)
        if not isExist:
            makedirs(path_results)
        np.savez('heuristics/Results/insinterv_heuristics_'+ self.date_record, ret_total = ret_total, opt_heur = self.opt_heur)
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
        obs = np.ones((self.struct_envtwin.n_comp,))*self.struct_envtwin.n_st_stress*2
        self.struct_envtwin.reset()
        action = {}
        for agent in self.struct_envtwin.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.struct_envtwin.time_step%insp_int)==0 and self.struct_envtwin.time_step>0:
                pf = np.sum(self.struct_envtwin.beliefs[:, -self.struct_envtwin.n_st_stress:], axis=1)
                inspection_index = (-pf).argsort()[:comp_insp] 
                for index in inspection_index:
                    action_[self.struct_envtwin.agent_list[index]] = 1
            if np.sum(obs) < self.struct_envtwin.n_comp*self.struct_envtwin.n_st_stress*2: # at least one observation in an element
                index_repair = np.where(obs<self.struct_envtwin.n_st_stress)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.struct_envtwin.agent_list[index]] = 4
            [bel_, rew_, done_, obs] = self.struct_envtwin.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_
    
    def search_sensinterv(self, eval_size):
        sens_interval = np.arange(1, self.struct_envtwin.ep_length)
        comp_sensing = np.arange(1, self.struct_envtwin.n_comp+1)
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
        np.savez('heuristics/Results/sensinterv_heuristics_'+ self.date_record, ret_total = ret_total, opt_heur = self.opt_heur)
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
        obs = np.ones((self.struct_envtwin.n_comp,))*self.struct_envtwin.n_st_stress*2
        self.struct_envtwin.reset()
        action = {}
        for agent in self.struct_envtwin.agent_list:
            action[agent] = 0
        while not done_:
            action_ = action.copy()
            if (self.struct_envtwin.time_step%sens_int)==0 and self.struct_envtwin.time_step>0:
                pf = np.sum(self.struct_envtwin.beliefs[:, -self.struct_envtwin.n_st_stress:], axis=1)
                sensor_index = (-pf).argsort()[:comp_insp]
                for index in sensor_index:
                    action_[self.struct_envtwin.agent_list[index]] = 2
            if np.sum(obs) < self.struct_envtwin.n_comp*self.struct_envtwin.n_st_stress*2:
                index_repair = np.where(obs%self.struct_envtwin.n_st_stress>3)[0]
                if len(index_repair) > 0:
                    for index in index_repair:
                        action_[self.struct_envtwin.agent_list[index]] = 4
            [bel_, rew_, done_, obs] = self.struct_envtwin.step(action_)
            rew_total_ += rew_['agent_0']
        return rew_total_
    


