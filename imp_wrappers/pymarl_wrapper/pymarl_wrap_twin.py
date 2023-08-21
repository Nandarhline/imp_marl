""" Wrapper for struct_env respecting the interface of PyMARL. """

import numpy as np
import torch

from imp_env.owf_envtwin import Owf_twin
from imp_env.struct_envtwin import Struct_twin
from imp_wrappers.pymarl_wrapper.MultiAgentEnv import MultiAgentEnv


class PymarlMATwin(MultiAgentEnv):
    """
    Wrapper for Struct_twin and Owf_twin respecting the interface of PyMARL.

    It manipulates an imp_env to create all inputs for PyMARL agents.
    """

    def __init__(self,
                 struct_type: str = "struct",
                 n_comp: int = 2,
                 custom_param: dict = None,
                 discount_reward: float = 1.,
                 state_obs: bool = True,
                 state_twin: bool = True,
                 state_eps: bool = True,
                 state_d_rate: bool = False,                
                 obs_twin: bool = True,
                 obs_eps: bool = True,
                 obs_d_rate: bool = False,
                 obs_multiple: bool = False,
                 obs_all_d_rate: bool = False,
                 campaign_cost: bool = False,
                 virtual_sensor: bool = True,
                 seed=None):
        """
        Initialise based on the full configuration.

        Args:
            struct_type: (str) Type of the struct env, either "struct" or "owf".
            n_comp: (int) Number of structure
            custom_param: (dict)
                struct: Number of structure required
                        {"k_comp": int} for k_comp out of n_comp
                        Default is None, meaning k_comp=n_comp-1
                 owf: Number of levels per wind turbine
                        {"lev": int}
                        Default is 3
            discount_reward: (float) Discount factor [0,1[
            state_obs: (bool) State contains the concatenation of obs
            state_twin: (bool) State contains the concatenation of physical/virtual twin state
            state_eps: (bool) State contains the concatenation of physical/virtual sensor's uncertainty
            state_d_rate: (bool) State contains the concatenation of drate          
            obs_twin: (bool) State contains the concatenation of all physical/virtual twin states
            obs_eps: (bool) State contains the concatenation of all physical/virtual sensor's uncertainties
            obs_d_rate: (bool) Obs contains the drate of the agent
            obs_multiple: (bool) Obs contains the concatenation of all obs
            obs_all_d_rate: (bool) Obs contains the concatenation of all drate
            campaign_cost: (bool) campaign_cost = True=campaign cost taken into account
            virtual_sensor: (bool) virtual_sensor = True=virtual sensor is trained after installing a physical sensor
            seed: (int) seed for the random number generator
        """
        # Check struct type and default values
        assert struct_type == "owf" or struct_type == "struct", "Error in struct_type"
        if struct_type == "struct":
            self.k_comp = custom_param.get("k_comp", None) if (
                        custom_param is not None) else None
            assert self.k_comp is None or self.k_comp <= n_comp, "Error in k_comp"
        elif struct_type == "owf":
            self.lev = custom_param.get("lev", 3) if (
                        custom_param is not None) else 3
            assert self.lev is not None, "Error in lev"

        assert isinstance(state_obs, bool) \
               and isinstance(state_twin, bool) \
               and isinstance(state_eps, bool) \
               and isinstance(state_d_rate, bool) \
               and isinstance(obs_twin, bool) \
               and isinstance(obs_eps, bool) \
               and isinstance(obs_d_rate, bool) \
               and isinstance(obs_multiple, bool) \
               and isinstance(obs_all_d_rate, bool) \
               and isinstance(campaign_cost, bool) \
               and isinstance(virtual_sensor, bool), "Error in env parameters"
        assert 0 <= discount_reward <= 1, "Error in discount_reward"
        assert not (obs_d_rate and obs_all_d_rate), "Error in env parameters"
        assert state_obs or state_d_rate, "Error in env parameters"

        self.n_comp = n_comp
        self.custom_param = custom_param
        self.discount_reward = discount_reward
        self.state_obs = state_obs
        self.state_twin = state_twin
        self.state_eps = state_eps
        self.state_d_rate = state_d_rate
        self.obs_twin = obs_twin
        self.obs_eps = obs_eps
        self.obs_d_rate = obs_d_rate
        self.obs_multiple = obs_multiple
        self.obs_all_drate = obs_all_d_rate
        self.campaign_cost = campaign_cost
        self.virtual_sensor = virtual_sensor
        self._seed = seed

        if struct_type == "struct":
            self.config = {"n_comp": n_comp,
                           "discount_reward": discount_reward,
                           "k_comp": self.k_comp,
                           "campaign_cost": campaign_cost}
            self.struct_env = Struct_twin(self.config)
            self.n_agents = self.struct_env.n_comp
        elif struct_type == "owf":
            self.config = {"n_owt": n_comp,
                           "lev": self.lev,
                           "discount_reward": discount_reward,
                           "campaign_cost": campaign_cost,
                           "virtual_sensor": virtual_sensor
                           }

            self.struct_env = Owf_twin(self.config)
            self.n_agents = self.struct_env.n_agents

        self.episode_limit = self.struct_env.ep_length
        self.agent_list = self.struct_env.agent_list
        self.n_actions = self.struct_env.actions_per_agent

        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}

        self.unit_dim = self.get_unit_dim()  # Qplex requirement

    def update_action_histogram(self, actions):
        """
        Update the action histogram for logging.

        Args:
            actions: list of actions
        """
        for k, action in zip(self.struct_env.agent_list, actions):
            if type(action) is torch.Tensor:
                action_str = str(action.cpu().numpy())
            else:
                action_str = str(action)
            self.action_histogram["action_" + action_str] += 1

    def step(self, actions):
        """
        Ask to run a step in the environment.

        Args:
            actions: list of actions

        Returns:
            rewards: list of rewards
            done: True if the episode is finished
            info: dict of info for logging
        """

        self.update_action_histogram(actions)
        action_dict = {k: action
                       for k, action in
                       zip(self.struct_env.agent_list, actions)}
        _, rewards, done, _ = self.struct_env.step(action_dict)
        info = {}
        if done:
            for k in self.action_histogram:
                self.action_histogram[k] /= self.episode_limit * self.n_agents
            info = self.action_histogram
        return rewards[self.struct_env.agent_list[0]], done, info

    def get_obs(self):
        """ Returns all agent observations in a list. """
        return [self.get_obs_agent(i) for i in
                range(self.n_agents)]

    def get_unit_dim(self):
        """ Returns the dimension of the unit observation used by QPLEX. """
        return len(self.all_obs_from_struct_env()) // self.n_agents

    def get_obs_agent(self, agent_id: int):
        """
        Returns observation for agent_id

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        agent_name = self.struct_env.agent_list[agent_id]

        if self.obs_multiple:
            obs = self.all_obs_from_struct_env()
        else:
            obs = self.struct_env.observations[agent_name]

        if self.obs_twin:
            obs = np.append(obs, self.get_twin_state()[agent_id])

        if self.obs_eps:
            obs = np.append(obs, self.get_epsilon()[agent_id])

        if self.obs_d_rate:
            obs = np.append(obs, self.get_normalized_drate()[agent_id])

        if self.obs_all_drate:
            obs = np.append(obs, self.get_normalized_drate())

        return obs

    def get_obs_size(self):
        """ Returns the size of the observation. """
        return len(self.get_obs_agent(0))

    def get_normalized_drate(self):
        """ Returns the normalized d_rate. """
        return self.struct_env.d_rate / self.struct_env.ep_length

    def all_obs_from_struct_env(self):
        """ Returns all observations concatenated in a single vector. """
        # Concatenate all obs with a single time.
        idx = 0
        obs = None
        for k, v in self.struct_env.observations.items():
            if idx == 0:
                obs = v
                idx = 1
            else:
                obs = np.append(obs, v)
        return obs
    
    def get_twin_state(self):
        """ Returns the agents' twin state. """
        twin_states = np.reshape(self.struct_env.twin_state, (self.n_agents,-1))
        return twin_states
    
    def get_epsilon(self):
        """ Returns the agents' epsilon parameters. """
        epsilons = np.reshape(self.struct_env.epsilon_proba, (self.n_agents,-1))
        return epsilons

    def get_state(self):
        """ Returns the state of the environment. """
        state = []
        if self.state_obs:
            state = np.append(state, self.all_obs_from_struct_env())
        if self.state_twin:
            state = np.append(state, np.reshape(self.get_twin_state(),-1))
        if self.state_eps:
            state = np.append(state, np.reshape(self.get_epsilon(),-1))
        if self.state_d_rate:
            state = np.append(state, self.get_normalized_drate())
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        """ Returns the available actions of all agents in a list. """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """
        Returns the available actions for agent_id.

        Args:
            agent_id: id of the agent (int in range(self.n_agents)).
        """
        return [1] * self.n_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take. """
        return self.struct_env.actions_per_agent

    def reset(self):
        """ Returns initial observations and states. """
        self.action_histogram = {"action_" + str(k): 0 for k in
                                 range(self.n_actions)}
        self.struct_env.reset()
        return self.get_obs(), self.get_state()

    def render(self):
        """ See base class. """
        pass

    def close(self):
        """ See base class. """
        pass

    def seed(self):
        """ Returns the random seed """
        return self._seed

    def save_replay(self):
        """ See base class. """
        pass

    def get_stats(self):
        """ See base class. """
        return {}
