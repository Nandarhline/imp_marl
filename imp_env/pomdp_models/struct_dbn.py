import numpy as np
import scipy.stats as stats
import math

class dynamicBN:

    def __init__(self, config = {"time_variant_q": False, "epsilonq_std": None}):
        self.T = config["T"] 
        self.ncycles = config["ncycles"] 
        self.d0_mean = config["d0_mean"] 
        self.dcrit = config["dcrit"] 
        self.q_mean = config["q_mean"] 
        self.q_cov = config["q_cov"] 
        self.time_variant_q = config["time_variant_q"] if config.get("time_variant_q") is not None\
            else False
        self.epsilonq_std = config["epsilonq_std"] if config.get("time_variant_q") is True\
            else None        

    
    def crack_growth(self, cg_params = None):
        np.random.seed(0)
        lnC_mean = cg_params["lnC_mean"]
        lnC_std = cg_params["lnC_std"]
        m = cg_params["m"]
        Ynormal_mean = cg_params["Ynormal_mean"]
        Ynormal_cov = cg_params["Ynormal_cov"]
        nsamples = int(cg_params["nsamples"])
        
        d0 = np.random.exponential(self.d0_mean, nsamples)
        q0 = np.random.normal(self.q_mean, self.q_cov*self.q_mean, nsamples)
        while (q0<0).sum() > 0:
            q0[q0<0] = np.random.normal(self.q_mean, self.q_cov*self.q_mean , (q0<0).sum())
        S = q0*math.gamma(1+1/0.8)
       
            
        C = np.random.lognormal(lnC_mean, lnC_std, nsamples) 
        Y_mean = np.log(Ynormal_mean)
        Y_std = (np.log(Ynormal_cov**2+1))**0.5
        Y = np.random.lognormal(Y_mean, Y_std, nsamples )  
        
        self.dd = np.zeros((self.T+1, nsamples))
        self.dd[0,:] = d0
        
        if self.time_variant_q is True:
            # epsilonq = np.random.normal(1, self.epsilonq_std, nsamples)
            self.qq = np.zeros((self.T+1, nsamples))
            self.qq[0,:] = q0
        
        for t in range(self.T):
            dt = (((2-m)/2)*C*(Y*math.pi**0.5*S)**m*self.ncycles + d0**((2-m)/2))**(2/(2-m))
            dt[(dt<d0)| (dt>self.dcrit)] = self.dcrit+0.1
            self.dd[t+1,:] = dt
            d0 = dt
            if self.time_variant_q is True:
                epsilonq = np.random.normal(1, self.epsilonq_std, nsamples)
                qt = q0*epsilonq # uncertainty of q increases
                while (qt<0).sum() > 0:
                    qt[qt<0] = q0[qt<0]*np.random.normal(1, self.epsilonq_std , (qt<0).sum())
                q0 = qt
                self.qq[t+1,:] = qt
                S = qt*math.gamma(1+1/0.8)
        return
#         if self.time_variant_q is True:
#             return self.dd, self.qq
#         else: 
#             return self.dd
        
    def transition_models(self, n_dstates=30, n_qstates = 20):
        if self.time_variant_q is True:
            self.d_interv = 0
            self.d_interv = np.append(self.d_interv, np.linspace(self.d0_mean, self.dcrit, n_dstates-1))
            self.d_interv = np.append(self.d_interv, 1e20)
            self.q_interv = np.linspace(1e-20, np.max(self.qq)+0.1, n_qstates)
            self.q_interv = np.append(self.q_interv, 1e20)
            det_rates = self.T+1
            nsamples = self.dd.shape[-1]   
        
            H, _, _ = np.histogram2d(self.dd[0,:], self.qq[0,:], [self.d_interv, self.q_interv])
            self.b0 = (H/nsamples).reshape(-1) # d is the outer loop
            
            self.T0 = np.zeros((det_rates, n_dstates*n_qstates, n_dstates*n_qstates))
            for i in range(det_rates-1):
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                Q = self.qq[i,:] # Samples q at det. rate i, 
                Q_ = self.qq[i+1,:] # Samples q at det. rate i+1
                for j in range(n_dstates):
                    countd = (D>self.d_interv[j]) &  (D<self.d_interv[j+1])
                    for k in range(n_qstates):
                        countq =(Q>self.q_interv[k]) &  (Q<self.q_interv[k+1])
                        Dnext = D_[countd & countq]
                        Qnext = Q_[countd & countq]
                        if (countd & countq).sum() < 1:
                            self.T0[i,j*n_qstates+k,j*n_qstates+k]=1
                        else:
                            H, _, _ = np.histogram2d(Dnext, Qnext , [self.d_interv, self.q_interv]) 
                            self.T0[i,j*n_qstates+k,:] = (H/(countd & countq).sum()).reshape(-1)
            self.T0[-1,] = self.T0[-2,]

            self.Tr = np.zeros((det_rates, n_dstates*n_qstates, n_dstates*n_qstates))
            self.Tr = np.tile(self.b0,(det_rates, n_dstates*n_qstates,1))
            
            return self.d_interv, self.q_interv
            
        else:
            self.d_interv = 1e-20
            self.d_interv = np.append(self.d_interv, np.linspace(self.d0_mean, self.dcrit, n_dstates-1))
            self.d_interv = np.append(self.d_interv, 1e20)
            det_rates = self.T+1
            nsamples = self.dd.shape[-1]   
        
            H, _ = np.histogram(self.dd[0,:], self.d_interv)
            self.b0 = H/nsamples
            
            self.T0 = np.zeros((det_rates, n_dstates, n_dstates))
            for i in range(det_rates-1):
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                for j in range(n_dstates):
                    countd = (D>self.d_interv[j]) &  (D<self.d_interv[j+1])
                    Dnext = D_[countd]
                    if countd.sum() < 1:
                        self.T0[i,j,j]=1
                    else:
                        H, _ = np.histogram(Dnext, self.d_interv) 
                        self.T0[i,j,:] = H/countd.sum()
            self.T0[-1,] = self.T0[-2,]

            self.Tr = np.zeros((det_rates, n_dstates, n_dstates))
            self.Tr = np.tile(self.b0,(det_rates, n_dstates,1))                
            
            return self.d_interv
    
    
    def observation_models(self, obs_params = {"inspect": True, 
                                               "beta0": 7.3704, "beta1": 2.092, "sigma_epsilon": 4.189, "det_thres": 5.4898,
                                               "monitor": False,
                                               "error_cov": None}):
        inspect = obs_params["inspect"] if obs_params.get("inspect") is not None\
            else True      
        beta0 = obs_params["beta0"] if obs_params.get("beta0") is not None\
            else 7.3704
        beta1 = obs_params["beta1"] if obs_params.get("beta1") is not None\
            else 2.092
        sigma_epsilon = obs_params["sigma_epsilon"] if obs_params.get("sigma_epsilon") is not None\
            else 4.189
        det_thres = obs_params["det_thres"] if obs_params.get("det_thres") is not None\
            else 5.4898
        monitor = obs_params["monitor"] if obs_params.get("monitor") is not None\
            else False
        error_cov = obs_params["error_cov"] if obs_params["monitor"] is True\
            else None
        
        n_dstates = len(self.d_interv)-1 
        if self.time_variant_q is True:
            n_qstates = len(self.q_interv)-1
        
        if inspect is False and monitor is False:
            dobs = np.zeros((n_dstates, 2))
            dobs[:,0] = 1
            dobs[:,1] = 0
            if self.time_variant_q is True:             
                dobs = np.repeat(dobs,n_qstates,axis=0)            
            
        if inspect is True: # inspection model
            dobs = np.zeros((n_dstates, 2))
            d_ref = (self.d_interv[0:-1]+self.d_interv[1:])/2
            dobs[:,0] = 1-stats.norm.cdf((det_thres-beta0-beta1*np.log(d_ref))/sigma_epsilon)
            dobs[:,1] = 1-dobs[:,0] 
            if self.time_variant_q is True:             
                dobs = np.repeat(dobs,n_qstates,axis=0)
                    
        if monitor is True: # monitoring model
            q_ref = -1e100
            q_ref = np.append(q_ref, self.q_interv[1:])
            q_ref[-1] = 1e100
            q_ref = np.tile(q_ref,(100,1)).T
            qobs_cov = np.ones((100,))*error_cov           
            qobs = np.zeros((n_qstates, n_qstates))
            for i in range(n_qstates):
                qobs_mean = np.linspace(self.q_interv[i],self.q_interv[i+1],100)
                qobs_cdf = stats.norm.cdf(q_ref, qobs_mean, qobs_mean*qobs_cov).T
                qobs_cdf[:,-1] = 1 # to make sure the probabilities sum to one
                qobs_pdf = np.diff(qobs_cdf)/100
                qobs[i,:] += np.sum(qobs_pdf, axis=0) 
            if self.time_variant_q is True:
                qobs = np.tile(qobs,(n_dstates,1))
                    
        if inspect is True and monitor is True: # joint probability of inspection and monitoring
            ins_monitor = np.concatenate((qobs.T*dobs[:,0],qobs.T*dobs[:,1]),axis=0).T
            monitor = np.zeros(ins_monitor.shape)
            monitor[:,0:n_qstates] = qobs
            ins = np.zeros(ins_monitor.shape)
            ins[:,[0, n_qstates]] = dobs
            no_obs = np.zeros(ins_monitor.shape)
            no_obs[:,0] =1
            self.O = {"ins_monitor": ins_monitor,"monitor": monitor, "ins": ins, "no_obs": no_obs}
        elif inspect is True:
            ins = dobs
            no_ins = np.zeros(ins.shape)
            no_ins[:,0] =1
            self.O = {"ins": ins, "no_ins": no_ins}
        elif monitor is True:            
            monitor = qobs
            no_monitor = np.zeros(monitor.shape)
            no_monitor[:,0] =1
            self.O = {"monitor": monitor, "no_monitor": no_monitor}
        else:
            self.O = {"no_obs": dobs}
                       
        return self.O
    

            
          
            
    


