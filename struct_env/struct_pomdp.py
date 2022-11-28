import numpy as np
import math

class Pomdp:

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
        Y = np.random.lognormal(Y_mean, Y_std, nsamples)  
        
        self.dd = np.zeros((self.T+1, nsamples))
        self.dd[0,:] = d0
        
        if self.time_variant_q is True:
            epsilonq = np.random.normal(0, self.epsilonq_std, nsamples)
            self.qq = np.zeros((self.T+1, nsamples))
            self.qq[0,:] = q0
        
        for t in range(self.T):
            dt = (((2-m)/2)*C*(Y*math.pi**0.5*S)**m*self.ncycles + d0**((2-m)/2))**(2/(2-m))
            dt[(dt<d0)| (dt>self.dcrit)] = self.dcrit+0.1
            self.dd[t+1,:] = dt
            d0 = dt
            if self.time_variant_q is True:
                qt = q0+epsilonq
                while (qt<0).sum() > 0:
                    qt[qt<0] = q0[qt<0]+np.random.normal(0, self.epsilonq_std , (qt<0).sum())
                q0 = qt
                self.qq[t+1,:] = qt
                S = qt*math.gamma(1+1/0.8)
        if self.time_variant_q is True:
            return self.dd, self.qq
        else: 
            return self.dd
        
    def transition_models(self, n_dstates=30, n_qstates = 20):
        if self.time_variant_q is True:
            d_interv = 1e-20
            d_interv = np.append(d_interv, np.linspace(self.d0_mean, self.dcrit, n_dstates-1))
            d_interv = np.append(d_interv, 1e20)
            q_interv = np.linspace(0, np.max(self.qq)+1, n_qstates+1)
            det_rates = self.T+1
            nsamples = self.dd.shape[-1]   
        
            H, _, _ = np.histogram2d(self.dd[0,:], self.qq[0,:], [d_interv, q_interv])
            self.b0 = (H/nsamples).reshape(-1) # d is the outer loop
            
            self.T0 = np.zeros((det_rates, n_dstates*n_qstates, n_dstates*n_qstates))
            for i in range(det_rates-1):
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                Q = self.qq[i,:] # Samples q at det. rate i, 
                Q_ = self.qq[i+1,:] # Samples q at det. rate i+1
                for j in range(n_dstates):
                    countd = (D>d_interv[j]) &  (D<d_interv[j+1])
                    for k in range(n_qstates):
                        countq =(Q>q_interv[k]) &  (Q<q_interv[k+1])
                        Dnext = D_[countd & countq]
                        Qnext = Q_[countd & countq]
                        if (countd & countq).sum() < 1:
                            self.T0[i,j*n_qstates+k,j*n_qstates+k]=1
                        else:
                            H, _, _ = np.histogram2d(Dnext, Qnext , [d_interv, q_interv]) 
                            self.T0[i,j*n_qstates+k,:] = (H/(countd & countq).sum()).reshape(-1)
            self.T0[-1,] = self.T0[-2,]

            self.Tr = np.zeros((det_rates, n_dstates*n_qstates, n_dstates*n_qstates))
            self.Tr = np.tile(self.b0,(det_rates, n_dstates*n_qstates,1))  
            
        else:
            d_interv = 1e-20
            d_interv = np.append(d_interv, np.linspace(self.d0_mean, self.dcrit, n_dstates-1))
            d_interv = np.append(d_interv, 1e20)
            det_rates = self.T+1
            nsamples = self.dd.shape[-1]   
        
            H, _ = np.histogram(self.dd[0,:], d_interv)
            self.b0 = H/nsamples
            
            self.T0 = np.zeros((det_rates, n_dstates, n_dstates))
            for i in range(det_rates-1):
                D = self.dd[i,:] # Samples a at det. rate i
                D_ = self.dd[i+1,:] # Samples a at det. rate i+1
                for j in range(n_dstates):
                    countd = (D>d_interv[j]) &  (D<d_interv[j+1])
                    Dnext = D_[countd]
                    if countd.sum() < 1:
                        self.T0[i,j,j]=1
                    else:
                        H, _ = np.histogram(Dnext, d_interv) 
                        self.T0[i,j,:] = H/countd.sum()
            self.T0[-1,] = self.T0[-2,]

            self.Tr = np.zeros((det_rates, n_dstates, n_dstates))
            self.Tr = np.tile(self.b0,(det_rates, n_dstates,1))                
        return
    
    
    def observation_models(self, n_dstates=30, n_qstates = 20):
        pass
            
          
            
    


