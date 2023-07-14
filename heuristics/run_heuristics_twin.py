from heuristics_intervals_twin import Heuristics
import timeit

if __name__ == '__main__':

    n_comp = 3
    k_comp = 3
    discount_reward = 0.95
    eval_size = 100
    
    #### Search or evaluate inspection intervals and no of components
    search_ins = True
    eval_ins = True
    # Eval values
    insp_int = 22
    insp_comp = 10
    
    #### Search or evaluate install-sensor intervals and no of components
    search_sens = True
    eval_sens = True
    # Eval values
    sens_int = 22
    sens_comp = 10
    
    h1 = Heuristics(n_comp,
                     # Number of structure
                     discount_reward,
                     # float [0,1] importance of
                     # short-time reward vs long-time reward
                     k_comp)
    
    # Equadistant inspection heuristics
    if search_ins:
        starting_time = timeit.default_timer()
        h1.search_insinterv(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)
    if eval_ins:
        h1.eval_insinterv(eval_size, insp_int, insp_comp)
    
    # Equadistant install-sensor heuristics
    if search_sens:
        starting_time = timeit.default_timer()
        h1.search_sensinterv(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)
    if eval_sens:
        h1.eval_sensinterv(eval_size, sens_int, sens_comp)