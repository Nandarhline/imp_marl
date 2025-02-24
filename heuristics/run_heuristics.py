from heuristics.heuristics_intervals import Heuristics
import timeit

if __name__ == '__main__':

    #### Search
    search = True
    n_comp = 5
    k_comp = 4
    discount_reward = 0.95
    eval_size = 2000

    #### Evaluation
    insp_int = 10
    insp_comp = 5


    h1 = Heuristics(n_comp,
                     # Number of structure
                     discount_reward,
                     # float [0,1] importance of
                     # short-time reward vs long-time reward
                     k_comp)

    if search:
        starting_time = timeit.default_timer()
        h1.search(eval_size)
        print("Time (s):", timeit.default_timer() - starting_time)
    else:
        h1.eval(eval_size, insp_int, insp_comp)
