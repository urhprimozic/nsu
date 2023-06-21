from ProGEDescend import uniform_rational_grammar, fast_converging_rational_grammar, uniform_probabilities, progedescend
from DN4_2_podatky import generiraj_enacbo_1, generiraj_enacbo_2, generiraj_enacbo_3
import ProGED as pg
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def neodvisno_vzorcenje(data, num_equations):
    n = data.shape[1] - 1
    grammar = uniform_rational_grammar(n)
    ED = pg.EqDisco(data=data, 
                sample_size=num_equations,
                lhs_vars=["y"],
                rhs_vars=[f"x{i}" for i in range(1, n)],
                strategy_settings = {"max_repeat":1000},
                generator = grammar,
                verbosity=0)
    ED.generate_models()
    ED.fit_models()
    error = ED.models.retrieve_best_models(N=1)[0].get_error()
    return error

def adaptive(data, num_equations):
    n = data.shape[1] - 1
    probs = uniform_probabilities(4)
    _, _, _, error = progedescend(data, ['y'], [f"x{i}" for i in range(1, n)],probabilities=probs,sampling_size=10, verbose=0, max_iter=num_equations/10, learning_rate=0.005)
    return error

results = {}
for index, enacba in enumerate([generiraj_enacbo_1, generiraj_enacbo_2, generiraj_enacbo_3]):
    errors_neodvisno = []
    errors_adaptive = []
    ns_equations = [10, 20, 30, 40]#[5, 10, 15, 20, 25, 50, 60]#], 70, 80, 90]
    data = enacba()

    for n_eq in tqdm(ns_equations):
        errors_neodvisno.append(neodvisno_vzorcenje(data, n_eq))
        errors_adaptive.append(adaptive(data, n_eq))

    with open(f'errors_neodvisno_{index}.pkl', 'wb') as file:
        pickle.dump(errors_neodvisno, file)

    with open(f'errors_adaptive_{index}.pkl', 'wb') as file:
        pickle.dump(errors_adaptive, file)
    results[index] = (errors_neodvisno, errors_adaptive)

    # plot 
    plt.plot(ns_equations, errors_neodvisno, label='neodvisno vzorčenje')
    plt.plot(ns_equations, errors_adaptive, label='gradient descend, |S| = 10')
    plt.legend()
    plt.xlabel('Število vzorčenih enačb')
    plt.ylabel('Napaka')
    plt.title(f'Podatkovje #{index + 1}')
    plt.savefig(f'dataset_{index}_biggerS.png')
    plt.show()
