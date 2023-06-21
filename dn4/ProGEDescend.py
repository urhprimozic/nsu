# uses gradient descent to choose optimal parameters of a given grammar to return the most promising results
import ProGED as pg
import numpy as np
import pandas as pd
from ProGED.model_box import ModelBox
import tqdm
import logging

"""
Ideja
za fikne podatke in fiksno gramatiko (brez fiksnih verjetnosti) nam verjetnostni bektor p poda porazdelitev P(p) na funkcijah, ki jih tvori ta gramatika 
Kar bi želeli, je da je verjetnost za funkcije, ki so "slabe" (torej ki imajo vleiko napako), mala.
To lahko dosežemo, če minimaliziramo funkcijo
L(p) = sum(Error(f)*P(f)), kjer gremo z f po vseh funkcijah te gramatike, P(f) je P(p)(f) (vejrentost da gramatika z verjetnostnim vektorjem p, vrne f)
(ker je P(f) pain in the ass zračunat (glej P(izraz|gramatika)), namesto po funkcijah seštevamo po izpeljevalnih drevesih gramatike. f je predpis drevesa, P(f) pa verjetnost drevesa )


Funkcije L ne znamo poračunat, znamo pa poračunat Lhat = sum(Error(f)*P(f)), kjer seštevamo po dovolj veliki množici samplanih funkcij f iz gramatike z verjetnostjo P. 

Lhat miimaliziramo z grad descentom.. (lahko bi fa tud )

Postopek iterativno ponavljamo. Enačba kadidatka vsake iteracije (tista, po kateri zmerimo napako) je enačba dane iteracije z najmanjšo napako.


"""


def uniform_rational_grammar(n):
    """
    Returns grammar with uniform probabilities distribution
    """
    p = [1 / 3] * 9 + [1 / n] * n
    return rational_grammar(*p)


def uniform_probabilities(n):
    return [1 / 3] * 9 + [1 / n] * n


def fast_converging_rational_grammar(n):
    """
    Returns grammar with probabilities that force faster convergion
    """
    p = [
        0.05,
        0.05,
        0.9,
        0.05,
        0.05,
        0.9,
        0.9,
        0.05,
        0.05,
    ] + [1 / n] * n
    return rational_grammar(*p)


def rational_grammar(*probabilities):
    """
    Returns grammar with production rules
    E −> E + F | E − F | F
    F −> F ∗ T | F / T | T
    T −> V | (E) | s i n (E)
    v -> x_1 | ... | x_n
    with giver probabilities.

    n is calculated from the len of probabilities

    Example
    -----------
    >>>rational_grammar(0.4, 0.3, 0.3,
                        0.4, 0.3, 0.3,
                        0.4, 0.3, 0.3,
                        0.25, 0.25, 0.25, 0.25
    )
    """
    n = len(probabilities) - 9
    grammar = f"E -> E '+' F | E '-' F | F |\n"
    grammar += f"F -> F '*' T | F '/' T | T |\n"
    grammar += f"T -> V | '(' E ')' | 'sin' '(' E ')' |\n"
    grammar += f"V -> "
    for i in range(1, n + 1):
        grammar += f" 'x{i}' "
        if i < n:
            grammar += " | "
    # add probabilities:

    probabilistic_grammar = ""
    for s, p in zip(grammar.split("|"), probabilities):
        probabilistic_grammar += s + f"[{p}] |"
    ans = ""
    for s in probabilistic_grammar.split("\n"):
        ans += s[:-1] + "\n"
    ans = ans[:-1]

    return pg.GeneratorGrammar(ans)


def gradL(probabilities, models: ModelBox, grammar):
    """
    Returns gradiendLhat (probabilities)
    """
    # get productions - in the same order as probabilities
    productions = grammar.grammar.productions()
    gradient = []
    n = len(productions)
    # verjetnosti se morajo seštevati v 1. Zato velja p3 = 1-p2-p1, ..., pn = 1 - p(9+1) - ... - p(n-1)
    # L je funckija p1, p2, p4, p5, p7, p8, p10, ..., p(n-1).
    # d(tree) / dpi = d(p1^ g1 ... pn^gn) /d(pi)  = gi*p1^ g1 ... pn^gn/pi - p1^ g1 ... gj pn^gn/pi/(pj^gj), kjker je
    # j najmanjši index izmed (3, 6, 9, n), ki je VEČJI od i
    dependend_probs = [2, 5, 7, n - 1]

    def get_upper_index(i):
        for j in dependend_probs:
            if i < j:
                return j

    for index, production in enumerate(productions):
        if index in dependend_probs:
            gradient.append(0)
            continue

        dL_over_dp = 0
        for model in models:
            # get all the codes of all the trees:
            for code in list(model.info["trees"].keys()):
                # get all the productions used in the tree given by code
                productions_in_tree = grammar.code_to_expression(code)[1]

                # get the exponent of the production in P(tree)
                # P(tree) = p1^(g1) ... pn ^(gn) .
                #  power = gi, where pi = production
                # dP(tree) / dproduction = dP(tree) / dpi = gi P(tree) / pi = power * P(tree)/P(production)
                power = productions_in_tree.count(production)
                power_upper_index = productions_in_tree.count(
                    productions[get_upper_index(index)]
                )
                dL_over_dp += model.get_error()*(
                    power * model.p / production.prob()
                    - power_upper_index
                    * model.p
                    / productions[get_upper_index(index)].prob()
                )
        # multipy with error

        gradient.append(dL_over_dp)

    return gradient


def progedescend(
    data,
    lhs_vars,
    rhs_vars,
    probabilities,
    sampling_size=100,
    learning_rate=0.001,
    eps=None,
    max_iter=None,
    verbose=0,
    file_logging=None,
):
    """
    Iteratively generates equations using grammar and adapt probabilities of production rules in grammar to lower the porbability of sampling bad equations

    Parameters
    -----------
    X, y - data for which we want equation y=y(x)
    grammar - pcfg used for generating equations
    sampling_size - number of equations generated each iteration. Bigger number means Lhat is  probably more close to L.
    learning_rate - learnning rate of gradient descent

    Returns
    ------
    Tuple best_equation, grammar, best_model, min_error
    """
    if eps is None and max_iter is None:
        max_iter = 100
        eps = 0
    if eps is None:
        eps = 0
    if max_iter is None:
        max_iter = float("inf")

    # training loop
    iteration = 0
    error = float("inf")
    min_error = error
    best_equation = None
    best_model = None
    n = len(probabilities)
    while error > eps and iteration <= max_iter:
        iteration += 1
        print(f"Starting iteration {iteration}..", end="    ")
        # build grammar with new probabilities
        grammar = rational_grammar(*probabilities)

        # generate S
        ED = pg.EqDisco(
            data=data,
            sample_size=sampling_size,
            lhs_vars=lhs_vars,
            rhs_vars=rhs_vars,
            strategy_settings={"max_repeat": 100},
            generator=grammar,
            verbosity=0,
        )
        ED.generate_models()
        ED.fit_models()
        models = ED.models

        print("Gradient descend..", end="  ")
        # update probabilities (grad descend)
        # new probabilities = probabilities - learning_rate * GRAD(L)(probabilities)
        dependend_probs = [2, 5, 7, n - 1]
        for index, p in enumerate(probabilities):
            if index in dependend_probs:
                continue
            # p = p - learning_rate * dL/dp (probabilities)
            gradient = gradL(probabilities, models, grammar)
            for i in range(len(probabilities)):
                probabilities[i] -= learning_rate * gradient[i]
        # fix dependable probabilities
        probabilities[2] = 1 - probabilities[1] - probabilities[0]
        probabilities[5] = 1 - probabilities[4] - probabilities[3]
        probabilities[8] = 1 - probabilities[7] - probabilities[6]
        probabilities[n - 1] = 1 - sum(probabilities[9:-1])

        print(f"avg={np.mean(gradient)} grad={gradient}")
        # update error
        error = ED.models.retrieve_best_models(N=1)[0].get_error()
        if error < min_error:
            min_error = error
            best_equation = ED.models.retrieve_best_models(N=1)[0].expr[0]
            best_model = ED.models.retrieve_best_models(N=1)[0]

        print(f"error: {error}. Min: {min_error}")

        # logging
        if file_logging is not None:
            with open(file_logging, "a") as f:
                log_p = f"Iteration {iteration}:"
                for p in probabilities:
                    log_p += str(p) + ",  "
                log_p += f"\nError: {error}, Min Error : {min_error}\n"
                f.write(log_p)

        # check if any of the probabilities is out of [0,1] interval
        for p in probabilities:
            if p < 0 or p > 1:
                print(
                    f"Out of bounds probability reached (p={p}). Stopping the procedure. Try decreasing learning rate"
                )
                return best_equation, grammar, best_model, min_error
    return best_equation, grammar, best_model, min_error
