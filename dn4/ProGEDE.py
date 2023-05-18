# uses gradient descent to choose optimal parameters of a given grammar to return the most promising results
import ProGED as pq


def fit_models()


class EqDiscoExt(pq.EqDisco):
    def __init__(
        self,
        task=None,
        data=None,
        rhs_vars=None,
        lhs_vars=None,
        constant_symbol="C",
        task_type="algebraic",
        system_size=1,
        generator="grammar",
        generator_template_name="universal",
        variable_probabilities=None,
        generator_settings=...,
        strategy="monte-carlo",
        strategy_settings=...,
        sample_size=10,
        max_attempts=1,
        repeat_limit=100,
        depth_limit=1000,
        estimation_settings=...,
        success_threshold=1e-8,
        verbosity=0,
        # parameters of grad descent
    ):
        # pokliče vse kar ve od prej
        super().__init__(
            task,
            data,
            rhs_vars,
            lhs_vars,
            constant_symbol,
            task_type,
            system_size,
            generator,
            generator_template_name,
            variable_probabilities,
            generator_settings,
            strategy,
            strategy_settings,
            sample_size,
            max_attempts,
            repeat_limit,
            depth_limit,
            estimation_settings,
            success_threshold,
            verbosity,
        )


if __name__ == "__main__":
    pass
