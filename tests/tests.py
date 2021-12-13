

from mgm.tests.test_744_experiment import test_744_experiment
from mgm.tests.test_big_bang import test_big_bang
from mgm.tests.test_greedy_mgm_hotflip import test_greedy_mgm_hotflip
from mgm.tests.test_greedy_mgm_lookahead import test_greedy_mgm_lookahead
from mgm.tests.test_kuzmin import test_kuzmin
from mgm.tests.test_sequences import test_sequences
from mgm.tests.test_costs import import test_costs


if __name__ == "__main__":
    test_744_experiment()
    test_big_bang()
    test_greedy_mgm_hotflip()
    test_greedy_mgm_lookahead()
    test_kuzmin()
    test_sequences()
    test_costs()