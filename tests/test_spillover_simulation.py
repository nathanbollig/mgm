from mgm.common.utils import set_data_directory
from mgm.models.NN import make_CNN
from mgm.pipelines.spillover_simulation import spillover_experiment

set_data_directory("test_spillover_simulation1")
spillover_experiment(species_to_withhold = 'SARS_CoV_2', validate_model=True, model_initializer = make_CNN, desired_precision=0.9)