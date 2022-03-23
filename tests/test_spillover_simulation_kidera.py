from mgm.common.utils import set_data_directory
from mgm.models.NN import make_CNN
from mgm.pipelines.spillover_simulation import spillover_experiment
import datetime

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    set_data_directory("test_spillover_simulation_kidera")
    spillover_experiment(species_to_withhold = 'SARS_CoV_2', validate_model=True, model_initializer = make_CNN, confidence_threshold=0.95, representation='kidera',fixed_iterations=3)
    time_end = datetime.datetime.now()
    time_seconds = (time_end - start_time).total_seconds()
    print("Experiment took %.1f seconds" % (time_seconds,))