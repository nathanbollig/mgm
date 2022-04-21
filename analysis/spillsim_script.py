"""
Script for running instances of the spillsim post-analysis pipeline
"""
from mgm.analysis.spillsim_analysis_pipeline import spillsim_analysis_pipeline

########################################################################################################################
# Set analysis parameters
########################################################################################################################
sars2_params = {}
sars2_params['data_dir'] = "spillover_simulation_SARS2_v2"
sars2_params['SPILL_SEQ_DEFLINE'] = 'RaTG13|QHR63300|Bat|SARS_CoV_2'
sars2_params['SPILL_SEQ_PRETTY'] = 'RaTG13'
sars2_params['WITHHELD_SPECIES'] = 'SARS_CoV_2'
sars2_params['WITHHELD_SPECIES_PRETTY'] = 'SARS CoV 2'
sars2_params['THRESHOLD'] = None
sars2_params['keep_final_seq'] = None
sars2_params['LIM'] = None  # Scatter plot limit
########################################################################################################################

# sars2_params['THRESHOLD'] = 0.95
# sars2_params['keep_final_seq'] = True
# spillsim_analysis_pipeline(**sars2_params)
#
# sars2_params['THRESHOLD'] = 0.95
# sars2_params['keep_final_seq'] = False
# spillsim_analysis_pipeline(**sars2_params)

########################################################################################################################
# MERS
########################################################################################################################
mers_params = {}
mers_params['data_dir'] = "spillover_simulation_MERS_v2"
mers_params['WITHHELD_SPECIES'] = 'Middle_East_respiratory_syndrome_coronavirus'
mers_params['WITHHELD_SPECIES_PRETTY'] = 'MERS'
mers_params['THRESHOLD'] = None
mers_params['keep_final_seq'] = None
mers_params['LIM'] = None  # Scatter plot limit
########################################################################################################################

mers_params['THRESHOLD'] = 0.95
mers_params['keep_final_seq'] = True
spillsim_analysis_pipeline(**mers_params)

mers_params['THRESHOLD'] = 0.95
mers_params['keep_final_seq'] = False
spillsim_analysis_pipeline(**mers_params)

mers_params['THRESHOLD'] = 0.04
mers_params['keep_final_seq'] = True
spillsim_analysis_pipeline(**mers_params)

mers_params['THRESHOLD'] = 0.04
mers_params['keep_final_seq'] = False
spillsim_analysis_pipeline(**mers_params)