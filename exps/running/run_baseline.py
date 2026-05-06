from recbole.quick_start import run_recbole
from recbole.model.context_aware_recommender import DeepFM, DCNV2, DeepFM_MoE, WideDeep

run_recbole(model=DeepFM, config_file_list=["avazu.yaml"])
