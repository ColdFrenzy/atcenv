
from models.action_mask_model import FlightActionMaskModel, FlightActionMaskRNNModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("flight_model_mask", FlightActionMaskModel)
ModelCatalog.register_custom_model(
    "flight_rnn_model_mask", FlightActionMaskRNNModel)
