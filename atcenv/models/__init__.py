from models.action_mask_model import FlightActionMaskModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("flight_model_mask", FlightActionMaskModel)

