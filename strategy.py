from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic
from strategy_mixin import StrategyMixin

# Custom strategy combining FedXgbBagging with the logging/saving mixin.
class CustomFedXgbBagging(StrategyMixin, FedXgbBagging):
    pass

# Custom strategy combining FedXgbCyclic with the logging/saving mixin.
class CustomFedXgbCyclic(StrategyMixin, FedXgbCyclic):
    pass