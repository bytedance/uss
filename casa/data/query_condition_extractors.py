import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryConditionExtractor(nn.Module):
    def __init__(self, 
        at_model,
        condition_type,
    ):
        super(QueryConditionExtractor, self).__init__()

        self.at_model = at_model
        self.condition_type = condition_type

    def __call__(self, segment):

        with torch.no_grad():
            self.at_model.eval()
            condition = self.at_model(segment)[self.condition_type]

        return condition

