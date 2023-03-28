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

    def __call__(self, segments):

        with torch.no_grad():
            self.at_model.eval()
            conditions = self.at_model(segments)[self.condition_type]

        return conditions

