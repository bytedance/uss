import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryConditionExtractor(nn.Module):
    def __init__(self, 
        at_model,
        condition_type,
    ):
        super(QueryConditionExtractor, self).__init__()

        assert condition_type in ["embedding", "at_soft"]

        self.at_model = at_model
        self.condition_type = condition_type


    def __call__(self, segments):

        if self.condition_type == "embedding":
            condition_type = "embedding"

        elif self.condition_type == "at_soft":
            condition_type = "clipwise_output"

        else:
            raise NotImplementedError

        with torch.no_grad():
            self.at_model.eval()
            conditions = self.at_model(segments)[condition_type]

        return conditions

