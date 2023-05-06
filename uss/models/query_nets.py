from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from uss.config import panns_paths_dict
from uss.models.base import init_layer
from uss.utils import get_path, load_pretrained_panns


def initialize_query_net(configs):
    r"""Initialize query net.

    Args:
        configs (Dict)

    Returns:
        model (nn.Module)
    """

    model_type = configs["query_net"]["model_type"]
    bottleneck_type = configs["query_net"]["bottleneck_type"]
    # base_checkpoint_path = configs["query_net"]["base_checkpoint_path"]
    base_checkpoint_type = configs["query_net"]["base_checkpoint_type"]
    freeze_base = configs["query_net"]["freeze_base"]
    outputs_num = configs["query_net"]["outputs_num"]

    base_checkpoint_path = get_path(panns_paths_dict[base_checkpoint_type])

    if model_type == "Cnn14_Wrapper":

        model = Cnn14_Wrapper(
            bottleneck_type=bottleneck_type,
            base_checkpoint_path=base_checkpoint_path,
            freeze_base=freeze_base,
        )

    elif model_type == "AdaptiveCnn14_Wrapper":

        model = AdaptiveCnn14_Wrapper(
            bottleneck_type=bottleneck_type,
            base_checkpoint_path=base_checkpoint_path,
            freeze_base=freeze_base,
            freeze_adaptor=configs["query_net"]["freeze_adaptor"],
            outputs_num=outputs_num,
        )

    elif model_type == "YourOwn_QueryNet":
        model = YourOwn_QueryNet(outputs_num=outputs_num)

    else:
        raise NotImplementedError

    return model


def get_panns_bottleneck_type(bottleneck_type: str) -> str:
    r"""Get PANNs bottleneck name.

    Args:
        bottleneck_type (str)

    Returns:
        panns_bottleneck_type (str)
    """

    if bottleneck_type == "at_soft":
        panns_bottleneck_type = "clipwise_output"

    else:
        panns_bottleneck_type = bottleneck_type

    return panns_bottleneck_type


class Cnn14_Wrapper(nn.Module):
    def __init__(self,
                 bottleneck_type: str,
                 base_checkpoint_path: str,
                 freeze_base: bool,
                 ) -> None:
        r"""Query Net based on Cnn14 of PANNs. There are no extra learnable
        parameters.

        Args:
            bottleneck_type (str), "at_soft" | "embedding"
            base_checkpoint_path (str), Cnn14 checkpoint path
            freeze_base (bool), whether to freeze the parameters of the Cnn14
        """

        super(Cnn14_Wrapper, self).__init__()

        self.panns_bottleneck_type = get_panns_bottleneck_type(bottleneck_type)

        self.base = load_pretrained_panns(
            model_type="Cnn14",
            checkpoint_path=base_checkpoint_path,
            freeze=freeze_base,
        )

        self.freeze_base = freeze_base

    def forward_base(self, source: torch.Tensor) -> torch.Tensor:
        r"""Forward a source into a the base part of the query net.

        Args:
            source (torch.Tensor), (batch_size, audio_samples)

        Returns:
            bottleneck (torch.Tensor), (bottleneck_dim,)
        """

        if self.freeze_base:
            self.base.eval()
            with torch.no_grad():
                base_output_dict = self.base(source)
        else:
            self.base.train()
            base_output_dict = self.base(source)

        bottleneck = base_output_dict[self.panns_bottleneck_type]

        return bottleneck

    def forward_adaptor(self, bottleneck: torch.Tensor) -> torch.Tensor:
        r"""Forward a bottleneck into a the adaptor part of the query net.

        Args:
            bottleneck (torch.Tensor), (bottleneck_dim,)

        Returns:
            output (torch.Tensor), (output_dim,)
        """

        output = bottleneck

        return output

    def forward(self, source: torch.Tensor) -> Dict:
        r"""Forward a source into a query net.

        Args:
            source (torch.Tensor), (batch_size, audio_samples)

        Returns:
            output_dict (Dict), {
                "bottleneck": (bottleneck_dim,)
                "output": (output_dim,)
            }
        """

        bottleneck = self.forward_base(source=source)

        output = self.forward_adaptor(bottleneck=bottleneck)

        output_dict = {
            "bottleneck": bottleneck,
            "output": output,
        }

        return output_dict


class AdaptiveCnn14_Wrapper(nn.Module):
    def __init__(self,
                 bottleneck_type: str,
                 base_checkpoint_path: str,
                 freeze_base: bool,
                 freeze_adaptor: bool,
                 outputs_num: int,
                 ) -> None:
        r"""Query Net based on Cnn14 of PANNs. There are no extra learnable
        parameters.

        Args:
            bottleneck_type (str), "at_soft" | "embedding"
            base_checkpoint_path (str), Cnn14 checkpoint path
            freeze_base (bool), whether to freeze the parameters of the Cnn14
            freeze_adaptor (bool), whether to freeze the parameters of the
                adaptor
            outputs_num (int), output dimension
        """

        super(AdaptiveCnn14_Wrapper, self).__init__()

        self.freeze_base = freeze_base

        self.panns_bottleneck_type = get_panns_bottleneck_type(bottleneck_type)

        self.base = load_pretrained_panns(
            model_type="Cnn14",
            checkpoint_path=base_checkpoint_path,
            freeze=freeze_base,
        )

        bottleneck_units = self._get_bottleneck_units(
            self.panns_bottleneck_type)

        self.fc1 = nn.Linear(bottleneck_units, 2048)
        self.fc2 = nn.Linear(2048, outputs_num)

        if freeze_adaptor:
            for param in self.fc1.parameters():
                param.requires_grad = False

            for param in self.fc2.parameters():
                param.requires_grad = False

        self.init_weights()

    def _get_bottleneck_units(self, panns_bottleneck_type) -> int:

        if panns_bottleneck_type == "embedding":
            bottleneck_hid_units = self.base.fc_audioset.in_features

        elif panns_bottleneck_type == "clipwise_output":
            bottleneck_hid_units = self.base.fc_audioset.out_features

        else:
            raise NotImplementedError

        return bottleneck_hid_units

    def init_weights(self):
        r"""Initialize weights."""
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward_base(self, source: torch.Tensor) -> torch.Tensor:
        r"""Forward a source into a the base part of the query net.

        Args:
            source (torch.Tensor), (batch_size, audio_samples)

        Returns:
            bottleneck (torch.Tensor), (bottleneck_dim,)
        """

        if self.freeze_base:
            self.base.eval()
            with torch.no_grad():
                base_output_dict = self.base(source)
        else:
            self.base.train()
            base_output_dict = self.base(source)

        bottleneck = base_output_dict[self.panns_bottleneck_type]

        return bottleneck

    def forward_adaptor(self, bottleneck: torch.Tensor) -> torch.Tensor:
        r"""Forward a bottleneck into a the adaptor part of the query net.

        Args:
            bottleneck (torch.Tensor), (bottleneck_dim,)

        Returns:
            output (torch.Tensor), (output_dim,)
        """

        x = F.leaky_relu(self.fc1(bottleneck), negative_slope=0.01)
        x = F.dropout(x, p=0.5, training=self.training, inplace=True)

        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        output = F.dropout(x, p=0.5, training=self.training, inplace=True)

        return output

    def forward(self, source: torch.Tensor) -> Dict:
        r"""Forward a source into a query net.

        Args:
            source (torch.Tensor), (batch_size, audio_samples)

        Returns:
            output_dict (Dict), {
                "bottleneck": (bottleneck_dim,)
                "output": (output_dim,)
            }
        """

        bottleneck = self.forward_base(source=source)

        output = self.forward_adaptor(bottleneck=bottleneck)

        output_dict = {
            "bottleneck": bottleneck,
            "output": output,
        }

        return output_dict


class YourOwn_QueryNet(nn.Module):
    def __init__(self, outputs_num: int) -> None:
        r"""User defined query net."""

        super(YourOwn_QueryNet, self).__init__()

        self.fc1 = nn.Linear(1, outputs_num)

    def forward(self, source: torch.Tensor) -> Dict:
        r"""Forward a source into a query net.

        Args:
            source (torch.Tensor), (batch_size, audio_samples)

        Returns:
            output_dict (Dict), {
                "bottleneck": (bottleneck_dim,)
                "output": (output_dim,)
            }
        """

        x = torch.mean(source, dim=-1, keepdim=True)
        bottleneck = self.fc1(x)

        output_dict = {
            "bottleneck": bottleneck,
            "output": bottleneck,
        }

        return output_dict
