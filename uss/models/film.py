from typing import Dict, List

import torch
import torch.nn as nn

from uss.models.base import init_layer


def get_film_meta(module: nn.Module) -> Dict:
    r"""Get FiLM meta dict of a module.

    Args:
        module (nn.Module), the module to extract meta dict

    Returns:
        film_meta (Dict), FiLM meta dict
    """

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            film_meta['beta2'] = module.bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    # Pre-order traversal of modules
    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta

    return film_meta


class FiLM(nn.Module):
    def __init__(
        self,
        film_meta: Dict,
        condition_size: int,
    ) -> None:
        r"""Create FiLM modules from film meta dict.

        Args:
            film_meta (Dict), e.g.,
                {'encoder_block1': {'conv_block1': {'beta1': 32, 'beta2': 32}},
                 ...}
            condition_size: int

        Returns:
            None
        """

        super(FiLM, self).__init__()

        self.condition_size = condition_size

        self.modules, _ = self._create_film_modules(
            film_meta=film_meta,
            prefix_names=[],
        )

    def _create_film_modules(
        self,
        film_meta: Dict,
        prefix_names: List[str],
    ):
        r"""Create FiLM modules.

        Args:
            film_meta (Dict), e.g.,
                {"encoder_block1": {"conv_block1": {"beta1": 32, "beta2": 32}},
                 ...}
            prefix_names (str), only used to get correct module name, e.g.,
                ["encoder_block1", "conv_block1"]
        """

        modules = {}

        # Pre-order traversal of modules
        for module_name, value in film_meta.items():

            if isinstance(value, dict):

                prefix_names.append(module_name)

                modules[module_name], _ = self._create_film_modules(
                    film_meta=value,
                    prefix_names=prefix_names,
                )

            elif isinstance(value, int):

                prefix_names.append(module_name)
                unique_module_name = '->'.join(prefix_names)

                modules[module_name] = self._add_film_layer_to_module(
                    num_features=value,
                    unique_module_name=unique_module_name,
                )

            prefix_names.pop()

        return modules, prefix_names

    def _add_film_layer_to_module(
        self,
        num_features: int,
        unique_module_name: str,
    ) -> nn.Module:
        r"""Add a FiLM layer."""

        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)

        return layer

    def _calculate_film_data(self, conditions, modules):

        film_data = {}

        # Pre-order traversal of modules
        for module_name, module in modules.items():

            if isinstance(module, dict):
                film_data[module_name] = self._calculate_film_data(
                    conditions, module)

            elif isinstance(module, nn.Module):
                film_data[module_name] = module(conditions)[:, :, None, None]

        return film_data

    def forward(self, conditions: torch.Tensor) -> Dict:
        r"""Forward conditions to all FiLM layers to get FiLM data.

        Args:
            conditions (torch.Tensor): query net outputs,
                (batch_size, condition_dim)

        Returns:
            film_dict (Dict): e.g., {
                "encoder_block1": {
                    "conv_block1": {
                        "beta1": (16, 32, 1, 1),
                        "beta2": (16, 32, 1, 1),
                    },
                    ...,
                },
                ...,
            }
        """

        film_dict = self._calculate_film_data(
            conditions=conditions,
            modules=self.modules,
        )

        return film_dict
