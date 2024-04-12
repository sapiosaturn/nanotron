from typing import Any, Callable, Dict, Iterable, Tuple, Union

import torch

from nanotron.optim.inherit_from_other_optimizer import InheritFromOtherOptimizer
from nanotron.scaling.parametrization import LearningRateForParametrizator


class NamedOptimizer(InheritFromOtherOptimizer):
    """Mimics somewhat the torch optimizer API"""

    def __init__(
        self,
        named_params_or_groups: Iterable[Union[Tuple[str, torch.Tensor], Dict[str, Any]]],
        learning_rate_mapper: LearningRateForParametrizator,
        optimizer_builder: Callable[[Iterable[Dict[str, Any]]], torch.optim.Optimizer],
    ):
        named_param_groups = list(named_params_or_groups)
        if len(named_param_groups) == 0 or not isinstance(named_param_groups[0], dict):
            named_param_groups = [{"named_params": named_param_groups}]

        id_to_name = {}
        params = []
        for named_param_group in named_param_groups:
            assert "named_params" in named_param_group
            # Don't need to check that param_groups are overlapping since the optimizer will do it for me.
            #  https://github.com/pytorch/pytorch/blob/88b3810c94b45f5982df616e2bc4c471d173f491/torch/optim/optimizer.py#L473
            id_to_name.update(
                {id(param): name for name, param in named_param_group["named_params"] if id(param) not in id_to_name}
            )
            params.append(
                {
                    **{k: v for k, v in named_param_group.items() if k != "named_params"},
                    "params": [param for _, param in named_param_group["named_params"]],
                }
            )

        name_to_id = {v: k for k, v in id_to_name.items()}
        assert len(id_to_name) == len(name_to_id)

        # NOTE: mapping learning rate based on the paramitrization
        params_with_lr = []
        for param_group in params:
            for p in param_group["params"]:
                name = id_to_name[id(p)]
                lr = learning_rate_mapper.get_lr(name, p)
                assert lr is not None, f"Learning rate for {name} is None"
                other_hyperparameters = {k: v for k, v in param_group.items() if k != "params"}
                params_with_lr.append({"params": [p], "lr": lr, **other_hyperparameters})

        # Sanity check
        for param_group in params_with_lr:
            _params = param_group["params"]
            for param in _params:
                # https://github.com/pytorch/pytorch/issues/100701
                assert param.numel() > 0

        super().__init__(optimizer=optimizer_builder(params_with_lr), id_to_name=id_to_name)

    def state_dict(self) -> dict:
        optim_state_dict = super().state_dict()

        assert "names" not in optim_state_dict

        state_id_to_name = {id(state): self.id_to_name[id(param)] for param, state in self.optimizer.state.items()}
        optim_state_dict["names"] = {
            index: state_id_to_name[id(state)] for index, state in optim_state_dict["state"].items()
        }
        return optim_state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        # TODO @thomasw21: Make a more robust test
        assert set(self.id_to_name.values()) == set(
            state_dict["names"].values()
        ), f"Elements don't match:\n - Elements in `self.id_to_name` that aren't in the other one: {set(self.id_to_name.values()) - set(state_dict['names'].values())}\n - Elements in `state_dict[\"names\"]` that aren't in the other one: {set(state_dict['names'].values()) - set(self.id_to_name.values())}"

        return super().load_state_dict(state_dict)
