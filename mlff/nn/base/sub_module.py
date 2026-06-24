from abc import abstractmethod
from typing import Any

import flax.linen as nn


class BaseSubModule(nn.Module):
    @abstractmethod
    def __dict_repr__(self) -> dict[str, dict[str, Any]] | None:
        pass

    def reset_prop_keys(self, prop_keys):
        self.prop_keys.update(prop_keys)
