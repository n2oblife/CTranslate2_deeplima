import numpy as np
import operator
from typing import Iterator, Union, Iterable
from collections import OrderedDict, abc as container_abcs
from itertools import chain
from ctranslate2.specs import common_spec, model_spec

class PairwiseSpec(model_spec.LayerSpec):
    """Might be considered the hidden state of the Deep Biaffine cell"""
    # TODO check if useful
    def __init__(self) -> None:
        pass
    
class DeepBiaffineSpec(model_spec.LayerSpec):
    """Adapted from trankit's implementation"""
    def __init__(self, 
                 activation : common_spec.Activation = common_spec.Activation.RELU
                 ) -> None:
        # TODO check if one activation is sufficient for both the Linear layers and the type
        # TODO change the building of this class to be dynamic
        self.activation = np.dtype("int8").type(activation)
        # self.ffn1 = ModuleListSpec([common_spec.LinearSpec])
        # self.ffn2 = ModuleListSpec([common_spec.LinearSpec])
        self.ffn1 = SequentialSpec(spec=common_spec.LinearSpec)
        self.ffn2 = SequentialSpec(spec=common_spec.LinearSpec)
        self.pairwise_weight = None

class PosdepDecoderSpec(model_spec.LayerSpec):
    # TODO be sure that during conversion the forward keeps the same
    def __init__(self) -> None:
        self.upos_embeddings = common_spec.EmbeddingsSpec()
        # postagging
        self.upos_ffn = common_spec.LinearSpec()
        self.xpos_ffn = common_spec.LinearSpec()
        self.feats_ffn = common_spec.LinearSpec()
        self.down_project = common_spec.LinearSpec()
        # dep parsing
        self.unlabeled = DeepBiaffineSpec()
        self.deprel = DeepBiaffineSpec()
    
    @classmethod
    def from_config(cls):
        return cls()

class PosdepDecoderConfig(model_spec.ModelConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class SequentialSpec(model_spec.LayerSpec):
    """Builds layers with number to differenciate the multiple layers of same type in same depth.
    This class is callable to ease getting the parameters.
    Only used in posdep so far might be replaced by a modulelistspec
    Args:
        model_spec (_type_): _description_
    """
    def __init__(self, name:str='', 
                 spec:model_spec.LayerSpec|list[model_spec.LayerSpec] = common_spec.LinearSpec, 
                 num=0) -> None:
        super().__init__()
        if isinstance(spec, list):
            for i in range(num+1):
                for sp in spec:
                    self.__dict__[name+str(i)] = sp()
        else :
            for i in range(num+1):
                self.__dict__[name+str(i)] = spec()

    def __call__(self, num=0, name='') -> model_spec.LayerSpec:
        """Instead of calling the param spec.ffn.0 -> spec.ffn(0)

        Args:
            num (int, optional): _description_. Defaults to 0.
            name (str, optional): _description_. Defaults to ''.

        Raises:
            KeyError: _description_

        Returns:
            _type_: _description_
        """
        if num > len(self.__dict__) or num < 0:
            raise KeyError(f"This Layer has a max depth of {len(self.__dict__)} and must be positive")
        return getattr(self, name+str(num))

# Copied from torch.nn.modules.module, required for a cusom __repr__ for ModuleListSpec
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

class ModuleListSpec(model_spec.LayerSpec):
    """DONE"""

    def __init__(self, modules:[Iterable[model_spec.LayerSpec]] = None) -> None:
        super().__init__()
        self._modules: dict[str, model_spec.LayerSpec] = {}
        if modules is not None:
            self._modules += modules
    
    def _get_abs_string_index(self, idx) -> str:
        """Get the absolute indedx for the list of modules

        Args:
            idx (Union[int, slice]): 

        Returns:
            str: 
        """
        idx = operator.index(idx)
        if not(-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        return str(idx)
    
    def __getitem__(self, idx:Union[int, slice]) -> Union[model_spec.LayerSpec,'ModuleListSpec']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]
    
    def __setitem__(self, idx:int, module:model_spec.LayerSpec):
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)
    
    def __delitem__(self, idx:Union[int, slice]) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # to preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(
            zip(str_indices, self._modules.values())
            ))
    
    def add_module(self, key, mod):
        self._modules[key] = mod

    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterable[model_spec.LayerSpec]:
        return iter(self._modules.values())
    
    def __iadd__(self, modules: Iterable[model_spec.LayerSpec]) -> 'ModuleListSpec':
        return self.extend(modules)

    def __add__(self, others:Iterable[model_spec.LayerSpec]) -> 'ModuleListSpec':
        combined = ModuleListSpec()
        for i, module in enumerate(chain(self, others)):
            combined.add_module(str(i), module)
        return combined
    
    def __repr__(self):
        """A custom repr for ModuleListSpec that compresses repeated module representations"""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + '()'

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + '('
        for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: model_spec.LayerSpec) -> None:
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module


    def append(self, module: model_spec.LayerSpec) -> 'ModuleListSpec':
        r"""Appends a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self


    def pop(self, key: Union[int, slice]) -> model_spec.LayerSpec:
        v = self[key]
        del self[key]
        return v

    def extend(self, modules: Iterable[model_spec.LayerSpec]) -> 'ModuleListSpec':
        r"""Appends modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self