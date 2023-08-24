# import logging
import numpy as np
import operator
from dataclasses import field
from ctranslate2.converters import utils
from typing import Iterator, Union, Iterable
from collections import OrderedDict, abc as container_abcs
from itertools import chain, islice

from trankit.adapter_transformers.adapter_utils import AdapterType
import transformers.adapters.configuration  as cfg
from trankit.adapter_transformers.configuration_xlm_roberta import XLMRobertaConfig
from ctranslate2.specs import common_spec, model_spec, transformer_spec, attention_spec, posdep_spec

# logger = logging.getLogger(__name__)
# TODO add the management of the adapter fusion
# TODO review the architecture of he project to gather the dicts and functions

# --------------------------------------------------------
# dicts and functions
ACTIVATION = {'relu': common_spec.Activation.RELU,
              'gelu': common_spec.Activation.GELU,
              'fast-gelu': common_spec.Activation.GELUTanh,
              'swish': common_spec.Activation.SWISH,
              'gelu-sigmoid': common_spec.Activation.GELUSigmoid,
              'tanh': common_spec.Activation.Tanh,
              }

FEATURES_MERGE = {
    "concat": common_spec.EmbeddingsMerge.CONCAT,
    "sum": common_spec.EmbeddingsMerge.ADD,
}

ADAPTER_CONFIG = {
    'pfeiffer' : cfg.PfeifferConfig,
    'houslby' : cfg.HoulsbyConfig,
    'generic' : cfg.AdapterConfig
}

def _get_variable(variables, name):
    return variables[name].numpy()

def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    bias = variables.get("%s.bias" % scope)
    if bias is not None:
        spec.bias = bias.numpy()

def set_embeddings(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)

def set_conv1d(spec, variables, scope):
    spec.weight = _get_variable(variables, scope +'.weight')
    bias = _get_variable(variables, scope +'.bias')
    if bias is  not None :
        spec.bias = bias

def set_layer_norm(spec, variables, scope):
    try:
        spec.gamma = _get_variable(variables, "%s.weight" % scope)
        spec.beta = _get_variable(variables, "%s.bias" % scope)
    except KeyError:
        # Compatibility with older models using a custom LayerNorm module.
        spec.gamma = _get_variable(variables, "%s.a_2" % scope)
        spec.beta = _get_variable(variables, "%s.b_2" % scope) 

def set_ffn(spec, variables, scope):
    set_layer_norm(spec.LayerNorm, variables, "%s.LayerNorm" % scope)
    set_linear(spec.linear_0, variables, "%s.w_1" % scope)
    set_linear(spec.linear_1, variables, "%s.w_2" % scope)       

def set_multi_head_attention(spec, variables, scope, self_attention=False):
    if self_attention:
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        set_linear(split_layers[0], variables, "%s.linear_query" % scope)
        set_linear(split_layers[1], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[2], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[0], split_layers)
    else:
        set_linear(spec.linear[0], variables, "%s.linear_query" % scope)
        split_layers = [common_spec.LinearSpec() for _ in range(2)]
        set_linear(split_layers[0], variables, "%s.linear_keys" % scope)
        set_linear(split_layers[1], variables, "%s.linear_values" % scope)
        utils.fuse_linear(spec.linear[1], split_layers)
    set_linear(spec.linear[-1], variables, "%s.final_linear" % scope)
    if hasattr(spec, "relative_position_keys"):
        spec.relative_position_keys = _get_variable(
            variables, "%s.relative_positions_embeddings.weight" % scope
        )
        spec.relative_position_values = spec.relative_position_keys

def set_pairwise(spec, variables, scope):
    spec.data = _get_variable(variables, scope+'.data')

def set_deep_biaffine(spec, variables, scope):
    set_linear(spec.ffn1, variables, scope+'.ffn1')
    set_linear(spec.ffn2, variables, scope+'.ffn2')
    set_pairwise(spec.pairwise, variables, scope+'.pairwise')

def set_add_norm(spec, variables, scope):
    spec.merge = _get_variable() # TODO finish the merge
    set_layer_norm(spec.norm, variables, scope+'.norm')

def set_adapter(spec, variables, scope):
    set_ffn(spec.ffn_down, variables, scope+'.ffn_down')
    set_ffn(spec.ffn_up, variables, scope+'.ffn_up')
    if spec.ln_before is not None:
        set_layer_norm(spec.ln_before, variables, scope+'.ln_before')
    if spec.ln_after is not None:
        set_layer_norm(spec.ln_after, variables, scope+'.ln_after')

def set_posdep(spec, variables, scope):
    set_embeddings(spec.upos_embeddings, variables, scope+'.upos_embeddings')
    set_ffn(spec.upos_ffn, variables, scope+'.upos_ffn')
    set_ffn(spec.xpos_ffn, variables, scope+'.xpos_ffn')
    set_ffn(spec.feats_ffn, variables, scope+'.feats_ffn')
    set_ffn(spec.down_project, variables, scope+'.down_project')
    set_deep_biaffine(spec.unlabeled, variables, scope+'.unlabeled')
    set_deep_biaffine(spec.deprel, variables, scope+'.deprel')

def set_transformer(spec, variables, scope, adapter_config = 'pfeiffer'):
    # TODO change to set the transformer dynamicaly
    if adapter_config == 'pfeiffer':
        set_multi_head_attention(spec.attention_head, variables, scope+'.attention_head')
        set_add_norm(spec.original_ln_before, variables, scope+'.original_ln_before')
        set_ffn(spec.ffn, variables, scope+'.ffn')
        set_add_norm(spec.residual_ln_after, variables, scope+'.residual_ln_after')
        set_adapter(spec.output_adapter, variables, scope+'.output_adapter')
        set_add_norm(spec.original_ln_after, variables, scope+'.original_ln_after')
    elif adapter_config == 'houlsby':
        set_multi_head_attention(spec.attention_head, variables, scope+'.attention_head')
        set_adapter(spec.mh_adapter, variables, scope+'.mh_adapter')
        set_add_norm(spec.original_ln_before, variables, scope+'.original_ln_before')
        set_ffn(spec.ffn, variables, scope+'.ffn')
        set_add_norm(spec.residual_ln_after, variables, scope+'.residual_ln_after')
        set_adapter(spec.output_adapter, variables, scope+'.output_adapter')
        set_add_norm(spec.original_ln_after, variables, scope+'.original_ln_after')    
    else :
        try :
            auto_set(spec, variables, scope)
        except:
            raise NotImplemented("auto set couldn't setup the config")


def get_key_from_value(dic: dict, value):
    for key, val in dict.items():
        if val == value:
            return key
    return None

def build_adapter_cfg(config: XLMRobertaConfig):
    adapter_dict = build_adapter_dict(config)
    adapter_cfg = {}
    for adapter, str_cfg in adapter_dict.items():
        if adapter in adapter_cfg:
            adapter_cfg[adapter].append(ADAPTER_CONFIG[str_cfg])
        else :
            adapter_cfg[adapter] = [ADAPTER_CONFIG[str_cfg]]
    return adapter_cfg


def build_adapter_dict(config: XLMRobertaConfig):
    """Builds a dict with adapter names as key and adapter type as items with string format.

    Args:
        variables (_type_): model from the checkpoint

    Returns:
        dict: a dict with adapter config as key and adapter name as item to enable auto_set
    """
    # TODO for now only one adapter is supported for the next steps in the building of xlmr spec
    adapter_dict = {}
    try :
        adapters = [adapter for adapter in config.adapters.adapters.adapters]
        for adapter in adapters:
            adapter_cfg = type(config.adapters.get(adapter))
            str_cfg = get_key_from_value(ADAPTER_CONFIG, adapter_cfg)
            if str_cfg is None:
                str_cfg = 'generic'
            if adapter in adapter_dict.items():
                adapter_dict[adapter].append(str_cfg)
            else:
                adapter_dict[adapter] = [str_cfg]
    except :
        pass
    return adapter_dict

def get_attributes(instance):
    """Returns all the attributes of a class

    Args:
        instance (any): an instance of any class

    Returns:
        list: the attributes of a class
    """
    try:
        attributes =  list(vars(instance).keys())
    except:
        attributes = None
    return attributes

def get_spec(spec:model_spec.LayerSpec, attributes:list):
    """
    Args:
        spec (model_spec.ModelSpec): _description_
        attributes (list): _description_

    Returns:
        dict: the link between the attributes and the spec class
    """
    link = dict()
    for attribute in attributes:
        att_cls = spec.attribute.__class__
        str_cls = get_key_from_value(SPECS,att_cls)
        if str_cls is None:
            raise NotImplementedError(f"The spec layer hasn't been implemented, must be in {SPECS}")
        if str_cls in link.keys():
            link[str_cls].append(attribute)
        else :
            link[str_cls] = [attribute]
    return link

def get_spec_link(spec:model_spec.LayerSpec):
    """
    Args:
        spec (model_spec.ModelSpec | model_spec.LayerSpec): a spec for linking its attributes and their class

    Returns:
        dict: the link between the attributes of the spec and their class
    """
    attributes = get_attributes(spec)
    return get_spec(spec, attributes)

def auto_set(spec:model_spec.LayerSpec, variables, scope=''):
    """Auto sets the spec and the model, for a model with basic specs in the dict above

    Args:
        spec (model_spec.LayerSpec): the spec of the decoder
        variables (decoder): the decoder of the model
    """
    spec_link = get_spec_link(spec)
    for spec_key, attributes in spec_link.items():
        for attribute in attributes:
            SET[spec_key](getattr(spec, attribute) , variables, scope+'.'+attribute)


SET = { # see position encoding and input layer?
    'linear' : set_linear,
    'embeddings' : set_embeddings,
    'conv1D' : set_conv1d,
    'layernorm' : set_layer_norm,
    'ffn' : set_ffn,
    'mha' : set_multi_head_attention,
    'pairwise' : set_pairwise,
    'deepbiaffine' : set_deep_biaffine,
    'addnorm' : set_add_norm,
    'adapter' : set_adapter,
    'posdep' : set_posdep,
    'transformer' : set_transformer
}
# --------------------------------------------------------

class InvertibleAdapterSpec(model_spec.LayerSpec):
    def __init__(self, config=None, # TODO check that
                 block_type= 'nice',
                 non_linearity = 'relu',
                 reduction_factor = 16) -> None:
        super().__init__()
        # self.activation = ACTIVATION[non_linearity]
        adapters = build_adapter_dict(config)
        for adapter in adapters:
            pass
         # TODO add the functions to manage the adapters
         # even if not used

class AdapterSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self,
                 non_linearity="relu",
                 add_layer_norm_before=True,
                 add_layer_norm_after=False,
                 residual_before_ln=True,
                 adapter_cfg='pfeiffer') -> None:
        super().__init__()
        self.non_linearity = ACTIVATION[non_linearity]
        if adapter_cfg == 'pfeiffer':
            self.ff_down = SPECS['linear']()
            self.ff_up = SPECS['linear']()
            self.merge = FEATURES_MERGE['sum']
        elif adapter_cfg == 'houlsby':
            self.ff_down = SPECS['linear']()
            self.ff_up = SPECS['linear']()
            self.merge = FEATURES_MERGE['sum']
        else:
            if add_layer_norm_before:
                self.ln_before = SPECS['layernorm']()
            self.ff_down = SPECS['linear']()
            self.ff_up = SPECS['linear']()
            if residual_before_ln:
                self.merge_ln = FEATURES_MERGE['sum']
            if add_layer_norm_after:
                self.ln_after = SPECS['layernorm']()
            self.merge = FEATURES_MERGE['sum']
        

class AddNormSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.merge = FEATURES_MERGE['sum']
        self.norm = SPECS['layernorm']()

class TransformerMixinSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self, original_ln_before = True,
                 original_ln_after = True,
                 residual_before_ln = True,
                 adapter_residual_before_ln = False,
                 ln_before = False,
                 ln_after = False,
                 mh_adapter = False,
                 output_adapter = True,
                 non_linearity = 'relu',
                 reduction_factor = 16, # might be useless
                 invertible_adapter  = InvertibleAdapterSpec(),
                 leave_out : list[int] = field(default_factory=list),
                 adapter_config = 'pfeiffer' # TODO remove to enable building dynamically
                 ) -> None:
        super().__init__()
        # TODO will be removed and use the class below for improvement
        if adapter_config == 'pfeiffer':
            self.pfeiffer_spec()
        elif adapter_config == 'houlsby':
            self.houlsby_spec()
        else :
            raise NotImplementedError("There is not yet a dynamic build for Transformers")
        # TODO add the invertible adapter and leave_out
        self.invertible_adapter = invertible_adapter
        self.leave_out = leave_out
        self.activation = ACTIVATION[non_linearity]
        # TODO check these functions to enable building dyamically
        # not finished ------------------------------
        # self.attention_head = SPECS['mha']
        # self.original_ln_before = ( AddNormSpec()
        #                             if original_ln_before
        #                             else None)        
        # self.mh_adapter = (AdapterSpec(ln_before, ln_after, reduction_factor)
        #                    if mh_adapter else None)
        # self.ffn = transformer_spec.FeedForwardSpec()
        # self.original_ln_after = ( AddNormSpec()
        #                            if original_ln_after
        #                            else None)
        # self.residual_before_ln(residual_before_ln) # find what it changes
        # self.adapter_residual_before_ln(adapter_residual_before_ln) # find what it changes
        # self.output_adapter = (AdapterSpec(ln_before, ln_after, reduction_factor)
        #                        if output_adapter else None)
        # self.activation = np.dtype("int8").type(ACTIVATION.get(non_linearity))
        # --------------------------------------------

    def pfeiffer_spec(self):
        self.attention_head = SPECS['mha']()
        self.original_ln_before = AddNormSpec()
        self.ffn = transformer_spec.FeedForwardSpec()
        self.residual_ln_after = AddNormSpec()
        self.output_adapter = AdapterSpec()
        self.original_ln_after = AddNormSpec()

    def houlsby_spec(self):
        self.attention_head = SPECS['mha']()
        self.mh_adapter = AdapterSpec() 
        self.original_ln_before = AddNormSpec()
        self.ffn = transformer_spec.FeedForwardSpec()
        self.residual_ln_after = AddNormSpec()
        self.output_adapter = AdapterSpec()
        self.original_ln_after = AddNormSpec()

class PfeifferSpec(TransformerMixinSpec):
    """DONE
    The specs of the most used adapter architectur, the Pfeiffer
    """
    def __init__(self) -> None:
        super().__init__(original_ln_before=True,
                         original_ln_after = True,
                         residual_before_ln = True,
                         adapter_residual_before_ln = False,
                         ln_before = False,
                         ln_after = False,
                         mh_adapter = False,
                         output_adapter = True,
                         non_linearity = "relu",
                         reduction_factor = 16,
                         invertible_adapter = InvertibleAdapterSpec(
                            block_type="nice", non_linearity="relu", 
                            reduction_factor=2
                            )
                         )

class HoulsbySpec(TransformerMixinSpec):
    """DONE
    The specs of the Houslby architectur
    """
    def __init__(self) -> None:
        super().__init__(original_ln_before = False, 
                         original_ln_after = True, 
                         residual_before_ln = True, 
                         adapter_residual_before_ln = False, 
                         ln_before = False, 
                         ln_after = False, 
                         mh_adapter = True, 
                         output_adapter = True, 
                         non_linearity = 'swish', 
                         reduction_factor = 16,
                         config='houlsby' # TODO remove to build dynamically
                         )

class ModuleDictSpec(model_spec.LayerSpec):
    """DONE
    Enables to go throug some torch.nn layers got like a dictionary (core of adapters)
    A dict of spec to go throug elements built like the nn.ModuleDict
    """

    def __init__(self, modules:dict[str, model_spec.LayerSpec]=dict()) -> None:
        super().__init__()
        # TODO finish this class to be dynamic 
        self._modules: dict[str,model_spec.LayerSpec] = {}
        self.update(modules)
        self.check_modules()
    
    def check_modules(self):
        if len(self._modules)>0:
            self.empty = False
        else:
            self.empty = True
    
    def add_module(self, key, mod):
        self._modules[key] = mod 
        self.check_modules()
    
    def __getitem__(self, key:str)->model_spec.LayerSpec:
        return self._modules.get(key)
    
    def __setitem__(self,key:str, mod:model_spec.LayerSpec) -> None:
        self.add_module(key,mod)

    def __delitem__(self, key:str) -> None:
        del self._modules[key]
        self.check_modules()
    
    def __len__(self) -> int:
        return len(self._modules)
    
    def __iter__(self) -> Iterable[str]:
        return iter(self._modules)
    
    def __containns__(self, key:str) -> bool:
        return key in self._modules
    
    def clear(self) -> None:
        self.empty = True
        return self._modules.clear()
    
    def pop(self, key:str) -> model_spec.LayerSpec:
        value = self[key]
        del self[key]
        self.check_modules()
        return value

    def items(self) -> Iterable[tuple[str, model_spec.LayerSpec]]:
        return self._modules.items()
    
    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    def values(self) -> Iterable[model_spec.LayerSpec]:
        return self._modules.values()
    
    def update(self, modules:dict[str,model_spec.LayerSpec]={}) -> None:
        if isinstance(modules, (dict, ModuleDictSpec)):
            for key, mod in modules.items():
                self[key] = mod
        else :
            raise TypeError("modules must be a dict with str as key and layerspec as values")

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
            self += modules
    
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
    
class ModelAdaptersMixinSpec:
    """DONE
    The class from which ModelAdaptersMixin inherits is a metaclass => no weights
    """
    def __init__(self, config:XLMRobertaConfig) -> None:
        self.config = config 
        self.model_name = config.model_type
        self._active_adapter_names = config.adapters.active_setup[:]
        # TODO add the elements : adapters.adapters, adapters.adapter_list, adapter_fusion, adapter_fusion_models

    def add_adapter(self, adapter_name:str, adapter_spec:model_spec.LayerSpec, config=None) -> None:
        pass

    def active_adapters(self):
        return self._active_adapter_names
    
    def has_adapters(self):
        return len(self.config.adapters.adapters) > 0
    
    def parse_adapter_names(adapter_names: list) -> list[list[str]]:
        if adapter_names is not None:
            if isinstance(adapter_names, str):
                adapter_names = [[adapter_names]]
            elif isinstance(adapter_names, list):
                if isinstance(adapter_names[0], str):
                    adapter_names = [adapter_names]
            if not isinstance(adapter_names[0][0], str):
                raise ValueError("Adapter names %s not set correctly", str(adapter_names))
        return adapter_names

class BertLayerAdapterMixinSpec:
    """DONE
    Adds adapters to the BertLayer module.
    """
    def add_adapter(self, adapter_name:str, adapter_cfg:str):
        self.attention.output.add_adapter(adapter_name, adapter_cfg)
        self.output.add_adapter(adapter_name, adapter_cfg)

class NICECouplingBlockSpec(model_spec.LayerSpec):
    """DONE
    Coupling block following the NICE design
    """
    def __init__(self, non_linearity='relu') -> None:
        super().__init__()
        self.F = ModuleListSpec([
            SPECS['linear'](),
            np.dtype("int8").type(ACTIVATION[non_linearity]),
            SPECS['linear'](),      
        ])
        self.G = ModuleListSpec([
            SPECS['linear'](),
            np.dtype("int8").type(ACTIVATION[non_linearity]),
            SPECS['linear'](),
        ])

class GLOWCouplingBlockSpec(model_spec.LayerSpec):
    """DONE
    Coupling block following the GLOW design
    """
    def __init__(self, non_linearity='relu') -> None:
        super().__init__()
        self.s1 = ModuleListSpec([
            SPECS['linear'](),
            np.dtype("int8").type(ACTIVATION[non_linearity]),
            SPECS['linear'](),      
        ])
        self.s2 = ModuleListSpec([
            SPECS['linear'](),
            np.dtype("int8").type(ACTIVATION[non_linearity]),
            SPECS['linear'](),      
        ])

class BertModelAdaptersMixinSpec(ModelAdaptersMixinSpec):
    """DONE
    Adds adapters to the BertModel module => weight
    """
    def __init__(self, config:XLMRobertaConfig) -> None:
        super().__init__(config)
    
    def init_adapter_modules(self, config) -> None:
        self.invertible_lang_adapters = ModuleDictSpec()
        adapters_spec = build_adapter_dict(config)
        for adapter_name, str_spec in adapters_spec:
            self.encoder.add_adapter(adapter_name,str_spec)

    def add_adapter(self, adapter_name:str, adapter_spec:str, config=None):
        if not (adapter_spec in ADAPTER_SPEC):
            raise ValueError(f"Invalid adapter type, must be in {ADAPTER_CONFIG.keys()}")
        self.config.adapters.add(adapter_name=adapter_name,
                                 adapter_type=adapter_spec,
                                 config=config)
        self.encoder.add_adapter(adapter_name,adapter_spec)
            # TODO add the mangement of the adapter type text_lang
    
    def add_invertible_lang_adapter(self, lgge):
        if lgge in self.invertible_lang_adapters:
            raise ValueError(f"Model already contains an adapter module for {lgge}.")
        inv_adap_cfg = self.config.adapters.get(lgge).inv_adapter
        if inv_adap_cfg["block_type"] == 'nice':
            inv_adap = NICECouplingBlockSpec(non_linearity=inv_adap_cfg['non_linearity'])
        elif inv_adap_cfg["block_type"] == 'glow':
            inv_adap = GLOWCouplingBlockSpec(non_linearity=inv_adap_cfg['non_linearity'])
        else:
            raise ValueError(f"Invalid invertible adapter typoe {inv_adap_cfg['block_type']}.")
        self.invertible_lang_adapters[lgge] = inv_adap

class ModuleUtilsMixin:
    """DONE"""
    def num_paramters(self, only_trainable:bool = False) -> int:
        params = filter(lambda x: x.requires_grad, self.parameters()) if only_trainable else self.parameters()
        return sum(p.numel() for p in params)

class PreTrainedModelSpec(model_spec.LayerSpec, ModuleUtilsMixin):
    """DONE"""

    config_class = None
    base_moel_prefix = ""

    def __init__(self, config:XLMRobertaConfig) -> None:
        if not isinstance(config, XLMRobertaConfig):
            raise TypeError(
                f"Parameter config in {config} must be an instance of 'XLMRobertaConfig' from the prepprosessing pipeline"
            )
        self.config = config
    
    def base_model(self):
        return getattr(self, self.base_model, self)
    
    def prune_heads(self, head_to_prune:dict):
        return NotImplemented

class BertPreTrainedModelSpec(PreTrainedModelSpec):
    """DONE"""
    def __init__(self,config) -> None:
        # TODO see if needed because class used to init weights
        super().__init__(config)

class BertEmbeddingsSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = SPECS['embeddings']()
        self.position_embeddings = SPECS['embeddings']()
        self.token_type_embeddings = SPECS['embeddings']()
        self.LayerNorm = BertLayerNormSpec() 

class BertSelfAttentionSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.query = SPECS['linear']()
        self.key = SPECS['linear']()
        self.value = SPECS['linear']()

class BertSelfOutputAdapterMixinSpec:
    """DONE
    Adds adapters to the BertSelfOutput module.
    """
    def _init_adapter_modules(self) -> None:
        self.attention_text_task_adapters = ModuleDictSpec()
        self.adapter_fusion_layer = ModuleDictSpec()
        self.attention_lang_adapters = ModuleDictSpec()
    
    def add_adapters(self, adapter_name:str, adapter_cfg:str):
        "For now the only task supported is the attention lang"
        if adapter_cfg in ADAPTER_CONFIG:
            adapter = AdapterSpec(adapter_config=adapter_cfg)
            self.attention_text_task_adapters[adapter_name] = adapter
        else :
            raise NotImplementedError("the dynamic adapter config is not implemented yet")
    
    def get_adapter_layer(self, adapter_name:str):
        if adapter_name in self.attention_text_task_adapters:
            return self.attention_text_task_adapters[adapter_name]
        else :
            return None

class BertSelfOutputSpec(model_spec.LayerSpec, BertSelfOutputAdapterMixinSpec):
    """DONE"""
    def __init__(self) -> None:
        model_spec.LayerSpec().__init__()
        self._init_adapter_modules()
        self.dense = SPECS['linear']()
        self.LayerNorm = BertLayerNormSpec()
        # TODO mixin part

class BertAttentionSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self) -> None:
        self.self = BertSelfAttentionSpec()
        self.output = BertSelfOutputSpec()

class BertIntermediateSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = SPECS['linear']()
        self.intermediate_act_fn = (np.dtype("int8").type(ACTIVATION.get(config.hidden_act))
                                    if isinstance(config.hidden_act, str)
                                    else config.hidden_act)

class BertOutputAdaptersMixinSpec:
    """DONE"""
    def _init_adapter_modules(self):
        self.adapter_fusion_layer = ModuleDictSpec()
        self.layer_text_task_adapters = ModuleDictSpec()
        self.layer_text_lang_adapters = ModuleDictSpec()
    
    def add_adapter(self, adapter_name:str, adapter_cfg:str):
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config['output_adapter']:
            adapter = AdapterSpec(adapter_cfg=adapter_cfg)
            self.layer_text_task_adapters[adapter_name] = adapter

    def get_adapter_layer(self, adapter_name:str):
        if adapter_name in self.layer_text_task_adapters:
            return self.layer_text_task_adapters[adapter_name]
        else:
            raise None

class BertOutputSpec(model_spec.LayerSpec, BertOutputAdaptersMixinSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.dense = SPECS['linear']()
        self.LayerNorm = BertLayerNormSpec()
        self._init_adapter_modules()
        # TODO complete the adaptermixin implementation

class BertLayerSpec(model_spec.LayerSpec,BertLayerAdapterMixinSpec):
    """DONE"""
    def __init__(self, config: XLMRobertaConfig) -> None:
        super().__init__()
        self.attention = BertAttentionSpec()
        self.is_decoder = config.is_decoder # always false
        if self.is_decoder:
            self.crossattention = BertAttentionSpec()
        self.intermediate = BertIntermediateSpec(config)
        self.output = BertOutputSpec()

class BertEncoderAdapterMixinSpec:
    """DONE
    Adds adapters to the BertEncoder module.
    """
    def add_adapter(self, adapter_name:str, adapter_cfg:str) -> None:
        leave_out = self.config.adapters.get(adapter_name).leave_out
        # TODO add mangement of the layers to skip
        for i, layer in enumerate(self.layer):
            if i not in leave_out:
                layer.add_adapter(adapter_name, adapter_cfg)

class BertEncoderSpec(model_spec.LayerSpec, BertEncoderAdapterMixinSpec):
    """DONE"""
    def __init__(self, config) -> None:
        # TODO modulelistspec
        self.layer = ModuleListSpec(
            [BertLayerSpec(config) for _ in range(config.num_hidden_layers)])

class BertPoolerSpec(model_spec.LayerSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.dense = SPECS['linear']()
        self.activation = np.dtype("int8").type(ACTIVATION['tanh'])

class BertModelSpec(BertModelAdaptersMixinSpec, BertPreTrainedModelSpec):
    """DONE"""
    def __init__(self, config) -> None:
        # TODO finish the mixin
        super().__init__(config)
        self.init_adapter_modules(config)
        self.config = config
        self.embeddings = BertEmbeddingsSpec()
        self.encoder = BertEncoderSpec(config)
        self.pooler = BertPoolerSpec()

        # TODO add the init of the adapters (init the weights or the adapters themselves, if just weight to nothing)

class RobertaEmbeddingsSpec(BertEmbeddingsSpec):
    """DONE"""
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = SPECS['embeddings']()
        self.position_embeddings = SPECS['embeddings']()

class RobertaModelSpec(BertModelSpec):
    """DONE"""
    def __init__(self, config:XLMRobertaConfig) -> None:
        super().__init__(config)
        self.embeddings = RobertaEmbeddingsSpec()
        # TODO check the init of the weight

class RobertaConfig():
    def __init__(self) -> None:
        pass

class XLMRobertaModelConfig(model_spec.LanguageModelConfig):
    def __init__(self) -> None:
        pass

class XLMRobertaModelSpec(RobertaModelSpec):
    """DONE"""
    def __init__(self, config:XLMRobertaConfig) -> None:
        """Generates a XLM-Roberta model with active adapters

        Args:
            config (_type_): _description_
            adapters (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(config)
        # to have the adapters use build_adapter_dict(config)
        # TODO see how the config
        self.config_class = XLMRobertaModelConfig

    @classmethod
    def from_config(cls, config:XLMRobertaConfig):
        return cls(config)
    
ADAPTER_SPEC = {
    'pfeiffer' : PfeifferSpec,
    'houslby' : HoulsbySpec,
    'generic' : TransformerMixinSpec
}

SPECS = { # add the xlmrspecs further
    'linear' : common_spec.LinearSpec,
    'embeddings' : common_spec.EmbeddingsSpec,
    'conv1D' : common_spec.Conv1DSpec,
    'layernorm' : common_spec.LayerNormSpec,
    'ffn' : transformer_spec.FeedForwardSpec,
    'mha' : attention_spec.MultiHeadAttentionSpec,
    'deepbiaffine' : posdep_spec.DeepBiaffineSpec,
    'pairwise' : posdep_spec.PairwiseSpec,
    'posdep' : posdep_spec.PosdepDecoderSpec,
    'addnorm' : AddNormSpec,
    'adapter' : AdapterSpec,
    'transformer' : TransformerMixinSpec,
    'bertlayer' : BertLayerSpec,
    'bertattention' : BertAttentionSpec,
    'bertintermediate' : BertIntermediateSpec,
    'bertoutput' : BertOutputSpec,
    'bertpooler' : BertPoolerSpec,
    'bertembeddings' : BertEmbeddingsSpec,
    'bertselfattention' : BertSelfAttentionSpec,
    'bertselfoutput' : BertSelfOutputSpec,
} # todo add the xlmr basic specs in this file to use it elsewhere than xlmr_spec if needed

class BertLayerNormSpec(SPECS['layernorm']):
    """DONE"""
    def __init__(self) -> None:
        """Nothing else than a layer norm but trankit wanted it to be different so we keep the architecture"""
        super().__init__()
        # TODO pb of the epsilon not managed during conversion
        # self.epsilon = None

def build_adapter_spec(config: XLMRobertaConfig):
    adapter_dict = build_adapter_dict(config)
    adapter_spec = {}
    for adapter, str_cfg in adapter_dict.items():
        if adapter in adapter_spec:
            adapter_spec[adapter].append(ADAPTER_SPEC[str_cfg])
        else :
            adapter_spec[adapter] = [ADAPTER_SPEC[str_cfg]]
    return adapter_spec

# to get the adapter need of model.config.adapters[adapter_name]
# or use model.encoder.xlmr.config.adapters.get(adapter_name) -> adapter config (pfeiffer or houlsby)