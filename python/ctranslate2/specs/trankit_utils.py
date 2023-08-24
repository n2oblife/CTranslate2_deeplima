import numpy as np
from ctranslate2.converters import utils

# this file suffers a problem of cyclic import and it would be useful to review the architecture
import transformers.adapters.configuration as cfg
from ctranslate2.specs import common_spec
from trankit.adapter_transformers.configuration_xlm_roberta import XLMRobertaConfig


TASK = ['tokenize', 'posdep', 'mwt', 'lemmatize', 'ner']
NEED_EMBEDDING = ['tokenize', 'posdep', 'ner']

# EMBEDDING_CLS = {'xlmr': xlmr_spec.XLMRobertaModelSpec} # encoder
# TASK_CLS = {'posdep': posdep_spec.PosdepDecoderSpec} # decoder

ACTIVATION = {'relu': common_spec.Activation.RELU,
              'gelu': common_spec.Activation.GELU,
              'fast-gelu': common_spec.Activation.GELUTanh,
              'silu': common_spec.Activation.SWISH,
              'gelu-sigmoid': common_spec.Activation.GELUSigmoid,
              'tanh': common_spec.Activation.Tanh,
              }

FEATURES_MERGE = {
    "concat": common_spec.EmbeddingsMerge.CONCAT,
    "sum": common_spec.EmbeddingsMerge.ADD,
}

# SPECS = { # add the xlmrspecs further
#     'linear' : common_spec.LinearSpec,
#     'embeddings' : common_spec.EmbeddingsSpec,
#     'conv1D' : common_spec.Conv1DSpec,
#     'layernorm' : common_spec.LayerNormSpec,
#     'ffn' : transformer_spec.FeedForwardSpec,
#     'mha' : attention_spec.MultiHeadAttentionSpec,
#     'deepbiaffine' : posdep_spec.DeepBiaffineSpec,
#     'pairwise' : posdep_spec.PairwiseSpec,
#     'addnorm' : xlmr_spec.AddNormSpec,
#     'adapter' : xlmr_spec.AdapterSpec,
#     'posdep' : posdep_spec.PosdepDecoderSpec,
#     'transformer' : xlmr_spec.TransformerMixinSpec
# } # todo add the xlmr basic specs in this file to use it elsewhere than xlmr_spec.pu if needed

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
    set_layer_norm(spec.layer_norm, variables, "%s.layer_norm" % scope)
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
        # try :
        #     auto_set(spec, variables, scope)
        # except:
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

# def get_spec(spec:model_spec.LayerSpec, attributes:list):
#     """
#     Args:
#         spec (model_spec.ModelSpec): _description_
#         attributes (list): _description_

#     Returns:
#         dict: the link between the attributes and the spec class
#     """
#     link = dict()
#     for attribute in attributes:
#         att_cls = spec.attribute.__class__
#         str_cls = get_key_from_value(SPECS,att_cls)
#         if str_cls is None:
#             raise NotImplementedError(f"The spec layer hasn't been implemented, must be in {SPECS}")
#         if str_cls in link.keys():
#             link[str_cls].append(attribute)
#         else :
#             link[str_cls] = [attribute]
#     return link

# def get_spec_link(spec:model_spec.LayerSpec):
#     """
#     Args:
#         spec (model_spec.ModelSpec | model_spec.LayerSpec): a spec for linking its attributes and their class

#     Returns:
#         dict: the link between the attributes of the spec and their class
#     """
#     attributes = get_attributes(spec)
#     return get_spec(spec, attributes)

# def auto_set(spec:model_spec.LayerSpec, variables, scope=''):
#     """Auto sets the spec and the model, for a model with basic specs in the dict above

#     Args:
#         spec (model_spec.LayerSpec): the spec of the decoder
#         variables (decoder): the decoder of the model
#     """
#     spec_link = get_spec_link(spec)
#     for spec_key, attributes in spec_link.items():
#         for attribute in attributes:
#             SET[spec_key](getattr(spec, attribute) , variables, scope+'.'+attribute)


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