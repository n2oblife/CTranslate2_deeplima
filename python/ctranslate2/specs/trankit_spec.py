# import sys 
# sys.path.append('/home/zk274707/Projet/proto/env_proto/lib/python3.10/site-packages/ctranslate2/specs')
import numpy as np
from typing import Optional, Tuple, Union
from ctranslate2.converters import utils
from ctranslate2 import converters as cv
from ctranslate2.specs import common_spec, model_spec, posdep_spec, xlmr_spec, transformer_spec, attention_spec

from onmt.utils.parse import ArgumentParser
from collections import OrderedDict

# TODO review the architecture of he project to gather the dicts and functions

# --------------------------------------------------------
# dicts and functions

EMBEDDING_CLS = {'xlmr': xlmr_spec.XLMRobertaModelSpec} # encoder
TASK_CLS = {'posdep': posdep_spec.PosdepDecoderSpec} # decoder

NEED_EMBEDDING = ['tokenize', 'posdep', 'ner']

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

SPECS = { # add the xlmrspecs further
    'linear' : common_spec.LinearSpec,
    'embeddings' : common_spec.EmbeddingsSpec,
    'conv1D' : common_spec.Conv1DSpec,
    'layernorm' : common_spec.LayerNormSpec,
    'ffn' : transformer_spec.FeedForwardSpec,
    'mha' : attention_spec.MultiHeadAttentionSpec,
    'deepbiaffine' : posdep_spec.DeepBiaffineSpec,
    'pairwise' : posdep_spec.PairwiseSpec,
    'addnorm' : xlmr_spec.AddNormSpec,
    'adapter' : xlmr_spec.AdapterSpec,
    'posdep' : posdep_spec.PosdepDecoderSpec,
    'transformer' : xlmr_spec.TransformerMixinSpec,
    'bertlayer' : xlmr_spec.BertLayerSpec,
    'bertattention' : xlmr_spec.BertAttentionSpec,
    'bertintermediate' : xlmr_spec.BertIntermediateSpec,
    'bertoutput' : xlmr_spec.BertOutputSpec,
    'bertpooler' : xlmr_spec.BertPoolerSpec,
    'bertembeddings' : xlmr_spec.BertEmbeddingsSpec,
    'bertselfattention' : xlmr_spec.BertSelfAttentionSpec,
    'bertselfoutput' : xlmr_spec.BertSelfOutputSpec,
} # todo add the xlmr basic specs in this file to use it elsewhere than xlmr_spec if needed

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

def get_key_from_value(dico: dict, value):
    for key, val in dico.items():
        if isinstance(value, val):
            return key
    return None

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
            raise NotImplementedError(f"The spec layer {att_cls} hasn't been implemented, must be in {SPECS.keys()}")
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
    spec.pairwise_weight = _get_variable(variables, scope+'.pairwise_weight')

def set_deep_biaffine(spec, variables, scope):
    # TODO change this number for a dyamic model of genericlayer
    # set_module_list(spec.ffn1, variables, scope+'.ffn1')
    # set_module_list(spec.ffn2, variables, scope+'.ffn2')
    set_linear(getattr(spec.ffn1, '0'), variables, scope+'.ffn1.0') 
    set_linear(getattr(spec.ffn2, '0'), variables, scope+'.ffn2.0')
    set_pairwise(spec, variables, scope) # indeed not a spec in model

def set_branch_layer(spec, variables, scope):
    for spec_key, spec_value in spec.__dict__.items():
        spec_str = get_key_from_value(SPECS, spec_value)
        if spec_str in SET:
            SET[spec_str](spec(spec_key), variables, scope+'.'+spec_key)
        else:
            raise NotImplementedError(f"The set {spec_value.__class__.__name__} has not been yet released, it must be in {SET.keys()}")

def set_add_norm(spec, variables, scope):
    set_layer_norm(spec.norm, variables, scope+'.norm')

def set_adapter(spec, variables, scope):
    set_ffn(spec.ffn_down, variables, scope+'.ffn_down')
    set_ffn(spec.ffn_up, variables, scope+'.ffn_up')
    if spec.ln_before is not None:
        set_layer_norm(spec.ln_before, variables, scope+'.ln_before')
    if spec.ln_after is not None:
        set_layer_norm(spec.ln_after, variables, scope+'.ln_after')

def set_posdep(spec, variables, scope):
    set_embeddings(spec.upos_embeddings, variables, scope+'.upos_embedding')
    set_linear(spec.upos_ffn, variables, scope+'.upos_ffn')
    set_linear(spec.xpos_ffn, variables, scope+'.xpos_ffn')
    set_linear(spec.feats_ffn, variables, scope+'.feats_ffn')
    set_linear(spec.down_project, variables, scope+'.down_project')
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
        raise NotImplementedError("auto set couldn't setup the config")

def set_roberta_embeddings(spec, variables, scope):
    set_embeddings(spec.word_embeddings, variables, scope+'.word_embeddings')
    set_embeddings(spec.position_embeddings, variables, scope+'.position_embeddings')
    set_bert_embeddings(spec, variables, scope)

def set_bert_embeddings(spec, variables, scope):
    set_embeddings(spec.word_embeddings, variables, scope+'.word_embeddings')
    set_embeddings(spec.position_embeddings, variables, scope+'.position_embeddings')
    set_embeddings(spec.token_type_embeddings, variables, scope+'.token_type_embeddings')
    set_layer_norm(spec.LayerNorm, variables, scope+'.LayerNorm') # check if LayerNorm instead

def set_module_list(spec, variables, scope):
    for i,sp in enumerate(spec):
        spec_str = get_key_from_value(SPECS, sp)
        if spec_str in SET:
            SET[spec_str](spec[i], variables, scope+'.'+str(i))
        else:
            raise NotImplementedError(f"The set {sp.__class__.__name__} has not been yet released, it must be in {SET.keys()}")

def set_module_dict(spec, variables, scope):
    for spec_key, spec_value in spec.items():
        spec_str = get_key_from_value(SPECS, spec_value)
        if spec_str in SET:
            SET[spec_str](getattr(spec,'.'+spec_key), variables, scope+'.'+spec_key)
        else:
            raise NotImplementedError(f"The set has not been yet released, it must be in {SET.keys()}")

def set_bert_self_attention(spec, variables, scope):
    set_linear(spec.query, variables, scope+'.query')
    set_linear(spec.key, variables, scope+'.key')
    set_linear(spec.value, variables, scope+'.value')


def set_bert_self_output(spec, variables, scope):
    set_linear(spec.dense, variables, scope+'.dense')
    set_layer_norm(spec.LayerNorm, variables, scope+'.LayerNorm')
    # TODO add the mangement of the different tasks
    # only the attention_text_task_adapters for the mixin part
    set_module_dict(spec.attention_text_task_adapters, variables, scope+'.attention_text_task_adapters')

def set_bert_attention(spec, variables, scope):
    set_bert_self_attention(spec.self, variables, scope+'.self')
    set_bert_self_output(spec.output, variables, scope+'.output')

def set_bert_intermediate(spec, variables, scope):
    set_linear(spec.dense, variables, scope+'.dense')

def set_bert_output(spec, variables, scope):
    set_linear(spec.dense, variables, scope+'.dense')
    set_layer_norm(spec.LayerNorm, variables, scope+'.LayerNorm')
    # TODO add the management of the different tasks
    # only the layer_text_task_adapters for the mixin part
    set_module_dict(spec.layer_text_task_adapters, variables, scope+'.layer_text_task_adapters')

def set_bert_layer(spec, variables, scope):
    set_bert_attention(spec.attention, variables, scope+'.attention')
    if spec.is_decoder:
        set_bert_attention(spec.crossattention, variables, scope+'.crossattention')
    set_bert_intermediate(spec.intermediate, variables, scope+'.intermediate')
    set_bert_output(spec.output, variables, scope+'.output')

def set_bert_encoder(spec, variables, scope):
    set_module_list(spec.layer, variables, scope+'.layer')

def set_bert_pooler(spec, variables, scope):
    set_linear(spec.dense, variables, scope+'.dense')

def set_bert_model(spec, variables, scope):
    set_bert_embeddings(spec.embeddings, variables, scope+'.embeddings')
    set_bert_encoder(spec.encoder, variables, scope+'.encoder')
    set_bert_pooler(spec.pooler, variables, scope+'.pooler')
    set_module_dict(spec.invertible_lang_adapters, variables, scope+'.invertible_lang_adapters')
    # TODO not finished

def set_roberta_model(spec, variables, scope):
    set_roberta_embeddings(spec.embeddings, variables, scope+'.embeddings')
    set_bert_model(spec, variables, scope)

def set_xlm_roberta_model(spec, variables, scope):
    set_roberta_model(spec, variables, scope)
    # TODO check the adapters

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
    'transformer' : set_transformer,
    'bertlayer' : set_bert_layer,
    'bertattention' : set_bert_attention,
    'bertintermediate' : set_bert_intermediate,
    'bertoutput' : set_bert_output,
    'bertpooler' : set_bert_pooler,
    'bertembeddings' : set_bert_embeddings,
    'bertselfattention' : set_bert_self_attention,
    'bertselfoutput' : set_bert_self_output,
}
# the specs of invertible lang adapter is done but not the sets (coupling blocks)
# --------------------------------------------------------

class TrankitModelSpec(model_spec.ModelSpec):
    """Initialize a spec model from trankit"""
    def __init__(self, opt, config, decoder: model_spec.LayerSpec, 
                 encoder:model_spec.LayerSpec = None) -> None:
        super().__init__()
        self._opt = opt
        self._cfg = config # TODO  manage the json file saving etc (aimed for opt ?)
        self._vocabulary = None
        if encoder is not None:
            self.encoder = encoder        
        self.decoder = decoder
        self._name = opt.exp
        self._quantization = None # TODO change the checking of the quantization function 
    
    @classmethod
    def from_config(cls, opt, config):
        # check the opt and config
        encoder_spec = (EMBEDDING_CLS[opt.encoder_type].from_config(config) 
                        if opt.task in NEED_EMBEDDING else None)
        decoder_spec = TASK_CLS[opt.task].from_config()
        return cls(opt, config, decoder_spec, encoder_spec)
    
    def get_config(self):
        return self._cfg
    
    def get_default_config(self):
        return super().get_default_config()
    
    def register_vocabulary(self, tokens: list[str]) -> None:
        """Registers the vocabulary of tokens.

        Arguments:
          tokens: List of tokens.
        """
        self._vocabulary = list(tokens)
    
    @property
    def name(self):
        return self._name

class TrankitModelConfig(model_spec.ModelConfig):
    # TODO add the xlmr in the config
    def __init__(self, unk_token: str = "<unk>", bos_token: str = "<s>", eos_token: str = "</s>", **kwargs):
        super().__init__(unk_token, bos_token, eos_token, **kwargs)

def set_trankit_decoder(spec, variables, task):
    if task == 'posdep':
        set_posdep(spec.decoder, variables, 'decoder._tagger')
    else :
        try : 
            auto_set(spec.decoder, variables, 'decoder')
        except:
            raise NotImplementedError("Decoders are not implemented yet")

def set_trankit_encoder(spec, variables, task):
    if task == 'xlmr':
        set_xlm_roberta_model(spec.encoder, variables, 'encoder.xlmr.roberta')
    else :
        try :
            auto_set(spec.encoder, variables, 'encoder.xlmr.roberta')
        except:
            raise NotImplementedError("Encoders are not implemented yet")


def set_trankit_model(spec:model_spec.ModelSpec, 
                      variables, opt):
    if spec.encoder is not None:
        set_trankit_encoder(spec, variables, opt.encoder_type)
    set_trankit_decoder(spec, variables, opt.decoder_type)

def check_opt_trankit(opt, num_source_embeddings):
    activation_fn = getattr(opt, "pos_ffn_activation_fn", "relu")
    feat_merge = getattr(opt, "feat_merge", "concat")
    # TODO check for the relative position and self_attention type
    check = utils.ConfigurationChecker()
    check(
        opt.encoder_type in EMBEDDING_CLS,
        f"Option --encoder_type must be in {EMBEDDING_CLS}, or not supported"
    )
    check(
        opt.decoder_type in TASK_CLS,
        f"Option --deocder_type must be in {TASK_CLS}, or not supported"
    )
    check(
        activation_fn in ACTIVATION,
        "Option --pos_ffn_activation_fn %s is not supported (supported activations are: %s)"
        % (activation_fn, ", ".join(ACTIVATION.keys())),
    )
    check(
        num_source_embeddings == 1 or feat_merge in FEATURES_MERGE,
        "Option --feat_merge %s is not supported (supported merge modes are: %s)"
        % (feat_merge, " ".join(FEATURES_MERGE.keys())),
    )
    check.validate()

def get_model_spec_trankit(
        opt:ArgumentParser, variables:OrderedDict, src_vocab = [], tgt_vocab = [],num_source_embeddings:int=None, config=None ):
    """Creates a model specification from the model options adapted from trankit then sets the weights 
    from the model to the specification for conversion in ctranslate format.

    Args:
        opt (config from onmt): The .yaml option file parsed 
        variables (model): The model from the checkpoint
        src_vocab (list, optional): The source vocablary
        tgt_vocab (list, optional): The target vocabulary, here depends on the task
        num_source_embeddings (int, optional): The length of the src vocab 

    Returns:
        model_spec.ModelSpec: The architecture of the model
    """
    # checks if parameters are good
    if (opt.task in NEED_EMBEDDING) and (not opt.encoder_type in EMBEDDING_CLS) :
        raise TypeError(f"encoder argument must be in {EMBEDDING_CLS.keys()}, others are not implemented")
    if not opt.decoder_type in TASK_CLS:
        raise TypeError(f"decoder argument must be in {TASK_CLS.keys()}, others are not implemented")
    # model_spec = TrankitModelSpec.from_config(opt=variables.encoder.xlmr.config)
    model_spec = TrankitModelSpec.from_config(opt=opt, config=config)
    set_trankit_model(model_spec, variables, opt)

    for voc in tgt_vocab:
        model_spec.register_vocabulary(voc)

    return model_spec




