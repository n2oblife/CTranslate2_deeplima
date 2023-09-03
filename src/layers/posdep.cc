#include "ctranslate2/layers/posdep.h"
#include "ctranslate2/ops/ops.h"

#include <cmath>

namespace ctranslate2::layers
{

/**
* This class is based on the implementation from trankit
* from the work of Dozat & Manning
*/
DeepBiaffine::DeepBiaffine(const models::Model& model,
                           const std::string& scope,
                           const int in_dim1,
                           const int in_dim2,
                           const int output_dim,
                           const ops::ActivationType activation_type
                           ) :
    _activation(activation_type),
    // TODO ensure the class of the ffn
    // TODO review this class for weighs retrieval
    _ffn1(),
    _ffn2(),
    _pairwise_weight(),
    _cat(0),
    _matmul(),
    _in_dim1(in_dim1),
    _in_dim2(in_dim2),
    _output_dim(output_dim)
{
}

void DeepBiaffine::operator()(const StorageView& x1,
                              const StorageView& x2,
                              StorageView& output) const
{
  StorageView h1(x1.dtype(), x1.device()), h2(x2.dtype(), x2.device());
  // TODO check the operator() for the class
  (*_ffn1)(x1, h1);
  (*_ffn2)(x2, h2);

  // make interactions
  StorageView g1(h1.dtype(), h1.device()), g2(h2.dtype(), h2.device());
  make_interactions(h1, g1);
  make_interactions(h2, g2);

  Shape g1_size(g1.shape()), g2_size(g2.shape());

  StorageView g1_w(g1.dtype(), g1.device());
  g1 = g1.view(-1, g1_size[g1_size.size()]);
  _matmul(g1, pairwise(), g1_w);
  StorageView g2_t(g2.dtype(), g2.device());
  ops::Transpose({1,2})(g2, g2_t);

  StorageView g1_w_g2(g2.dtype(), g2.device());
  Shape g_w_shape{g1_size[0], g1_size[1]*_output_dim, g2_size[2]};
  _matmul.bmm(g1_w.view(g_w_shape), g2_t, g1_w_g2);
  ops::Transpose({2,3})(g1_w_g2.view(g_w_shape));
  output = g1_w_g2;
}

void DeepBiaffine::make_interactions(const StorageView& h,
                                     StorageView& g) const
{
  // TODO what is the goal of this call?
  // g(h.dtype(), h.device());
  StorageView h_ones(h.dtype(), h.device());
  h_ones.fill(1);
  ops::Concat(h.shape().size() - 1)({&h, &h_ones}, g);
}

/**
* This class is based on the trankit library in python
*/
PosdepDecoder::PosdepDecoder(const models::Model& model,
                             const std::string& scope,
                             const ops::ActivationType activation_type) :
    Decoder(model.device()),
    _upos_embeddings(model, scope),
    _upos_ffn(build_optional_layer<Dense>(model, scope + "/_upos_ffn")),
    _xpos_ffn(build_optional_layer<Dense>(model, scope + "/_xpos_ffn")),
    _feats_ffn(build_optional_layer<Dense>(model, scope + "/_feats_ffn")),
    _down_project(build_optional_layer<Dense>(model, scope + "/_down_project")),
    _unlabeled(
      model.get_variable(scope+"/unlabeled"),
               scope+"/unlabeled",
               activation_type),
    _deprel(model.get_variable(scope+"/deprel"),
            scope+'/deprel',
            activation_type)
{
}

void PosdepDecoder::operator()(
  const StorageView& batch,
  const StorageView& word_reprs,
  const StorageView& cls_reprs,
  StorageView& preds_output,
  StorageView& deps_idxs_output
) const
{
}

void PosdepDecoder::predict(
  const StorageView batch,
  const StorageView word_reprs,
  const StorageView cls_reprs,
  StorageView& predicted_upos,
  StorageView& predicted_xpos,
  StorageView& predicted_feats,
  StorageView& dep_preds
) const
{
  // upos
  StorageView upos_score(word_reprs.dtype(), word_reprs.device());
  predicted_upos.move_to(word_reprs.device(), word_reprs.dtype()
                         );
  (*_upos_ffn)(word_reprs, upos_score);

}

StorageView DeepBiaffine::pairwise() const
{
  return _pairwise_weight.view(-1, (_in_dim2+1)*_output_dim);
}

} // namespace ctranslate2::layers
