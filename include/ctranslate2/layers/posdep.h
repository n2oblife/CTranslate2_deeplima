#pragma once

#include "ctranslate2/layers/common.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/layers/transformer.h"
#include "ctranslate2/layers/decoder.h"
#include "ctranslate2/ops/matmul.h"
#include "ctranslate2/ops/activation.h"
#include "ctranslate2/padder.h"

namespace ctranslate2::layers {

class DeepBiaffine : public Layer
{
public:
  DeepBiaffine(
      const models::Model& model,
      const std::string& scope,
      const int in_dim1,
      const int in_dim2,
      const int output_dim,
      const ops::ActivationType activation_type = ops::ActivationType::ReLU
  );

  ~DeepBiaffine() = default;

  void operator()(
      const StorageView& x1,
      const StorageView& x2,
      StorageView& output
  ) const;

  void make_interactions(const StorageView& h,
                          StorageView& g) const;

  StorageView pairwise() const;

private:
  const ops::ActivationType _activation;
  const std::unique_ptr<const Dense> _ffn1;
  const std::unique_ptr<const Dense> _ffn2;
  const StorageView _pairwise_weight;
  ops::Concat _cat;
  const ops::MatMul _matmul;
  const dim_t _in_dim1;
  const dim_t _in_dim2;
  const dim_t _output_dim;
};

class PosdepDecoder : public Decoder // or Layer
{
public:
  PosdepDecoder(
      const models::Model& model,
      const std::string& scope,
      const ops::ActivationType activation_type = ops::ActivationType::ReLU
  );

  ~PosdepDecoder() = default;

  void operator()( // add the parameters needed for feedforward
      const StorageView& batch,
      const StorageView& word_reprs,
      const StorageView& cls_reprs,
      StorageView& preds_output,
      StorageView& deps_idxs_output
  ) const;

  void predict(
      const StorageView batch,
      const StorageView word_reprs,
      const StorageView cls_reprs,
      StorageView& predicted_upos,
      StorageView& predicted_xpos,
      StorageView& predicted_feats,
      StorageView& dep_preds
  ) const;

  void preds_to_cpu(const StorageView& prediction);

  void deprel_to_deps(
      const StorageView& predicted_dep,
      const int& batch_size,
      StorageView& output
  );

  void padding_deps(
      const StorageView& dep,
      StorageView& padded_dep
  );


private:
  const Embeddings _upos_embeddings;
  const std::unique_ptr<const Dense> _upos_ffn;
  const std::unique_ptr<const Dense> _xpos_ffn;
  const std::unique_ptr<const Dense> _feats_ffn;
  const std::unique_ptr<const Dense> _down_project;
  const std::unique_ptr<const DeepBiaffine> _unlabeled;
  const std::unique_ptr<const DeepBiaffine> _deprel;
};

class DeepLayer : public Layer
{
public:
  DeepLayer(
    const models::Model& model,
    const std::string& scope
  );

  void operator()(

  );

private: // build dynamic parameter
};

} // namespace ctranslate2::layers

