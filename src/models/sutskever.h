#pragma once

#include "marian.h"

#include "layers/constructors.h"
#include "rnn/constructors.h"

namespace marian {

class EncoderSutskever : public EncoderBase {
public:
  EncoderSutskever(Ptr<Options> options) : EncoderBase(options) {}

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch) override {
    // create source embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];
    auto embeddings = embedding()
                      ("dimVocab", dimVoc)
                      ("dimEmb", opt<int>("dim-emb"))
                      ("prefix", prefix_ + "_Wemb")
                          .construct(graph);

    // select embeddings that occur in the batch
    Expr batchEmbeddings, batchMask; std::tie
    (batchEmbeddings, batchMask) = embeddings->apply((*batch)[batchIndex_]);

    // backward RNN for encoding
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnnBw = rnn::rnn()
                 ("type", "lstm")
                 ("prefix", prefix_)
                 ("direction", rnn::dir::backward)
                 ("dimInput", opt<int>("dim-emb"))
                 ("dimState", opt<int>("dim-rnn"))
                 ("dropout", dropoutRnn)
                 ("layer-normalization", opt<bool>("layer-normalization"))
                     .push_back(rnn::cell())
                     .construct(graph);

    auto context = rnnBw->transduce(batchEmbeddings, batchMask);

    return New<EncoderState>(context, batchMask, batch);
  }

  virtual void clear() override {}
};

class DecoderSutskever : public DecoderBase {
public:
  DecoderSutskever(Ptr<Options> options) : DecoderBase(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {
    // Use last encoded word as start state
    auto start = slice(encStates[0]->getContext(), /*axis=*/0, 0);

    rnn::States startStates({ {start, start} });
    return New<DecoderState>(startStates, nullptr, encStates, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {
    auto embeddings = state->getTargetEmbeddings();

    // forward RNN for decoder
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn()
               ("type", "lstm")
               ("prefix", prefix_)
               ("dimInput", opt<int>("dim-emb"))
               ("dimState", opt<int>("dim-rnn"))
               ("dropout", dropoutRnn)
               ("layer-normalization", opt<bool>("layer-normalization"))
                   .push_back(rnn::cell())
                   .construct(graph);

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn->lastCellStates();

    // construct deep output multi-layer network layer-wise
    auto layer1 = mlp::dense()
                  ("prefix", prefix_ + "_ff_logit_l1")
                  ("dim", opt<int>("dim-emb"))
                  ("activation", mlp::act::tanh);
    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs").back();
    auto layer2 = mlp::dense()
                  ("prefix", prefix_ + "_ff_logit_l2")
                  ("dim", dimTrgVoc);

    // assemble layers into MLP and apply to embeddings, decoder context and
    // aligned source context
    auto logits = mlp::mlp()
                      .push_back(layer1)
                      .push_back(layer2)
                      .construct(graph)
                  ->apply(embeddings, decoderContext);

    // return unnormalized(!) probabilities
    return New<DecoderState>(decoderStates, logits, state->getEncoderStates(), state->getBatch());
  }

  virtual void clear() override {}
};

} // namespace marian
