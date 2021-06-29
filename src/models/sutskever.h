#pragma once

#include "marian.h"

#include "layers/constructors.h"
#include "rnn/constructors.h"

namespace marian {

class DecoderSutskever : public DecoderBase {
private:
  Ptr<rnn::RNN> rnn_;
  Ptr<mlp::MLP> output_;

  Ptr<rnn::RNN> constructDecoderRNN(Ptr<ExpressionGraph> graph,
                                    Ptr<DecoderState> state) {
    float dropoutRnn = inference_ ? 0 : opt<float>("dropout-rnn");
    auto rnn = rnn::rnn()                                          //
        ("type", opt<std::string>("dec-cell"))                     //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-rnn"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        ("nematus-normalization",
         options_->has("original-type")
             && opt<std::string>("original-type") == "nematus")  //
        ("skip", opt<bool>("skip"));

    size_t decoderLayers = opt<size_t>("dec-depth");
    size_t decoderBaseDepth = opt<size_t>("dec-cell-base-depth");
    size_t decoderHighDepth = opt<size_t>("dec-cell-high-depth");

    // setting up conditional (transitional) cell
    auto baseCell = rnn::stacked_cell();
    for(size_t i = 1; i <= decoderBaseDepth; ++i) {
      bool transition = (i > 2);
      auto paramPrefix = prefix_ + "_cell" + std::to_string(i);
      baseCell.push_back(rnn::cell()              //
                         ("prefix", paramPrefix)  //
                         ("final", i > 1)         //
                         ("transition", transition));
    }
    // Add cell to RNN (first layer)
    rnn.push_back(baseCell);

    // Add more cells to RNN (stacked RNN)
    for(size_t i = 2; i <= decoderLayers; ++i) {
      // deep transition
      auto highCell = rnn::stacked_cell();

      for(size_t j = 1; j <= decoderHighDepth; j++) {
        auto paramPrefix
            = prefix_ + "_l" + std::to_string(i) + "_cell" + std::to_string(j);
        highCell.push_back(rnn::cell()("prefix", paramPrefix));
      }

      // Add cell to RNN (more layers)
      rnn.push_back(highCell);
    }

    return rnn.construct(graph);
  }

public:
  DecoderSutskever(Ptr<ExpressionGraph> graph, Ptr<Options> options) :
    DecoderBase(graph, options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {

    std::vector<Expr> meanContexts;
    for(auto& encState : encStates) {
      // average the source context weighted by the batch mask
      // this will remove padded zeros from the average
      meanContexts.push_back(weighted_average(
          encState->getContext(), encState->getMask(), /*axis =*/ -3));
      //meanContexts.push_back(slice(encState->getContext(), /*axis =*/-3, 0));
    }

    Expr start;
    if(!meanContexts.empty()) {
      // apply single layer network to mean to map into decoder space
      auto mlp = mlp::mlp().push_back(
          mlp::dense()                                               //
          ("prefix", prefix_ + "_ff_state")                          //
          ("dim", opt<int>("dim-rnn"))                               //
          ("activation", (int)mlp::act::tanh)                        //
          ("layer-normalization", opt<bool>("layer-normalization"))  //
          ("nematus-normalization",
          options_->has("original-type")
               && opt<std::string>("original-type") == "nematus")  //
      )
      .construct(graph);

      start = mlp->apply(meanContexts);
    } else {
      int dimBatch = (int)batch->size();
      int dimRnn = opt<int>("dim-rnn");

      start = graph->constant({dimBatch, dimRnn}, inits::zeros());
    }

    rnn::States startStates(opt<size_t>("dec-depth"), {start, start});
    return New<DecoderState>(startStates, Logits(), encStates, batch);
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {

    auto embeddings = state->getTargetHistoryEmbeddings();

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      embeddings = dropout(embeddings, dropoutTrg, {trgWords, 1, 1});
    }

    if(!rnn_)
      rnn_ = constructDecoderRNN(graph, state);

    // apply RNN to embeddings, initialized with encoder context mapped into
    // decoder space
    auto decoderContext = rnn_->transduce(embeddings, state->getStates());

    // retrieve the last state per layer. They are required during translation
    // in order to continue decoding for the next word
    rnn::States decoderStates = rnn_->lastCellStates();

    if(!output_) {
      // construct deep output multi-layer network layer-wise
      auto hidden = mlp::dense()                                     //
          ("prefix", prefix_ + "_ff_logit_l1")                       //
          ("dim", opt<int>("dim-emb"))                               //
          ("activation", (int)mlp::act::tanh)                        //
          ("layer-normalization", opt<bool>("layer-normalization"))  //
          ("nematus-normalization",
            options_->has("original-type")
                && opt<std::string>("original-type") == "nematus");

      int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

      auto last = mlp::output()                 //
          ("prefix", prefix_ + "_ff_logit_l2")  //
          ("dim", dimTrgVoc);

      if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
        std::string tiedPrefix = prefix_ + "_Wemb";
        if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
          tiedPrefix = "Wemb";
        last.tieTransposed(tiedPrefix);
      }

      if(shortlist_)
        last.setShortlist(shortlist_);

      // assemble layers into MLP and apply to embeddings, decoder context and
      // aligned source context
      output_ = mlp::mlp()              //
                    .push_back(hidden)  //
                    .push_back(last)
                    .construct(graph);
    }

    Logits logits = output_->applyAsLogits({embeddings, decoderContext});

    // return unnormalized(!) probabilities
    auto nextState = New<DecoderState>(
        decoderStates, logits, state->getEncoderStates(), state->getBatch());

    // Advance current target token position by one
    nextState->setPosition(state->getPosition() + 1);
    return nextState;
  }

  virtual void clear() override {
    rnn_ = nullptr;
    output_ = nullptr;
  }
};
} // namespace marian
