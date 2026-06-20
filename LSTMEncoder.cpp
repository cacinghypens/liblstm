#include "LSTMEncoder.h"

namespace lstm_vae {

LSTMEncoderImpl::LSTMEncoderImpl(const Options& options) : options_(options) {
    num_directions_ = options_.bidirectional ? 2 : 1;
    
    // Create LSTM layer
    lstm_ = torch::nn::LSTM(
        torch::nn::LSTMOptions(options_.input_dim, options_.hidden_dim)
            .num_layers(options_.num_layers)
            .batch_first(true)
            .dropout(options_.num_layers > 1 ? options_.dropout : 0.0)
            .bidirectional(options_.bidirectional)
    );
    register_module("lstm", lstm_);
    
    // Latent space parameters
    int64_t encoder_output_dim = options_.hidden_dim * num_directions_;
    mu_layer_ = torch::nn::Linear(encoder_output_dim, options_.latent_dim);
    logvar_layer_ = torch::nn::Linear(encoder_output_dim, options_.latent_dim);
    register_module("mu_layer", mu_layer_);
    register_module("logvar_layer", logvar_layer_);
    
    // Dropout
    dropout_ = torch::nn::Dropout(options_.dropout);
    register_module("dropout", dropout_);
}

LSTMEncoderImpl::LSTMEncoderImpl(int64_t input_dim, int64_t hidden_dim, 
                                  int64_t num_layers, int64_t latent_dim,
                                  double dropout, bool bidirectional)
    : LSTMEncoderImpl(Options{input_dim, hidden_dim, num_layers, latent_dim, dropout, bidirectional}) {}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
LSTMEncoderImpl::forward(torch::Tensor x) {
    // LSTM forward
    auto lstm_out_tuple = lstm_->forward(x);
    auto lstm_out = std::get<0>(lstm_out_tuple);
    auto hidden = std::get<1>(lstm_out_tuple).squeeze(0);  // (num_layers * num_directions, batch, hidden)
    auto cell = std::get<2>(lstm_out_tuple).squeeze(0);
    
    // Extract last hidden state(s)
    // hidden shape: (num_layers * num_directions, batch, hidden_dim)
    // We need the last layer's hidden state(s)
    
    torch::Tensor last_hidden;
    if (options_.bidirectional) {
        // For bidirectional, concatenate the last forward and backward states
        // Shape: (2, batch, hidden_dim) for last layer
        auto num_layer_states = hidden.size(0);
        auto forward_last = hidden.index({-2, "..."});   // Second to last (last forward)
        auto backward_last = hidden.index({-1, "..."});  // Last (last backward)
        last_hidden = torch::cat({forward_last, backward_last}, -1);
    } else {
        last_hidden = hidden.index({-1, "..."});
    }
    
    // Apply dropout
    last_hidden = dropout_->forward(last_hidden);
    
    // Compute latent parameters
    auto mu = mu_layer_->forward(last_hidden);
    auto logvar = logvar_layer_->forward(last_hidden);
    
    return std::make_tuple(mu, logvar, lstm_out);
}

} // namespace lstm_vae
