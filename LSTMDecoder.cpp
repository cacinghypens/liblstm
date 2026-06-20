#include "LSTMDecoder.h"

namespace lstm_vae {

LSTMDecoderImpl::LSTMDecoderImpl(const Options& options) : options_(options) {
    // Project latent vector to initial hidden state
    hidden_proj_ = torch::nn::Linear(options_.latent_dim, options_.hidden_dim * options_.num_layers);
    cell_proj_ = torch::nn::Linear(options_.latent_dim, options_.hidden_dim * options_.num_layers);
    register_module("hidden_proj", hidden_proj_);
    register_module("cell_proj", cell_proj_);
    
    // Create LSTM layer
    lstm_ = torch::nn::LSTM(
        torch::nn::LSTMOptions(options_.latent_dim, options_.hidden_dim)
            .num_layers(options_.num_layers)
            .batch_first(true)
            .dropout(options_.num_layers > 1 ? options_.dropout : 0.0)
            .bidirectional(options_.bidirectional)
    );
    register_module("lstm", lstm_);
    
    // Output layer
    output_layer_ = torch::nn::Linear(options_.hidden_dim, options_.output_dim);
    register_module("output_layer", output_layer_);
    
    // Dropout
    dropout_ = torch::nn::Dropout(options_.dropout);
    register_module("dropout", dropout_);
}

LSTMDecoderImpl::LSTMDecoderImpl(int64_t latent_dim, int64_t hidden_dim, int64_t output_dim,
                                  int64_t num_layers, int64_t seq_len,
                                  double dropout, bool bidirectional)
    : LSTMDecoderImpl(Options{latent_dim, hidden_dim, output_dim, num_layers, seq_len, dropout, bidirectional}) {}

torch::Tensor LSTMDecoderImpl::forward(torch::Tensor z, std::optional<int64_t> seq_len) {
    int64_t actual_seq_len = seq_len.value_or(options_.seq_len);
    int64_t batch_size = z.size(0);
    
    // Initialize hidden states from latent vector
    // hidden_proj output: (batch, num_layers * hidden_dim)
    auto hidden = hidden_proj_->forward(z);
    hidden = hidden.view({batch_size, options_.num_layers, options_.hidden_dim});
    hidden = hidden.transpose(0, 1).contiguous();  // (num_layers, batch, hidden_dim)
    
    auto cell = cell_proj_->forward(z);
    cell = cell.view({batch_size, options_.num_layers, options_.hidden_dim});
    cell = cell.transpose(0, 1).contiguous();  // (num_layers, batch, hidden_dim)
    
    // Create sequence of latent vectors: (batch, seq_len, latent_dim)
    auto z_seq = z.unsqueeze(1).repeat({1, actual_seq_len, 1});
    
    // LSTM forward
    auto lstm_out_tuple = lstm_->forward(z_seq, std::make_tuple(hidden, cell));
    auto lstm_out = std::get<0>(lstm_out_tuple);
    
    // Output layer: (batch, seq_len, output_dim)
    auto reconstructed = output_layer_->forward(lstm_out);
    
    return reconstructed;
}

} // namespace lstm_vae
