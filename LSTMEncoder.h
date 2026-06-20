#ifndef LSTM_ENCODER_H
#define LSTM_ENCODER_H

#include <torch/torch.h>
#include <tuple>
#include <memory>

namespace lstm_vae {

/**
 * LSTM Encoder for VAE
 * 
 * Encodes input sequences into latent distribution parameters (mu, logvar)
 */
class LSTMEncoderImpl : public torch::nn::Module {
public:
    struct Options {
        int64_t input_dim{128};
        int64_t hidden_dim{256};
        int64_t num_layers{2};
        int64_t latent_dim{64};
        double dropout{0.2};
        bool bidirectional{true};
    };

    explicit LSTMEncoderImpl(const Options& options = Options());
    explicit LSTMEncoderImpl(int64_t input_dim, int64_t hidden_dim = 256, 
                             int64_t num_layers = 2, int64_t latent_dim = 64,
                             double dropout = 0.2, bool bidirectional = true);

    /**
     * Forward pass through encoder
     * 
     * @param x Input tensor of shape (batch_size, seq_len, input_dim)
     * @return Tuple of (mu, logvar, lstm_out)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);

    // Accessor methods
    int64_t getHiddenDim() const { return options_.hidden_dim; }
    int64_t getNumLayers() const { return options_.num_layers; }
    int64_t getLatentDim() const { return options_.latent_dim; }
    bool isBidirectional() const { return options_.bidirectional; }

private:
    Options options_;
    int64_t num_directions_;
    
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear mu_layer_{nullptr};
    torch::nn::Linear logvar_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(LSTMEncoder);

} // namespace lstm_vae

#endif // LSTM_ENCODER_H
