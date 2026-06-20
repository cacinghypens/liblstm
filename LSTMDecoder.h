#ifndef LSTM_DECODER_H
#define LSTM_DECODER_H

#include <torch/torch.h>
#include <memory>
#include <optional>

namespace lstm_vae {

/**
 * LSTM Decoder for VAE
 * 
 * Decodes latent vectors back into sequences
 */
class LSTMDecoderImpl : public torch::nn::Module {
public:
    struct Options {
        int64_t latent_dim{64};
        int64_t hidden_dim{256};
        int64_t output_dim{128};
        int64_t num_layers{2};
        int64_t seq_len{100};
        double dropout{0.2};
        bool bidirectional{false};
    };

    explicit LSTMDecoderImpl(const Options& options = Options());
    explicit LSTMDecoderImpl(int64_t latent_dim, int64_t hidden_dim, int64_t output_dim,
                             int64_t num_layers = 2, int64_t seq_len = 100,
                             double dropout = 0.2, bool bidirectional = false);

    /**
     * Forward pass through decoder
     * 
     * @param z Latent vector of shape (batch_size, latent_dim)
     * @param seq_len Sequence length (uses default if not provided)
     * @return Reconstructed sequence of shape (batch_size, seq_len, output_dim)
     */
    torch::Tensor forward(torch::Tensor z, std::optional<int64_t> seq_len = std::nullopt);

    // Accessor methods
    int64_t getHiddenDim() const { return options_.hidden_dim; }
    int64_t getNumLayers() const { return options_.num_layers; }
    int64_t getSeqLen() const { return options_.seq_len; }
    int64_t getLatentDim() const { return options_.latent_dim; }

private:
    Options options_;
    
    torch::nn::Linear hidden_proj_{nullptr};
    torch::nn::Linear cell_proj_{nullptr};
    torch::nn::LSTM lstm_{nullptr};
    torch::nn::Linear output_layer_{nullptr};
    torch::nn::Dropout dropout_{nullptr};
};

TORCH_MODULE(LSTMDecoder);

} // namespace lstm_vae

#endif // LSTM_DECODER_H
