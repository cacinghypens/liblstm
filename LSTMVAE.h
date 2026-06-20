#ifndef LSTM_VAE_H
#define LSTM_VAE_H

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include "LSTMEncoder.h"
#include "LSTMDecoder.h"

namespace lstm_vae {

// Default prediction horizons
const std::vector<int64_t> DEFAULT_PREDICTION_HORIZONS = {15, 60, 240, 480};

/**
 * Output structure for LSTMVAE forward pass
 */
struct VAEOutput {
    torch::Tensor x_reconstructed;
    torch::Tensor mu;
    torch::Tensor logvar;
    torch::Tensor z;
    torch::Tensor event_logits;
    torch::Tensor event_probs;
    std::map<std::string, torch::Tensor> horizon_event_logits;
    std::map<std::string, torch::Tensor> horizon_event_probs;
    torch::Tensor anomaly_score;
};

/**
 * Loss output structure
 */
struct VAELoss {
    torch::Tensor total_loss;
    torch::Tensor reconstruction_loss;
    torch::Tensor kl_loss;
    torch::Tensor event_loss;
};

/**
 * LSTM Variational Autoencoder for temporal anomaly detection
 * 
 * Combines LSTM encoder-decoder architecture with VAE framework
 * to learn a probabilistic latent representation of time series data.
 */
class LSTMVAEImpl : public torch::nn::Module {
public:
    struct Options {
        int64_t input_dim{128};
        int64_t hidden_dim{256};
        int64_t num_layers{3};
        int64_t latent_dim{64};
        double dropout{0.2};
        bool bidirectional{true};
        double beta{0.001};
        bool kl_annealing{true};
        int64_t kl_annealing_steps{10000};
        double learning_rate{0.001};
        double weight_decay{0.0001};
        int64_t seq_len{100};
        std::vector<int64_t> prediction_horizons = DEFAULT_PREDICTION_HORIZONS;
        std::map<std::string, double> pos_weights;  // Class imbalance weights
    };

    explicit LSTMVAEImpl(const Options& options = Options());
    
    /**
     * Constructor with individual parameters
     */
    explicit LSTMVAEImpl(int64_t input_dim, int64_t hidden_dim = 256, 
                         int64_t num_layers = 3, int64_t latent_dim = 64,
                         double dropout = 0.2, bool bidirectional = true,
                         double beta = 0.001, bool kl_annealing = true,
                         int64_t kl_annealing_steps = 10000,
                         double learning_rate = 0.001, double weight_decay = 0.0001,
                         int64_t seq_len = 100,
                         const std::vector<int64_t>& prediction_horizons = DEFAULT_PREDICTION_HORIZONS,
                         const std::map<std::string, double>& pos_weights = {});

    /**
     * Forward pass through the VAE
     * 
     * @param x Input sequence of shape (batch_size, seq_len, input_dim)
     * @return VAEOutput structure containing all outputs
     */
    VAEOutput forward(torch::Tensor x);

    /**
     * Compute loss for training
     * 
     * @param x Input sequence
     * @param labels Dictionary of labels by name
     * @param global_step Current training step (for KL annealing)
     * @return VAELoss structure containing loss components
     */
    VAELoss compute_loss(torch::Tensor x, 
                         const std::map<std::string, torch::Tensor>& labels,
                         int64_t global_step = 0);

    /**
     * Compute anomaly score for input sequences
     * 
     * @param x Input sequences
     * @param threshold_percentile Threshold percentile for normalization
     * @return Anomaly scores per sample
     */
    torch::Tensor compute_anomaly_score(torch::Tensor x, float threshold_percentile = 95.0);

    /**
     * Predict event probabilities
     * 
     * @param x Input sequences
     * @return Pair of (event_probs, predictions)
     */
    std::pair<torch::Tensor, torch::Tensor> predict_event(torch::Tensor x);

    /**
     * Get latent space embedding
     * 
     * @param x Input sequences
     * @return Latent embeddings (mu)
     */
    torch::Tensor get_latent_embedding(torch::Tensor x);

    /**
     * Reparameterization trick for VAE
     */
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor logvar);

    /**
     * Compute KL divergence
     */
    torch::Tensor compute_kl_divergence(torch::Tensor mu, torch::Tensor logvar);

    /**
     * Compute reconstruction loss (MSE)
     */
    torch::Tensor compute_reconstruction_loss(torch::Tensor x, torch::Tensor x_reconstructed);

    /**
     * Get beta value with optional KL annealing
     */
    double get_beta(int64_t step);

    // Accessor methods
    int64_t getInputDim() const { return options_.input_dim; }
    int64_t getHiddenDim() const { return options_.hidden_dim; }
    int64_t getLatentDim() const { return options_.latent_dim; }
    double getBeta() const { return options_.beta; }
    const std::vector<int64_t>& getPredictionHorizons() const { return options_.prediction_horizons; }

private:
    Options options_;
    int64_t current_step_{0};
    
    LSTMEncoder encoder_{nullptr};
    LSTMDecoder decoder_{nullptr};
    
    // Event prediction head (multi-class)
    torch::nn::Sequential event_head_{nullptr};
    
    // Per-horizon binary event heads
    torch::nn::ModuleDict horizon_event_heads_;
    
    // Anomaly score head
    torch::nn::Sequential anomaly_head_{nullptr};
    
    /**
     * Compute supervised event losses
     */
    torch::Tensor compute_event_loss(const VAEOutput& outputs,
                                     const std::map<std::string, torch::Tensor>& labels);
};

TORCH_MODULE(LSTMVAE);

} // namespace lstm_vae

#endif // LSTM_VAE_H
