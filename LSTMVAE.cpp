#include "LSTMVAE.h"
#include <algorithm>
#include <cmath>

namespace lstm_vae {

LSTMVAEImpl::LSTMVAEImpl(const Options& options) : options_(options) {
    // Create encoder
    encoder_ = LSTMEncoder(
        options_.input_dim,
        options_.hidden_dim,
        options_.num_layers,
        options_.latent_dim,
        options_.dropout,
        options_.bidirectional
    );
    register_module("encoder", encoder_);
    
    // Create decoder
    decoder_ = LSTMDecoder(
        options_.latent_dim,
        options_.hidden_dim,
        options_.input_dim,
        options_.num_layers,
        options_.seq_len,
        options_.dropout
    );
    register_module("decoder", decoder_);
    
    // Event prediction head (multi-class: normal, tsunami, volcano)
    event_head_ = torch::nn::Sequential(
        torch::nn::Linear(options_.latent_dim, options_.hidden_dim / 2),
        torch::nn::Functional(torch::relu),
        torch::nn::Dropout(options_.dropout),
        torch::nn::Linear(options_.hidden_dim / 2, 3)
    );
    register_module("event_head", event_head_);
    
    // Per-horizon binary event heads
    horizon_event_heads_ = torch::nn::ModuleDict();
    for (const auto& horizon : options_.prediction_horizons) {
        std::string horizon_str = std::to_string(horizon);
        auto head = torch::nn::Linear(options_.latent_dim, 2);
        horizon_event_heads_->register_module(horizon_str, head);
    }
    register_module("horizon_event_heads", horizon_event_heads_);
    
    // Anomaly score head
    anomaly_head_ = torch::nn::Sequential(
        torch::nn::Linear(options_.latent_dim, options_.hidden_dim / 4),
        torch::nn::Functional(torch::relu),
        torch::nn::Linear(options_.hidden_dim / 4, 1),
        torch::nn::Functional(torch::sigmoid)
    );
    register_module("anomaly_head", anomaly_head_);
}

LSTMVAEImpl::LSTMVAEImpl(int64_t input_dim, int64_t hidden_dim, int64_t num_layers,
                          int64_t latent_dim, double dropout, bool bidirectional,
                          double beta, bool kl_annealing, int64_t kl_annealing_steps,
                          double learning_rate, double weight_decay, int64_t seq_len,
                          const std::vector<int64_t>& prediction_horizons,
                          const std::map<std::string, double>& pos_weights)
    : LSTMVAEImpl(Options{input_dim, hidden_dim, num_layers, latent_dim, dropout,
                          bidirectional, beta, kl_annealing, kl_annealing_steps,
                          learning_rate, weight_decay, seq_len, prediction_horizons, pos_weights}) {}

torch::Tensor LSTMVAEImpl::reparameterize(torch::Tensor mu, torch::Tensor logvar) {
    auto std = torch::exp(0.5 * logvar);
    auto eps = torch::randn_like(std);
    return mu + eps * std;
}

torch::Tensor LSTMVAEImpl::compute_kl_divergence(torch::Tensor mu, torch::Tensor logvar) {
    // KL(q(z|x) || p(z)) where p(z) = N(0, I)
    auto kl = -0.5 * torch::mean(1 + logvar - mu.pow(2) - logvar.exp());
    return kl;
}

torch::Tensor LSTMVAEImpl::compute_reconstruction_loss(torch::Tensor x, torch::Tensor x_reconstructed) {
    return torch::nn::functional::mse_loss(x_reconstructed, x, 
                                           torch::nn::functional::MSELossFuncOptions().reduction(torch::kMean));
}

double LSTMVAEImpl::get_beta(int64_t step) {
    if (!options_.kl_annealing) {
        return options_.beta;
    }
    
    // Linear annealing
    return std::min(options_.beta, options_.beta * (static_cast<double>(step) / options_.kl_annealing_steps));
}

VAEOutput LSTMVAEImpl::forward(torch::Tensor x) {
    VAEOutput outputs;
    
    // Encode
    auto [mu, logvar, lstm_out] = encoder_->forward(x);
    outputs.mu = mu;
    outputs.logvar = logvar;
    
    // Reparameterize
    outputs.z = reparameterize(mu, logvar);
    
    // Decode
    outputs.x_reconstructed = decoder_->forward(outputs.z);
    
    // Event prediction (multi-class)
    auto event_logits = event_head_->forward(outputs.z);
    outputs.event_logits = event_logits;
    outputs.event_probs = torch::softmax(event_logits, 1);
    
    // Per-horizon event predictions
    for (const auto& horizon : options_.prediction_horizons) {
        std::string horizon_str = std::to_string(horizon);
        auto head = horizon_event_heads_->get_module(horizon_str);
        if (head) {
            auto logits = (*head)->forward(outputs.z);
            outputs.horizon_event_logits[horizon_str] = logits;
            outputs.horizon_event_probs[horizon_str] = torch::sigmoid(logits);
        }
    }
    
    // Anomaly score
    outputs.anomaly_score = anomaly_head_->forward(outputs.z).squeeze();
    
    return outputs;
}

torch::Tensor LSTMVAEImpl::compute_event_loss(const VAEOutput& outputs,
                                               const std::map<std::string, torch::Tensor>& labels) {
    std::vector<torch::Tensor> event_losses;
    auto device = outputs.event_logits.device();
    
    for (const auto& [label_name, label_tensor] : labels) {
        if (label_tensor.numel() == 0) {
            continue;
        }
        
        auto labels = label_tensor.to(device);
        
        // Parse horizon from label name (e.g., "tsunami_15m" -> horizon "15")
        std::string horizon;
        size_t last_underscore = label_name.rfind('_');
        if (last_underscore != std::string::npos && last_underscore < label_name.size() - 1) {
            horizon = label_name.substr(last_underscore + 1);
            // Remove 'm' suffix if present
            if (!horizon.empty() && horizon.back() == 'm') {
                horizon = horizon.substr(0, horizon.size() - 1);
            }
        }
        
        if (label_name.find("event_") == 0) {
            // Multi-class event loss
            auto loss = torch::nn::functional::cross_entropy(
                outputs.event_logits, labels.to(torch::kLong));
            event_losses.push_back(loss);
        } else if (outputs.horizon_event_logits.find(horizon) != outputs.horizon_event_logits.end()) {
            // Per-horizon binary loss
            auto logits = outputs.horizon_event_logits.at(horizon);
            
            // Get positive class weight if available
            torch::Tensor pos_weight;
            bool has_pos_weight = false;
            if (options_.pos_weights.find(horizon) != options_.pos_weights.end()) {
                pos_weight = torch::tensor({options_.pos_weights.at(horizon)}, 
                                          torch::device(device));
                has_pos_weight = true;
            }
            
            torch::Tensor loss;
            if (label_name.find("tsunami_") == 0) {
                // Tsunami: use first output of binary head
                auto loss_opts = torch::nn::functional::BCEWithLogitsLossFuncOptions();
                if (has_pos_weight) {
                    loss_opts = loss_opts.pos_weight(pos_weight);
                }
                loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    logits.index({".", 0}), labels.to(torch::kFloat), loss_opts);
                event_losses.push_back(loss);
            } else if (label_name.find("volcano_") == 0) {
                // Volcano: use second output of binary head
                auto loss_opts = torch::nn::functional::BCEWithLogitsLossFuncOptions();
                if (has_pos_weight) {
                    loss_opts = loss_opts.pos_weight(pos_weight);
                }
                loss = torch::nn::functional::binary_cross_entropy_with_logits(
                    logits.index({".", 1}), labels.to(torch::kFloat), loss_opts);
                event_losses.push_back(loss);
            }
        }
    }
    
    if (event_losses.empty()) {
        return torch::zeros({}, torch::device(device));
    }
    
    return torch::stack(event_losses).mean();
}

VAELoss LSTMVAEImpl::compute_loss(torch::Tensor x,
                                   const std::map<std::string, torch::Tensor>& labels,
                                   int64_t global_step) {
    VAELoss loss_output;
    
    // Forward pass
    auto outputs = forward(x);
    
    // Reconstruction loss
    loss_output.reconstruction_loss = compute_reconstruction_loss(x, outputs.x_reconstructed);
    
    // KL divergence
    loss_output.kl_loss = compute_kl_divergence(outputs.mu, outputs.logvar);
    
    // Apply KL annealing
    auto beta = get_beta(global_step);
    
    // Total VAE loss
    auto vae_loss = loss_output.reconstruction_loss + beta * loss_output.kl_loss;
    
    // Event prediction loss (if labels available)
    if (!labels.empty()) {
        loss_output.event_loss = compute_event_loss(outputs, labels);
    } else {
        loss_output.event_loss = torch::zeros_like(vae_loss);
    }
    
    // Total loss (weight event loss by 0.1)
    loss_output.total_loss = vae_loss + 0.1 * loss_output.event_loss;
    
    return loss_output;
}

torch::Tensor LSTMVAEImpl::compute_anomaly_score(torch::Tensor x, float threshold_percentile) {
    this->eval();
    torch::NoGradGuard no_grad;
    
    auto outputs = forward(x);
    
    // Reconstruction error per sample
    auto recon_error = torch::mean(torch::pow(x - outputs.x_reconstructed, 2), {1, 2});
    
    // Normalize reconstruction error
    auto recon_score = recon_error / (recon_error.max() + 1e-8);
    
    // Combine with anomaly head output
    auto anomaly_score = 0.5 * recon_score + 0.5 * outputs.anomaly_score;
    
    return anomaly_score;
}

std::pair<torch::Tensor, torch::Tensor> LSTMVAEImpl::predict_event(torch::Tensor x) {
    this->eval();
    torch::NoGradGuard no_grad;
    
    auto outputs = forward(x);
    auto event_probs = outputs.event_probs;
    auto predictions = event_probs.argmax(1);
    
    return std::make_pair(event_probs, predictions);
}

torch::Tensor LSTMVAEImpl::get_latent_embedding(torch::Tensor x) {
    this->eval();
    torch::NoGradGuard no_grad;
    
    auto [mu, logvar, lstm_out] = encoder_->forward(x);
    return mu;
}

} // namespace lstm_vae
