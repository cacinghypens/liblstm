/**
 * Example usage of the LSTM-VAE library
 * 
 * This example demonstrates:
 * - Creating an LSTMVAE model
 * - Running forward pass
 * - Computing anomaly scores
 * - Predicting events
 */

#include <iostream>
#include <torch/torch.h>
#include "LSTMVAE.h"

int main() {
    // Set device (CUDA if available, otherwise CPU)
    torch::DeviceType device_type = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Using GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "CUDA not available. Using CPU." << std::endl;
    }
    torch::Device device(device_type);

    // Model configuration
    int64_t batch_size = 32;
    int64_t seq_len = 100;
    int64_t input_dim = 128;
    int64_t hidden_dim = 256;
    int64_t num_layers = 3;
    int64_t latent_dim = 64;

    // Create model
    lstm_vae::LSTMVAEOptions options;
    options.input_dim(input_dim)
           .hidden_dim(hidden_dim)
           .num_layers(num_layers)
           .latent_dim(latent_dim)
           .dropout(0.2)
           .bidirectional(true)
           .beta(0.001)
           .kl_annealing(true)
           .kl_annealing_steps(10000)
           .seq_len(seq_len);

    auto model = lstm_vae::LSTMVAE(options);
    model->to(device);
    model->train();

    std::cout << "Model created successfully!" << std::endl;
    std::cout << "Input dim: " << model->getInputDim() << std::endl;
    std::cout << "Hidden dim: " << model->getHiddenDim() << std::endl;
    std::cout << "Latent dim: " << model->getLatentDim() << std::endl;

    // Create dummy input data
    auto x = torch::randn({batch_size, seq_len, input_dim}, device);

    // Forward pass
    std::cout << "\nRunning forward pass..." << std::endl;
    auto outputs = model->forward(x);

    std::cout << "Reconstructed shape: " << outputs.x_reconstructed.sizes() << std::endl;
    std::cout << "Mu shape: " << outputs.mu.sizes() << std::endl;
    std::cout << "Logvar shape: " << outputs.logvar.sizes() << std::endl;
    std::cout << "Z shape: " << outputs.z.sizes() << std::endl;
    std::cout << "Event probs shape: " << outputs.event_probs.sizes() << std::endl;
    std::cout << "Anomaly score shape: " << outputs.anomaly_score.sizes() << std::endl;

    // Compute loss
    std::cout << "\nComputing loss..." << std::endl;
    std::map<std::string, torch::Tensor> labels;
    // Add dummy multi-class event labels (0, 1, or 2)
    labels["event"] = torch::randint(0, 3, {batch_size}, device).to(torch::kFloat);
    
    auto loss_output = model->compute_loss(x, labels, 0);
    std::cout << "Total loss: " << loss_output.total_loss.item<float>() << std::endl;
    std::cout << "Reconstruction loss: " << loss_output.reconstruction_loss.item<float>() << std::endl;
    std::cout << "KL loss: " << loss_output.kl_loss.item<float>() << std::endl;
    std::cout << "Event loss: " << loss_output.event_loss.item<float>() << std::endl;

    // Compute anomaly scores
    std::cout << "\nComputing anomaly scores..." << std::endl;
    model->eval();
    auto anomaly_scores = model->compute_anomaly_score(x);
    std::cout << "Anomaly scores range: [" 
              << anomaly_scores.min().item<float>() << ", "
              << anomaly_scores.max().item<float>() << "]" << std::endl;

    // Predict events
    std::cout << "\nPredicting events..." << std::endl;
    auto [event_probs, predictions] = model->predict_event(x);
    std::cout << "Predictions shape: " << predictions.sizes() << std::endl;
    std::cout << "First 5 predictions: " << predictions.slice(0, 0, 5) << std::endl;

    // Get latent embeddings
    std::cout << "\nGetting latent embeddings..." << std::endl;
    auto embeddings = model->get_latent_embedding(x);
    std::cout << "Embeddings shape: " << embeddings.sizes() << std::endl;

    // Test per-horizon predictions
    std::cout << "\nPer-horizon event predictions:" << std::endl;
    for (const auto& [horizon, probs] : outputs.horizon_event_probs) {
        std::cout << "  Horizon " << horizon << ": shape " << probs.sizes() << std::endl;
    }

    std::cout << "\nExample completed successfully!" << std::endl;

    return 0;
}
