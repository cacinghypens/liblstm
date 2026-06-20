# LSTM-VAE C++ Library

A C++ implementation of LSTM Variational Autoencoder for temporal anomaly detection using PyTorch C++ (LibTorch).

## Overview

This library provides a complete LSTM-VAE implementation in C++ with the following features:

- **LSTM Encoder**: Bidirectional LSTM encoder that outputs latent distribution parameters (mu, logvar)
- **LSTM Decoder**: LSTM decoder that reconstructs sequences from latent vectors  
- **VAE Framework**: Reconstruction loss (MSE) + KL divergence with optional annealing
- **Event Prediction**: Multi-class and per-horizon binary event prediction heads
- **Anomaly Detection**: Combined reconstruction error and latent space anomaly scoring
- **Class Imbalance Handling**: Configurable positive weights for rare events

## Files

```
lstm_vae/
├── CMakeLists.txt          # Build configuration
├── LSTMEncoder.h           # Encoder header
├── LSTMEncoder.cpp         # Encoder implementation
├── LSTMDecoder.h           # Decoder header
├── LSTMDecoder.cpp         # Decoder implementation
├── LSTMVAE.h               # Main VAE model header
├── LSTMVAE.cpp             # Main VAE model implementation
└── examples/
    └── example.cpp         # Usage example
```

## Requirements

- CMake >= 3.18
- C++17 compatible compiler
- LibTorch (PyTorch C++)

## Building

### 1. Download LibTorch

Download LibTorch from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/):

```bash
# CPU version
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cpu.zip

# Or CUDA version (example for CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu118.zip
```

### 2. Build the Library

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(nproc)
```

### 3. Install (Optional)

```bash
sudo make install
```

## Usage

### Basic Example

```cpp
#include <torch/torch.h>
#include "LSTMVAE.h"

int main() {
    // Create model
    lstm_vae::LSTMVAEOptions options;
    options.input_dim(128)
           .hidden_dim(256)
           .num_layers(3)
           .latent_dim(64)
           .dropout(0.2)
           .bidirectional(true);

    auto model = lstm_vae::LSTMVAE(options);
    
    // Create input (batch_size=32, seq_len=100, input_dim=128)
    auto x = torch::randn({32, 100, 128});
    
    // Forward pass
    auto outputs = model->forward(x);
    
    // Get anomaly scores
    auto scores = model->compute_anomaly_score(x);
    
    // Predict events
    auto [probs, predictions] = model->predict_event(x);
    
    return 0;
}
```

### Training Loop

```cpp
auto model = lstm_vae::LSTMVAE(options);
auto optimizer = torch::optim::AdamW(model->parameters(), 
                                      torch::optim::AdamWOptions(0.001));

for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (const auto& batch : dataloader) {
        auto x = batch.data.to(device);
        auto labels = batch.labels;  // std::map<std::string, torch::Tensor>
        
        optimizer.zero_grad();
        
        auto loss_output = model->compute_loss(x, labels, global_step);
        loss_output.total_loss.backward();
        
        optimizer.step();
        global_step++;
    }
}
```

### Using Class Weights for Imbalanced Data

```cpp
std::map<std::string, double> pos_weights;
pos_weights["15"] = 10.0;   // Weight for 15-minute horizon
pos_weights["60"] = 8.0;    // Weight for 60-minute horizon
pos_weights["240"] = 5.0;   // Weight for 240-minute horizon

lstm_vae::LSTMVAEOptions options;
options.input_dim(128)
       .pos_weights(pos_weights);  // Set class weights
```

## API Reference

### LSTMEncoder

```cpp
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
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    // Returns: (mu, logvar, lstm_out)
};
```

### LSTMDecoder

```cpp
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
    
    torch::Tensor forward(torch::Tensor z, std::optional<int64_t> seq_len = std::nullopt);
    // Returns: reconstructed sequence
};
```

### LSTMVAE

```cpp
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
        std::vector<int64_t> prediction_horizons = {15, 60, 240, 480};
        std::map<std::string, double> pos_weights;
    };
    
    // Forward pass
    VAEOutput forward(torch::Tensor x);
    
    // Compute loss
    VAELoss compute_loss(torch::Tensor x, 
                         const std::map<std::string, torch::Tensor>& labels,
                         int64_t global_step = 0);
    
    // Anomaly detection
    torch::Tensor compute_anomaly_score(torch::Tensor x, float threshold_percentile = 95.0);
    
    // Event prediction
    std::pair<torch::Tensor, torch::Tensor> predict_event(torch::Tensor x);
    
    // Latent embeddings
    torch::Tensor get_latent_embedding(torch::Tensor x);
};
```

## Output Structures

```cpp
struct VAEOutput {
    torch::Tensor x_reconstructed;      // (batch, seq_len, input_dim)
    torch::Tensor mu;                   // (batch, latent_dim)
    torch::Tensor logvar;               // (batch, latent_dim)
    torch::Tensor z;                    // (batch, latent_dim)
    torch::Tensor event_logits;         // (batch, 3)
    torch::Tensor event_probs;          // (batch, 3)
    std::map<std::string, torch::Tensor> horizon_event_logits;
    std::map<std::string, torch::Tensor> horizon_event_probs;
    torch::Tensor anomaly_score;        // (batch,)
};

struct VAELoss {
    torch::Tensor total_loss;
    torch::Tensor reconstruction_loss;
    torch::Tensor kl_loss;
    torch::Tensor event_loss;
};
```

## License

MIT License
