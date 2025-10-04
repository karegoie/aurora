#include "rl_agent.h"
#include "fasta.h"
#include "cwt.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

// Transformer Actor-Critic Network
struct TransformerActorCriticImpl : torch::nn::Module {
    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::Linear actor_head{nullptr};
    torch::nn::Linear critic_head{nullptr};
    
    int d_model;
    int num_labels;
    
    TransformerActorCriticImpl(int d_model_, int nhead, int num_layers, 
                               int dim_feedforward, int num_labels_)
        : d_model(d_model_), num_labels(num_labels_) {
        
        // Transformer encoder
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(0.1)
        );
        
        encoder = torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, num_layers)
        );
        
        // Actor head (policy)
        actor_head = register_module("actor_head", 
                                     torch::nn::Linear(d_model, num_labels));
        
        // Critic head (value)
        critic_head = register_module("critic_head", 
                                      torch::nn::Linear(d_model, 1));
        
        register_module("encoder", encoder);
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        // x shape: [seq_len, batch, d_model]
        auto encoded = encoder->forward(x);
        
        // Get last time step
        auto last_hidden = encoded.index({-1}); // [batch, d_model]
        
        // Actor output (action probabilities)
        auto action_logits = actor_head->forward(last_hidden);
        auto action_probs = torch::softmax(action_logits, -1);
        
        // Critic output (state value)
        auto value = critic_head->forward(last_hidden);
        
        return {action_probs, value};
    }
};

TORCH_MODULE(TransformerActorCritic);

// RL Environment
class RLEnvironment {
private:
    const CWTFeatures* features;
    const LabelType* gold_labels;
    size_t seq_length;
    size_t current_pos;
    int window_size;
    LabelType prev_action;
    
public:
    RLEnvironment(const CWTFeatures* feat, const LabelType* labels, 
                  size_t length, int window)
        : features(feat), gold_labels(labels), seq_length(length),
          current_pos(0), window_size(window), prev_action(LABEL_INTERGENIC) {}
    
    void reset() {
        current_pos = 0;
        prev_action = LABEL_INTERGENIC;
    }
    
    torch::Tensor getState() {
        // Create state tensor: window of CWT features + previous action
        int start_pos = std::max(0, (int)current_pos - window_size / 2);
        int end_pos = std::min((int)seq_length, start_pos + window_size);
        
        std::vector<float> state_vec;
        
        // Add CWT features for window
        for (int pos = start_pos; pos < end_pos; pos++) {
            for (size_t scale = 0; scale < features->num_scales; scale++) {
                cplx_t val = features->data[scale][pos];
                state_vec.push_back(std::abs(val)); // Magnitude
            }
        }
        
        // Pad if necessary
        while ((int)state_vec.size() < window_size * (int)features->num_scales) {
            state_vec.push_back(0.0f);
        }
        
        // Add previous action as one-hot
        for (int i = 0; i < NUM_LABELS; i++) {
            state_vec.push_back(i == (int)prev_action ? 1.0f : 0.0f);
        }
        
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        return torch::from_blob(state_vec.data(), {(long)state_vec.size()}, options).clone();
    }
    
    std::pair<double, bool> step(LabelType action) {
        // Calculate reward
        double reward = (action == gold_labels[current_pos]) ? 1.0 : -1.0;
        
        // Update state
        prev_action = action;
        current_pos++;
        
        // Check if done
        bool done = (current_pos >= seq_length);
        
        return {reward, done};
    }
    
    bool isDone() const {
        return current_pos >= seq_length;
    }
    
    size_t getPosition() const {
        return current_pos;
    }
};

// PPO Training Function (simplified)
static void train_ppo(TransformerActorCritic& model, RLEnvironment& env,
                      const AuroraConfig* config) {
    torch::optim::Adam optimizer(model->parameters(), 
                                 torch::optim::AdamOptions(config->learning_rate));
    
    for (int epoch = 0; epoch < config->num_epochs; epoch++) {
        env.reset();
        
        std::vector<torch::Tensor> states;
        std::vector<int> actions;
        std::vector<double> rewards;
        std::vector<double> values;
        
        // Collect trajectory
        while (!env.isDone()) {
            auto state = env.getState();
            states.push_back(state);
            
            // Forward pass
            auto state_input = state.unsqueeze(0).unsqueeze(0); // [1, 1, state_dim]
            auto [action_probs, value] = model->forward(state_input);
            
            // Sample action
            auto dist = torch::multinomial(action_probs, 1);
            int action = dist.item<int>();
            
            actions.push_back(action);
            values.push_back(value.item<double>());
            
            // Take step
            auto [reward, done] = env.step((LabelType)action);
            rewards.push_back(reward);
        }
        
        // Compute returns and advantages
        std::vector<double> returns(rewards.size());
        double running_return = 0.0;
        for (int t = (int)rewards.size() - 1; t >= 0; t--) {
            running_return = rewards[t] + config->gamma * running_return;
            returns[t] = running_return;
        }
        
        // Update policy
        optimizer.zero_grad();
        double total_loss = 0.0;
        
        for (size_t t = 0; t < states.size(); t++) {
            auto state_input = states[t].unsqueeze(0).unsqueeze(0);
            auto [action_probs, value] = model->forward(state_input);
            
            // Compute advantage
            double advantage = returns[t] - values[t];
            
            // Actor loss (policy gradient)
            auto log_prob = torch::log(action_probs.index({0, actions[t]}));
            auto actor_loss = -log_prob * advantage;
            
            // Critic loss (value function)
            auto target_value = torch::tensor(returns[t], torch::kFloat32);
            auto critic_loss = torch::mse_loss(value, target_value);
            
            auto loss = actor_loss + 0.5 * critic_loss;
            total_loss += loss.item<double>();
            
            loss.backward();
        }
        
        optimizer.step();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " 
                     << total_loss / states.size() << std::endl;
        }
    }
}

// C interface implementation
extern "C" {

int run_training(const char *fasta_file, const char *gff_file, 
                 const AuroraConfig *config) {
    try {
        std::cout << "Starting training..." << std::endl;
        
        // Parse FASTA file
        FastaData fasta_data;
        if (parse_fasta(fasta_file, &fasta_data) != 0) {
            std::cerr << "Failed to parse FASTA file" << std::endl;
            return -1;
        }
        
        if (fasta_data.num_entries == 0) {
            std::cerr << "No sequences found in FASTA file" << std::endl;
            free_fasta_data(&fasta_data);
            return -1;
        }
        
        // Use first sequence for training
        FastaEntry* entry = &fasta_data.entries[0];
        std::cout << "Training on sequence: " << entry->header 
                 << " (length: " << entry->length << ")" << std::endl;
        
        // Convert DNA to complex
    cplx_t* complex_seq = dna_to_complex(entry->sequence, entry->length);
        if (!complex_seq) {
            free_fasta_data(&fasta_data);
            return -1;
        }
        
        // Compute CWT
        std::cout << "Computing CWT features..." << std::endl;
        CWTFeatures* features = compute_cwt(complex_seq, entry->length,
                                           config->num_scales, 
                                           config->min_scale, 
                                           config->max_scale);
        free(complex_seq);
        
        if (!features) {
            free_fasta_data(&fasta_data);
            return -1;
        }
        
        // Parse GFF labels
        LabelType* labels = nullptr;
        if (parse_gff_labels(gff_file, entry->header, entry->length, &labels) != 0) {
            std::cerr << "Failed to parse GFF file" << std::endl;
            free_cwt_features(features);
            free_fasta_data(&fasta_data);
            return -1;
        }
        
        // Create environment
        RLEnvironment env(features, labels, entry->length, config->window_size);
        
    // Create model
    (void)config; // keep config referenced to avoid unused warnings in some builds
    TransformerActorCritic model(config->d_model, config->nhead,
                                    config->num_encoder_layers,
                                    config->dim_feedforward, NUM_LABELS);
        
        // Train
        std::cout << "Training model..." << std::endl;
        train_ppo(model, env, config);
        
        // Save model
        std::cout << "Saving model to " << config->model_output << std::endl;
        torch::save(model, config->model_output);
        
        // Cleanup
        free(labels);
        free_cwt_features(features);
        free_fasta_data(&fasta_data);
        
        std::cout << "Training completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return -1;
    }
}

int run_inference(const char *model_file, const CWTFeatures *features,
                  LabelType **predictions, size_t *num_predictions) {
    try {
        std::cout << "Loading model from " << model_file << std::endl;
        
        // Create model (parameters will be loaded)
        TransformerActorCritic model(256, 8, 6, 1024, NUM_LABELS);
        torch::load(model, model_file);
        model->eval();
        
        // Allocate predictions
        *num_predictions = features->length;
        *predictions = (LabelType*)malloc(features->length * sizeof(LabelType));
        if (!*predictions) {
            return -1;
        }
        
        // Simple greedy prediction (not using environment)
        for (size_t pos = 0; pos < features->length; pos++) {
            // Create simple state (just CWT at current position)
            std::vector<float> state_vec;
            for (size_t scale = 0; scale < features->num_scales; scale++) {
                cplx_t val = features->data[scale][pos];
                state_vec.push_back(std::abs(val));
            }
            
            // Pad to expected size
            while ((int)state_vec.size() < 256) {
                state_vec.push_back(0.0f);
            }
            
            auto state = torch::from_blob(state_vec.data(), 
                                         {1, 1, (long)state_vec.size()}, 
                                         torch::kFloat32).clone();
            
            // Predict
            torch::NoGradGuard no_grad;
            auto [action_probs, value] = model->forward(state);
            auto action = torch::argmax(action_probs, -1).item<int>();
            
            (*predictions)[pos] = (LabelType)action;
        }
        
        std::cout << "Inference completed" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return -1;
    }
}

} // extern "C"
