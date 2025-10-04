#include "rl_agent.h"
#include "fasta.h"
#include "cwt.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>
#include <random>

// Transformer Actor-Critic Network
struct TransformerActorCriticImpl : torch::nn::Module {
    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::Linear actor_head{nullptr};
    torch::nn::Linear critic_head{nullptr};
    torch::nn::Linear input_proj{nullptr};
    
    int d_model;
    int num_labels;
    
    TransformerActorCriticImpl(int d_model_, int nhead, int num_layers, 
                               int dim_feedforward, int num_labels_,
                               int input_dim_ = -1)
        : d_model(d_model_), num_labels(num_labels_) {

        int input_dim = (input_dim_ == -1) ? d_model_ : input_dim_;
        
        // Transformer encoder
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(dim_feedforward)
                .dropout(0.1)
        );
        
        encoder = torch::nn::TransformerEncoder(
            torch::nn::TransformerEncoderOptions(encoder_layer, num_layers)
        );
        
        /* Optional input projection: if the incoming state vector size differs
         * from d_model, project it to the model dimension. */
        if (input_dim != d_model_) {
            input_proj = register_module("input_proj",
                                         torch::nn::Linear(input_dim, d_model_));
        }
        
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
        // If input last-dim doesn't match d_model, project it.
        if (x.size(2) != d_model) {
            auto seq_len = x.size(0);
            auto batch = x.size(1);
            auto feat = x.size(2);
            auto x_flat = x.view({seq_len * batch, feat});
            auto proj = input_proj->forward(x_flat);
            x = proj.view({seq_len, batch, d_model});
        }

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

// === PPO Improved Multi-Sequence Training ===
namespace {
struct SequenceData {
    const CWTFeatures* features;
    const LabelType* labels;
    size_t length;
    std::string header;
};
}

static void train_ppo_multi(TransformerActorCritic& model,
                            const std::vector<SequenceData>& sequences,
                            const AuroraConfig* config) {
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(config->learning_rate));
    const double clip_eps = config->clip_epsilon;
    const double entropy_coef = 0.01;
    const int update_epochs = config->num_update_epochs > 0 ? config->num_update_epochs : 1;
    const int minibatch_size = (config->batch_size > 0) ? config->batch_size : 64;
    const double gae_lambda = 0.95;

    for (int epoch = 0; epoch < config->num_epochs; ++epoch) {
        std::vector<torch::Tensor> states; states.reserve(4096);
        std::vector<int> actions; actions.reserve(4096);
        std::vector<double> rewards, values, old_log_probs; rewards.reserve(4096); values.reserve(4096); old_log_probs.reserve(4096);
        std::vector<size_t> episode_lengths; episode_lengths.reserve(sequences.size());

        // Collect trajectories
        for (const auto& seq : sequences) {
            RLEnvironment env(seq.features, seq.labels, seq.length, config->window_size);
            env.reset(); size_t start = rewards.size();
            while (!env.isDone()) {
                auto s = env.getState();
                auto inp = s.unsqueeze(0).unsqueeze(0);
                auto [ap, v] = model->forward(inp);
                auto dist = torch::multinomial(ap, 1);
                int a = dist.item<int>();
                double val = v.item<double>();
                double lp = std::log(ap.index({0,a}).item<double>() + 1e-8);
                auto step_r = env.step((LabelType)a);
                states.push_back(s); actions.push_back(a); rewards.push_back(step_r.first); values.push_back(val); old_log_probs.push_back(lp);
            }
            episode_lengths.push_back(rewards.size() - start);
        }
        const size_t N = rewards.size(); if (!N) return;

        // GAE & returns
        std::vector<double> advantages(N), returns(N);
        size_t offset = 0;
        for (size_t epi=0; epi<episode_lengths.size(); ++epi) {
            size_t L = episode_lengths[epi]; double next_value = 0.0; double gae = 0.0;
            for (int t=(int)L-1; t>=0; --t) {
                double delta = rewards[offset+t] + config->gamma * next_value - values[offset+t];
                gae = delta + config->gamma * gae_lambda * gae;
                advantages[offset+t] = gae; returns[offset+t] = gae + values[offset+t];
                next_value = values[offset+t];
            }
            offset += L;
        }
        // Normalize advantages
        double mean_adv=0; for(double a:advantages) mean_adv+=a; mean_adv/= (double)N;
        double var_adv=0; for(double a:advantages){double d=a-mean_adv; var_adv+=d*d;} var_adv/= (double)N; double std_adv=std::sqrt(var_adv+1e-8);
        for(double &a:advantages) a=(a-mean_adv)/std_adv;

        // Indices for shuffling
        std::vector<size_t> idx(N); for(size_t i=0;i<N;++i) idx[i]=i;
        double last_policy_loss=0,last_value_loss=0,last_entropy=0,last_clip_frac=0; size_t total_clipped=0, total_elems=0;
        for(int up=0; up<update_epochs; ++up){
            std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device{}()));
            for(size_t start=0; start<N; start+=minibatch_size){
                size_t end = std::min(start+(size_t)minibatch_size, N);
                optimizer.zero_grad();
                torch::Tensor batch_loss = torch::zeros({1}); size_t clipped=0;
                double p_acc=0,v_acc=0,e_acc=0; size_t count=0;
                for(size_t ii=start; ii<end; ++ii){
                    size_t i = idx[ii];
                    auto inp = states[i].unsqueeze(0).unsqueeze(0);
                    auto [ap, val] = model->forward(inp);
                    auto logp = torch::log(ap.index({0,actions[i]}) + 1e-8);
                    double old_lp = old_log_probs[i];
                    auto ratio = torch::exp(logp - torch::tensor(old_lp));
                    auto adv = torch::tensor((float)advantages[i]);
                    auto surr1 = ratio * adv;
                    auto surr2 = torch::clamp(ratio, 1-clip_eps, 1+clip_eps) * adv;
                    auto actor_term = -torch::min(surr1, surr2);
                    if(ratio.item<double>()>1+clip_eps || ratio.item<double>()<1-clip_eps) clipped++;
                    auto ret_t = torch::tensor((float)returns[i]);
                    auto critic_term = torch::mse_loss(val, ret_t);
                    auto entropy = - (ap * torch::log(ap + 1e-8)).sum();
                    batch_loss = batch_loss + actor_term + 0.5*critic_term - entropy_coef*entropy;
                    p_acc += actor_term.item<double>(); v_acc += critic_term.item<double>(); e_acc += entropy.item<double>(); count++;
                }
                batch_loss.backward(); optimizer.step();
                total_clipped += clipped; total_elems += (end-start);
                last_policy_loss = p_acc / std::max<size_t>(1,count);
                last_value_loss = v_acc / std::max<size_t>(1,count);
                last_entropy = e_acc / std::max<size_t>(1,count);
            }
        }
        last_clip_frac = (total_elems? (double)total_clipped/(double)total_elems : 0.0);
        double avg_reward = 0; for(double r:rewards) avg_reward += r; avg_reward /= (double)N;
        std::cout << "Epoch "<<epoch
                  << " | steps="<<N
                  << " | avgReward="<<avg_reward
                  << " | policyLoss="<<last_policy_loss
                  << " | valueLoss="<<last_value_loss
                  << " | entropy="<<last_entropy
                  << " | clipFrac="<<last_clip_frac
                  << " | updates="<<update_epochs
                  << std::endl;
    }
}

// C interface implementation
extern "C" {

int run_training(const char *fasta_file, const char *gff_file,
                 const AuroraConfig *config) {
    try {
        std::cout << "Starting training (multi-sequence)..." << std::endl;
        // 1) FASTA 파싱
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
        std::cout << "Sequences loaded: " << fasta_data.num_entries << std::endl;

        // 2) 각 시퀀스에 대해 특징/레이블 준비
        struct LocalSeq { CWTFeatures* feat; LabelType* labels; size_t len; std::string header; };
        std::vector<LocalSeq> localSeqs; localSeqs.reserve(fasta_data.num_entries);
        for (size_t i = 0; i < fasta_data.num_entries; ++i) {
            FastaEntry* entry = &fasta_data.entries[i];
            std::cout << "Preparing seq " << (i+1) << "/" << fasta_data.num_entries
                      << ": " << entry->header << " (len=" << entry->length << ")" << std::endl;
            cplx_t* complex_seq = dna_to_complex(entry->sequence, entry->length);
            if (!complex_seq) { std::cerr << "  skip: complex conversion failed" << std::endl; continue; }
            CWTFeatures* feat = compute_cwt(complex_seq, entry->length,
                                            config->num_scales, config->min_scale, config->max_scale);
            free(complex_seq);
            if (!feat) { std::cerr << "  skip: CWT failed" << std::endl; continue; }
            LabelType* labels = nullptr;
            if (parse_gff_labels(gff_file, entry->header, entry->length, &labels) != 0) {
                std::cerr << "  skip: GFF labels missing" << std::endl; free_cwt_features(feat); continue; }
            localSeqs.push_back({feat, labels, entry->length, entry->header});
        }
        if (localSeqs.empty()) {
            std::cerr << "No usable sequences for training" << std::endl;
            free_fasta_data(&fasta_data);
            return -1;
        }

        // 3) 모델 생성 (state_dim: window*CWT_scales + prev action one-hot)
        int state_dim = config->window_size * (int)localSeqs[0].feat->num_scales + NUM_LABELS;
        TransformerActorCritic model(config->d_model, config->nhead,
                                     config->num_encoder_layers,
                                     config->dim_feedforward, NUM_LABELS,
                                     state_dim);

        // 4) 시퀀스 래핑하여 PPO 학습
        std::vector<SequenceData> seqDatas; seqDatas.reserve(localSeqs.size());
        for (auto &ls : localSeqs) seqDatas.push_back({ls.feat, ls.labels, ls.len, ls.header});
        std::cout << "Training model over " << seqDatas.size() << " sequences..." << std::endl;
        train_ppo_multi(model, seqDatas, config);

        // 5) 저장
        std::cout << "Saving model to " << config->model_output << std::endl;
        torch::save(model, config->model_output);

        // 6) 클린업
        for (auto &ls : localSeqs) { free(ls.labels); free_cwt_features(ls.feat); }
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
        // Align inference model dims with saved config assumptions (using training defaults)
        // NOTE: For robustness, could load separate JSON config; here we keep prior constants.
        int d_model = 256, nhead=8, num_layers=6, d_ff=1024; // fallback
        size_t state_dim = features->num_scales; // position-wise only (no window) for now
        // Keep same projection logic (state_dim -> d_model)
        TransformerActorCritic model(d_model, nhead, num_layers, d_ff, NUM_LABELS, (int)state_dim);
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
            std::vector<float> state_vec; state_vec.reserve(features->num_scales);
            for (size_t scale=0; scale<features->num_scales; ++scale) {
                cplx_t val = features->data[scale][pos];
                state_vec.push_back(std::abs(val));
            }
            auto state = torch::from_blob(state_vec.data(), {1,1,(long)state_vec.size()}, torch::kFloat32).clone();
            torch::NoGradGuard ng;
            auto fw = model->forward(state);
            auto action = torch::argmax(fw.first, -1).item<int>();
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
