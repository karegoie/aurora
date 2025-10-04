#include "train.h"
#include "rl_agent.h"
#include "model.h"
#include <stdio.h>
#include <string.h>

int train_model(const AuroraConfig *config) {
    printf("=== Aurora Training Pipeline ===\n");
    printf("Training FASTA: %s\n", config->train_fasta);
    printf("Training GFF: %s\n", config->train_gff);
    printf("Output model: %s\n", config->model_output);
    
    /* Run training through C++ interface */
    int result = run_training(config->train_fasta, config->train_gff, config);
    
    if (result == 0) {
        /* Save configuration */
        char config_file[300];
        snprintf(config_file, sizeof(config_file), "%s.config.json", config->model_output);
        save_config_json(config, config_file);
        printf("Configuration saved to %s\n", config_file);
    }
    
    return result;
}
