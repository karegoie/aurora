#include "aurora.h"
#include "config.h"
#include "train.h"
#include "predict.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog_name) {
    printf("Aurora - Ab initio Gene Predictor\n");
    printf("Usage:\n");
    printf("  %s train <config.cfg>\n", prog_name);
    printf("  %s predict <model.pt> <genome.fasta> <config.cfg>\n", prog_name);
    printf("\n");
    printf("Commands:\n");
    printf("  train    - Train a new model using annotated sequences\n");
    printf("  predict  - Predict genes in new sequences using a trained model\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *command = argv[1];
    
    if (strcmp(command, "train") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Error: Missing configuration file\n");
            print_usage(argv[0]);
            return 1;
        }
        
        const char *config_file = argv[2];
        
        /* Parse configuration */
        AuroraConfig config;
        if (parse_config(config_file, &config) != 0) {
            fprintf(stderr, "Error: Failed to parse configuration file\n");
            return 1;
        }
        
        /* Run training */
        int result = train_model(&config);
        free_config(&config);
        
        return result == 0 ? 0 : 1;
        
    } else if (strcmp(command, "predict") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Error: Missing arguments for predict command\n");
            print_usage(argv[0]);
            return 1;
        }
        
        const char *model_file = argv[2];
        const char *fasta_file = argv[3];
        const char *config_file = argv[4];
        
        /* Parse configuration */
        AuroraConfig config;
        if (parse_config(config_file, &config) != 0) {
            fprintf(stderr, "Error: Failed to parse configuration file\n");
            return 1;
        }
        
        /* Run prediction */
        int result = predict_genes(model_file, fasta_file, &config);
        free_config(&config);
        
        return result == 0 ? 0 : 1;
        
    } else {
        fprintf(stderr, "Error: Unknown command '%s'\n", command);
        print_usage(argv[0]);
        return 1;
    }
}
