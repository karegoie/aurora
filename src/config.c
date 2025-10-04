#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void init_default_config(AuroraConfig *config) {
    /* General parameters */
    config->num_scales = 20;
    config->min_scale = 1.0;
    config->max_scale = 100.0;
    config->window_size = 50;
    
    /* Training parameters */
    config->num_epochs = 100;
    config->learning_rate = 0.0001;
    config->batch_size = 32;
    config->gamma = 0.99;
    config->clip_epsilon = 0.2;
    
    /* Model architecture */
    config->d_model = 256;
    config->nhead = 8;
    config->num_encoder_layers = 6;
    config->dim_feedforward = 1024;
    
    /* File paths */
    strcpy(config->train_fasta, "train.fasta");
    strcpy(config->train_gff, "train.gff");
    strcpy(config->model_output, "aurora.pt");
}

int parse_config(const char *filename, AuroraConfig *config) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open config file: %s\n", filename);
        return -1;
    }
    
    /* Initialize with defaults first */
    init_default_config(config);
    
    /* Simple key-value parser (basic implementation) */
    /* In a full implementation, would use tinytoml library */
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        
        /* Remove trailing newline */
        line[strcspn(line, "\r\n")] = 0;
        
        /* Parse key = value */
        char *eq = strchr(line, '=');
        if (!eq) continue;
        
        *eq = '\0';
        char *key = line;
        char *value = eq + 1;
        
        /* Trim whitespace */
        while (*key == ' ' || *key == '\t') key++;
        while (*value == ' ' || *value == '\t') value++;
        
        /* Remove trailing whitespace from key */
        char *end = key + strlen(key) - 1;
        while (end > key && (*end == ' ' || *end == '\t')) {
            *end = '\0';
            end--;
        }
        
        /* Parse different config values */
        if (strcmp(key, "num_scales") == 0) {
            config->num_scales = atoi(value);
        } else if (strcmp(key, "min_scale") == 0) {
            config->min_scale = atof(value);
        } else if (strcmp(key, "max_scale") == 0) {
            config->max_scale = atof(value);
        } else if (strcmp(key, "window_size") == 0) {
            config->window_size = atoi(value);
        } else if (strcmp(key, "num_epochs") == 0) {
            config->num_epochs = atoi(value);
        } else if (strcmp(key, "learning_rate") == 0) {
            config->learning_rate = atof(value);
        } else if (strcmp(key, "batch_size") == 0) {
            config->batch_size = atoi(value);
        } else if (strcmp(key, "gamma") == 0) {
            config->gamma = atof(value);
        } else if (strcmp(key, "clip_epsilon") == 0) {
            config->clip_epsilon = atof(value);
        } else if (strcmp(key, "d_model") == 0) {
            config->d_model = atoi(value);
        } else if (strcmp(key, "nhead") == 0) {
            config->nhead = atoi(value);
        } else if (strcmp(key, "num_encoder_layers") == 0) {
            config->num_encoder_layers = atoi(value);
        } else if (strcmp(key, "dim_feedforward") == 0) {
            config->dim_feedforward = atoi(value);
        } else if (strcmp(key, "train_fasta") == 0) {
            /* Remove quotes if present */
            if (value[0] == '"') value++;
            size_t len = strlen(value);
            if (len > 0 && value[len-1] == '"') value[len-1] = '\0';
            strncpy(config->train_fasta, value, sizeof(config->train_fasta) - 1);
        } else if (strcmp(key, "train_gff") == 0) {
            if (value[0] == '"') value++;
            size_t len = strlen(value);
            if (len > 0 && value[len-1] == '"') value[len-1] = '\0';
            strncpy(config->train_gff, value, sizeof(config->train_gff) - 1);
        } else if (strcmp(key, "model_output") == 0) {
            if (value[0] == '"') value++;
            size_t len = strlen(value);
            if (len > 0 && value[len-1] == '"') value[len-1] = '\0';
            strncpy(config->model_output, value, sizeof(config->model_output) - 1);
        }
    }
    
    fclose(fp);
    return 0;
}

void free_config(AuroraConfig *config) {
    /* Nothing to free in current implementation */
    (void)config;
}
