#include "model.h"
#include <stdio.h>
#include <string.h>

int save_config_json(const AuroraConfig *config, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file for writing: %s\n", filename);
        return -1;
    }
    
    fprintf(fp, "{\n");
    fprintf(fp, "  \"num_scales\": %d,\n", config->num_scales);
    fprintf(fp, "  \"min_scale\": %f,\n", config->min_scale);
    fprintf(fp, "  \"max_scale\": %f,\n", config->max_scale);
    fprintf(fp, "  \"window_size\": %d,\n", config->window_size);
    fprintf(fp, "  \"d_model\": %d,\n", config->d_model);
    fprintf(fp, "  \"nhead\": %d,\n", config->nhead);
    fprintf(fp, "  \"num_encoder_layers\": %d,\n", config->num_encoder_layers);
    fprintf(fp, "  \"dim_feedforward\": %d\n", config->dim_feedforward);
    fprintf(fp, "}\n");
    
    fclose(fp);
    return 0;
}

int load_config_json(AuroraConfig *config, const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file for reading: %s\n", filename);
        return -1;
    }
    
    /* Simple JSON parser (basic implementation) */
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int int_val;
        double double_val;
        
        if (sscanf(line, "  \"num_scales\": %d", &int_val) == 1) {
            config->num_scales = int_val;
        } else if (sscanf(line, "  \"min_scale\": %lf", &double_val) == 1) {
            config->min_scale = double_val;
        } else if (sscanf(line, "  \"max_scale\": %lf", &double_val) == 1) {
            config->max_scale = double_val;
        } else if (sscanf(line, "  \"window_size\": %d", &int_val) == 1) {
            config->window_size = int_val;
        } else if (sscanf(line, "  \"d_model\": %d", &int_val) == 1) {
            config->d_model = int_val;
        } else if (sscanf(line, "  \"nhead\": %d", &int_val) == 1) {
            config->nhead = int_val;
        } else if (sscanf(line, "  \"num_encoder_layers\": %d", &int_val) == 1) {
            config->num_encoder_layers = int_val;
        } else if (sscanf(line, "  \"dim_feedforward\": %d", &int_val) == 1) {
            config->dim_feedforward = int_val;
        }
    }
    
    fclose(fp);
    return 0;
}
