#ifndef MODEL_H
#define MODEL_H

#include "aurora.h"

/* Save model configuration to JSON file */
int save_config_json(const AuroraConfig *config, const char *filename);

/* Load model configuration from JSON file */
int load_config_json(AuroraConfig *config, const char *filename);

#endif /* MODEL_H */
