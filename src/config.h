#ifndef CONFIG_H
#define CONFIG_H

#include "aurora.h"

/* Parse TOML configuration file into AuroraConfig struct */
int parse_config(const char *filename, AuroraConfig *config);

/* Initialize config with default values */
void init_default_config(AuroraConfig *config);

#endif /* CONFIG_H */
