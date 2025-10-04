#ifndef RL_AGENT_H
#define RL_AGENT_H

#include "aurora.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C interface for training the RL agent */
int run_training(const char *fasta_file, const char *gff_file, 
                 const AuroraConfig *config);

/* C interface for running inference */
int run_inference(const char *model_file, const CWTFeatures *features,
                  LabelType **predictions, size_t *num_predictions);

#ifdef __cplusplus
}
#endif

#endif /* RL_AGENT_H */
