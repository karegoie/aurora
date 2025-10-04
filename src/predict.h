#ifndef PREDICT_H
#define PREDICT_H

#include "aurora.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Main prediction function */
int predict_genes(const char *model_file, const char *fasta_file, 
                  const AuroraConfig *config);

/* Convert label sequence to GFF3 format */
void labels_to_gff3(const char *seqid, const LabelType *labels, 
                    size_t length);

#ifdef __cplusplus
}
#endif

#endif /* PREDICT_H */
