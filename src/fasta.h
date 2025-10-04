#ifndef FASTA_H
#define FASTA_H

#include "aurora.h"

/* Parse FASTA file and return FastaData structure */
int parse_fasta(const char *filename, FastaData *data);

/* Parse GFF3 file and return label sequence for a FASTA entry */
int parse_gff_labels(const char *filename, const char *seqid, 
                     size_t seq_length, LabelType **labels);

#endif /* FASTA_H */
