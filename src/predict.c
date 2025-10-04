#include "predict.h"
#include "fasta.h"
#include "cwt.h"
#include "utils.h"
#include "rl_agent.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void labels_to_gff3(const char *seqid, const LabelType *labels, size_t length) {
    printf("##gff-version 3\n");
    printf("##sequence-region %s 1 %zu\n", seqid, length);
    
    /* Track current feature */
    LabelType current_type = LABEL_INTERGENIC;
    size_t feature_start = 0;
    int gene_id = 1;
    
    for (size_t i = 0; i <= length; i++) {
        LabelType new_type = (i < length) ? labels[i] : LABEL_INTERGENIC;
        
        /* Check for feature boundary */
        if (new_type != current_type) {
            /* Output previous feature if not intergenic */
            if (current_type != LABEL_INTERGENIC && feature_start < i) {
                const char *type_str = "exon";
                switch (current_type) {
                    case LABEL_EXON_INITIAL:
                    case LABEL_EXON_INTERNAL:
                    case LABEL_EXON_TERMINAL:
                    case LABEL_EXON_SINGLE:
                        type_str = "exon";
                        break;
                    case LABEL_INTRON:
                        type_str = "intron";
                        break;
                    default:
                        type_str = "unknown";
                        break;
                }
                
                printf("%s\tAurora\t%s\t%zu\t%zu\t.\t+\t.\tID=gene%d_%s_%zu\n",
                       seqid, type_str, feature_start + 1, i, 
                       gene_id, type_str, feature_start);
            }
            
            /* Start new feature */
            current_type = new_type;
            feature_start = i;
            
            if (new_type != LABEL_INTERGENIC) {
                gene_id++;
            }
        }
    }
}

int predict_genes(const char *model_file, const char *fasta_file, 
                  const AuroraConfig *config) {
    printf("=== Aurora Prediction Pipeline ===\n");
    printf("Model: %s\n", model_file);
    printf("Input FASTA: %s\n", fasta_file);
    
    /* Parse FASTA file */
    FastaData fasta_data;
    if (parse_fasta(fasta_file, &fasta_data) != 0) {
        fprintf(stderr, "Failed to parse FASTA file\n");
        return -1;
    }
    
    if (fasta_data.num_entries == 0) {
        fprintf(stderr, "No sequences found in FASTA file\n");
        free_fasta_data(&fasta_data);
        return -1;
    }
    
    /* Process each sequence */
    for (size_t seq_idx = 0; seq_idx < fasta_data.num_entries; seq_idx++) {
        FastaEntry* entry = &fasta_data.entries[seq_idx];
        fprintf(stderr, "Processing sequence: %s (length: %zu)\n", 
                entry->header, entry->length);
        
        /* Convert DNA to complex */
        double complex* complex_seq = dna_to_complex(entry->sequence, entry->length);
        if (!complex_seq) {
            continue;
        }
        
        /* Compute CWT */
        fprintf(stderr, "Computing CWT features...\n");
        CWTFeatures* features = compute_cwt(complex_seq, entry->length,
                                           config->num_scales,
                                           config->min_scale,
                                           config->max_scale);
        free(complex_seq);
        
        if (!features) {
            continue;
        }
        
        /* Run inference */
        fprintf(stderr, "Running inference...\n");
        LabelType* predictions = NULL;
        size_t num_predictions = 0;
        
        if (run_inference(model_file, features, &predictions, &num_predictions) == 0) {
            /* Convert to GFF3 and output */
            labels_to_gff3(entry->header, predictions, num_predictions);
            free(predictions);
        }
        
        free_cwt_features(features);
    }
    
    free_fasta_data(&fasta_data);
    return 0;
}

void free_gene_prediction(GenePrediction *pred) {
    if (pred && pred->features) {
        for (size_t i = 0; i < pred->num_features; i++) {
            free(pred->features[i].seqid);
            free(pred->features[i].source);
            free(pred->features[i].type);
            free(pred->features[i].attributes);
        }
        free(pred->features);
        pred->features = NULL;
        pred->num_features = 0;
    }
}
