#include "fasta.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int parse_fasta(const char *filename, FastaData *data) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open FASTA file: %s\n", filename);
        return -1;
    }
    
    /* Initialize data structure */
    data->entries = NULL;
    data->num_entries = 0;
    
    size_t capacity = 10;
    data->entries = (FastaEntry *)malloc(capacity * sizeof(FastaEntry));
    if (!data->entries) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return -1;
    }
    
    char line[4096];
    char *current_header = NULL;
    char *current_seq = NULL;
    size_t seq_capacity = 0;
    size_t seq_length = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        /* Remove trailing newline */
        line[strcspn(line, "\r\n")] = 0;
        
        if (line[0] == '>') {
            /* Save previous entry if exists */
            if (current_header && current_seq) {
                /* Resize array if needed */
                if (data->num_entries >= capacity) {
                    capacity *= 2;
                    FastaEntry *temp = (FastaEntry *)realloc(data->entries, 
                                                             capacity * sizeof(FastaEntry));
                    if (!temp) {
                        fprintf(stderr, "Error: Memory reallocation failed\n");
                        free(current_header);
                        free(current_seq);
                        free_fasta_data(data);
                        fclose(fp);
                        return -1;
                    }
                    data->entries = temp;
                }
                
                data->entries[data->num_entries].header = current_header;
                data->entries[data->num_entries].sequence = current_seq;
                data->entries[data->num_entries].length = seq_length;
                data->num_entries++;
            }
            
            /* Start new entry */
            current_header = strdup(line + 1); /* Skip '>' */
            if (!current_header) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                free_fasta_data(data);
                fclose(fp);
                return -1;
            }
            
            seq_capacity = 1024;
            current_seq = (char *)malloc(seq_capacity);
            if (!current_seq) {
                fprintf(stderr, "Error: Memory allocation failed\n");
                free(current_header);
                free_fasta_data(data);
                fclose(fp);
                return -1;
            }
            current_seq[0] = '\0';
            seq_length = 0;
            
        } else if (current_header) {
            /* Append to current sequence */
            size_t line_len = strlen(line);
            
            /* Resize sequence buffer if needed */
            while (seq_length + line_len + 1 > seq_capacity) {
                seq_capacity *= 2;
                char *temp = (char *)realloc(current_seq, seq_capacity);
                if (!temp) {
                    fprintf(stderr, "Error: Memory reallocation failed\n");
                    free(current_header);
                    free(current_seq);
                    free_fasta_data(data);
                    fclose(fp);
                    return -1;
                }
                current_seq = temp;
            }
            
            /* Append line to sequence (convert to uppercase) */
            for (size_t i = 0; i < line_len; i++) {
                current_seq[seq_length++] = toupper(line[i]);
            }
            current_seq[seq_length] = '\0';
        }
    }
    
    /* Save last entry */
    if (current_header && current_seq) {
        if (data->num_entries >= capacity) {
            capacity++;
            FastaEntry *temp = (FastaEntry *)realloc(data->entries, 
                                                     capacity * sizeof(FastaEntry));
            if (!temp) {
                fprintf(stderr, "Error: Memory reallocation failed\n");
                free(current_header);
                free(current_seq);
                free_fasta_data(data);
                fclose(fp);
                return -1;
            }
            data->entries = temp;
        }
        
        data->entries[data->num_entries].header = current_header;
        data->entries[data->num_entries].sequence = current_seq;
        data->entries[data->num_entries].length = seq_length;
        data->num_entries++;
    }
    
    fclose(fp);
    return 0;
}

int parse_gff_labels(const char *filename, const char *seqid, 
                     size_t seq_length, LabelType **labels) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open GFF file: %s\n", filename);
        return -1;
    }
    
    /* Allocate label array (initialized to INTERGENIC) */
    *labels = (LabelType *)calloc(seq_length, sizeof(LabelType));
    if (!*labels) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return -1;
    }
    
    char line[4096];
    while (fgets(line, sizeof(line), fp)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n') continue;
        
        /* Parse GFF3 fields */
        char seq_id[256];
        char source[256];
        char type[256];
        size_t start, end;
        double score;
        char strand;
        int phase;
        
        int ret = sscanf(line, "%255s\t%255s\t%255s\t%zu\t%zu\t%lf\t%c\t%d",
                        seq_id, source, type, &start, &end, &score, &strand, &phase);
        
        if (ret < 7) continue;
        
        /* Check if this line is for the requested sequence */
        if (strcmp(seq_id, seqid) != 0) continue;
        
        /* Assign labels based on feature type */
        LabelType label = LABEL_INTERGENIC;
        if (strcmp(type, "exon") == 0 || strcmp(type, "CDS") == 0) {
            /* Simplified: treat all exons as internal for now */
            /* In a full implementation, would determine initial/internal/terminal/single */
            label = LABEL_EXON_INTERNAL;
        } else if (strcmp(type, "intron") == 0) {
            label = LABEL_INTRON;
        }
        
        /* Assign labels to positions (GFF is 1-based, convert to 0-based) */
        if (start > 0 && end <= seq_length) {
            for (size_t i = start - 1; i < end; i++) {
                (*labels)[i] = label;
            }
        }
    }
    
    fclose(fp);
    return 0;
}

void free_fasta_data(FastaData *data) {
    if (data && data->entries) {
        for (size_t i = 0; i < data->num_entries; i++) {
            free(data->entries[i].header);
            free(data->entries[i].sequence);
        }
        free(data->entries);
        data->entries = NULL;
        data->num_entries = 0;
    }
}
