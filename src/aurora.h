#ifndef AURORA_H
#define AURORA_H

#include <stddef.h>

/* Cross-language complex type:
 * - In C++, use std::complex<double> as cplx_t
 * - In C, use C99 double complex as cplx_t
 */
#ifdef __cplusplus
#include <complex>
typedef std::complex<double> cplx_t;
#else
#include <complex.h>
typedef double complex cplx_t;
#endif

/* Core Data Structures for Aurora Gene Predictor */

/* Configuration Structure */
typedef struct {
    /* General parameters */
    int num_scales;           /* Number of scales for CWT */
    double min_scale;         /* Minimum scale for CWT */
    double max_scale;         /* Maximum scale for CWT */
    int window_size;          /* Window size for agent state */
    
    /* Training parameters */
    int num_epochs;           /* Number of training epochs */
    double learning_rate;     /* Learning rate for optimizer */
    int batch_size;           /* Batch size for training */
    double gamma;             /* Discount factor for RL */
    double clip_epsilon;      /* Clipping parameter for PPO */
    int num_update_epochs;    /* PPO update epochs per collected batch */
    
    /* Model architecture */
    int d_model;              /* Transformer model dimension */
    int nhead;                /* Number of attention heads */
    int num_encoder_layers;   /* Number of transformer layers */
    int dim_feedforward;      /* Feedforward dimension */
    
    /* File paths */
    char train_fasta[256];    /* Training FASTA file path */
    char train_gff[256];      /* Training GFF file path */
    char model_output[256];   /* Output model file path */
} AuroraConfig;

/* FASTA Data Structure */
typedef struct {
    char *header;             /* Sequence header/description */
    char *sequence;           /* DNA sequence */
    size_t length;            /* Length of sequence */
} FastaEntry;

typedef struct {
    FastaEntry *entries;      /* Array of FASTA entries */
    size_t num_entries;       /* Number of entries */
} FastaData;

/* CWT Features Structure */
typedef struct {
    cplx_t **data;    /* 2D array: [num_scales][sequence_length] */
    size_t num_scales;        /* Number of scales */
    size_t length;            /* Sequence length */
} CWTFeatures;

/* Gene Prediction Structure */
typedef struct {
    char *seqid;              /* Sequence identifier */
    char *source;             /* Prediction source */
    char *type;               /* Feature type (gene, exon, CDS, etc.) */
    size_t start;             /* Start position (1-based) */
    size_t end;               /* End position (1-based) */
    double score;             /* Confidence score */
    char strand;              /* Strand: '+' or '-' */
    int phase;                /* Phase: 0, 1, 2, or -1 */
    char *attributes;         /* GFF3 attributes */
} GeneFeature;

typedef struct {
    GeneFeature *features;    /* Array of gene features */
    size_t num_features;      /* Number of features */
} GenePrediction;

/* Label types for gene structure */
typedef enum {
    LABEL_INTERGENIC = 0,     /* Intergenic region */
    LABEL_EXON_INITIAL,       /* Initial exon */
    LABEL_EXON_INTERNAL,      /* Internal exon */
    LABEL_EXON_TERMINAL,      /* Terminal exon */
    LABEL_EXON_SINGLE,        /* Single exon gene */
    LABEL_INTRON,             /* Intron */
    NUM_LABELS                /* Total number of label types */
} LabelType;

/* Memory management functions */
#ifdef __cplusplus
extern "C" {
#endif

void free_fasta_data(FastaData *data);
void free_cwt_features(CWTFeatures *features);
void free_gene_prediction(GenePrediction *pred);
void free_config(AuroraConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* AURORA_H */
