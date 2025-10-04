#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

double complex *dna_to_complex(const char *sequence, size_t length) {
    double complex *result = (double complex *)malloc(length * sizeof(double complex));
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in dna_to_complex\n");
        return NULL;
    }
    
    for (size_t i = 0; i < length; i++) {
        char base = toupper(sequence[i]);
        
        switch (base) {
            case 'A':
                result[i] = 1.0 + 1.0 * I;
                break;
            case 'T':
                result[i] = 1.0 - 1.0 * I;
                break;
            case 'G':
                result[i] = -1.0 + 1.0 * I;
                break;
            case 'C':
                result[i] = -1.0 - 1.0 * I;
                break;
            default:
                /* Handle ambiguous bases as 0 */
                result[i] = 0.0 + 0.0 * I;
                break;
        }
    }
    
    return result;
}
