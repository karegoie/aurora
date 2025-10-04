#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

cplx_t *dna_to_complex(const char *sequence, size_t length) {
    cplx_t *result = (cplx_t *)malloc(length * sizeof(cplx_t));
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in dna_to_complex\n");
        return NULL;
    }
    
    for (size_t i = 0; i < length; i++) {
        char base = toupper(sequence[i]);
        
        switch (base) {
            case 'A':
                result[i] = (cplx_t)(1.0 + 1.0 * I);
                break;
            case 'T':
                result[i] = (cplx_t)(1.0 - 1.0 * I);
                break;
            case 'G':
                result[i] = (cplx_t)(-1.0 + 1.0 * I);
                break;
            case 'C':
                result[i] = (cplx_t)(-1.0 - 1.0 * I);
                break;
            default:
                /* Handle ambiguous bases as 0 */
                result[i] = (cplx_t)(0.0 + 0.0 * I);
                break;
        }
    }
    
    return result;
}
