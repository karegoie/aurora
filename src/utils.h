#ifndef UTILS_H
#define UTILS_H

#include "aurora.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Convert DNA sequence to complex numbers
 * A → 1+i, T → 1-i, G → -1+i, C → -1-i
 * Returns allocated array of complex numbers
 */
cplx_t *dna_to_complex(const char *sequence, size_t length);

#ifdef __cplusplus
}
#endif

#endif /* UTILS_H */
