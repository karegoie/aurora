#ifndef UTILS_H
#define UTILS_H

#include "aurora.h"
#include <complex.h>

/* Convert DNA sequence to complex numbers
 * A → 1+i, T → 1-i, G → -1+i, C → -1-i
 * Returns allocated array of complex numbers
 */
double complex *dna_to_complex(const char *sequence, size_t length);

#endif /* UTILS_H */
