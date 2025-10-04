#ifndef CWT_H
#define CWT_H

#include "aurora.h"
#include <complex.h>

/* Compute Continuous Wavelet Transform using Morlet wavelet
 * Uses FFTW3 for efficient FFT-based convolution
 * Parallelized with OpenMP
 */
CWTFeatures *compute_cwt(const double complex *signal, size_t length,
                         int num_scales, double min_scale, double max_scale);

#endif /* CWT_H */
