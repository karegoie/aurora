#ifndef CWT_H
#define CWT_H

#include "aurora.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compute Continuous Wavelet Transform using Morlet wavelet
 * Uses FFTW3 for efficient FFT-based convolution
 * Parallelized with OpenMP
 */
CWTFeatures *compute_cwt(const cplx_t *signal, size_t length,
                         int num_scales, double min_scale, double max_scale);

#ifdef __cplusplus
}
#endif

#endif /* CWT_H */
