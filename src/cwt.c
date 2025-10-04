#include "cwt.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <string.h>

/* Some editors/compilers may not have OpenMP headers available. Guard the
 * include to avoid red squiggles in editors and allow building without
 * OpenMP by providing minimal fallbacks. If OpenMP is available, use it. */
#if defined(__has_include)
#  if __has_include(<omp.h>)
#    include <omp.h>
#    define AURORA_HAVE_OMP 1
#  else
#    define AURORA_HAVE_OMP 0
#  endif
#else
/* Fallback: assume OpenMP is available when compiled with -fopenmp */
#  if defined(_OPENMP)
#    include <omp.h>
#    define AURORA_HAVE_OMP 1
#  else
#    define AURORA_HAVE_OMP 0
#  endif
#endif

#if !AURORA_HAVE_OMP
/* Provide minimal stubs so code can compile and editors won't flag missing
 * symbols. These are only used when OpenMP isn't enabled; they don't provide
 * parallelism. */
static inline int omp_get_max_threads(void) { return 1; }
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Morlet wavelet function */
static cplx_t morlet_wavelet(double t, double scale) {
    const double omega0 = 6.0; /* Center frequency */
    double scaled_t = t / scale;
    double envelope = exp(-scaled_t * scaled_t / 2.0);
    cplx_t oscillation = cexp(I * omega0 * scaled_t);
    return envelope * oscillation / sqrt(scale);
}

CWTFeatures *compute_cwt(const cplx_t *signal, size_t length,
                         int num_scales, double min_scale, double max_scale) {
    if (!signal || length == 0 || num_scales <= 0) {
        fprintf(stderr, "Error: Invalid input parameters to compute_cwt\n");
        return NULL;
    }
    /* Validate scale parameters early to avoid domain errors (log, sqrt, etc.) */
    if (min_scale <= 0.0 || max_scale <= 0.0 || min_scale >= max_scale) {
        fprintf(stderr, "Error: Invalid scale range: min_scale=%f max_scale=%f\n",
                min_scale, max_scale);
        return NULL;
    }
    
    /* Allocate CWT features structure */
    CWTFeatures *features = (CWTFeatures *)malloc(sizeof(CWTFeatures));
    if (!features) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return NULL;
    }
    
    features->num_scales = (size_t)num_scales;
    features->length = length;
    
    /* Allocate 2D array for CWT coefficients */
    features->data = (double complex **)malloc(num_scales * sizeof(double complex *));
    if (!features->data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(features);
        return NULL;
    }
    
    for (int s = 0; s < num_scales; s++) {
        features->data[s] = (double complex *)malloc(length * sizeof(double complex));
        if (!features->data[s]) {
            fprintf(stderr, "Error: Memory allocation failed\n");
            for (int j = 0; j < s; j++) {
                free(features->data[j]);
            }
            free(features->data);
            free(features);
            return NULL;
        }
    }
    
    /* Ensure FFTW internal multithreading is not used by this program.
     * We rely on the system FFTW library being the single-threaded variant
     * (or we avoid calling the threading APIs). Do not call
     * fftw_init_threads()/fftw_plan_with_nthreads() so that FFTW stays
     * single-threaded. */
    
    /* Compute scales logarithmically */
    double *scales = (double *)malloc(num_scales * sizeof(double));
    if (!scales) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free_cwt_features(features);
        return NULL;
    }
    
    /* Compute logarithmically spaced scales. Handle the degenerate case
     * where num_scales == 1 to avoid division by zero. */
    double log_min = log(min_scale);
    double log_max = log(max_scale);
    if (num_scales == 1) {
        scales[0] = min_scale;
    } else {
        for (int s = 0; s < num_scales; s++) {
            double t = (double)s / (double)(num_scales - 1);
            scales[s] = exp(log_min + t * (log_max - log_min));
        }
    }
    
    /* Allow disabling OpenMP parallelism at runtime for debugging.
     * Setting AURORA_CWT_SINGLE_THREAD=1 will run the loop serially which
     * helps isolate crashes caused by FFTW/OpenMP interactions. */
    /* Default: do not use OpenMP parallel loop because some FFTW builds
     * and OpenMP can interact badly and cause crashes. Allow overriding
     * with AURORA_CWT_SINGLE_THREAD=0 to enable parallel execution. */
    int use_parallel = 0;
    const char *env = getenv("AURORA_CWT_SINGLE_THREAD");
    if (env) {
        if (strcmp(env, "1") == 0) {
            use_parallel = 0;
        } else if (strcmp(env, "0") == 0) {
            use_parallel = 1;
        }
    }

    /* Optional debug logging: set DEBUG_CWT=1 to print per-scale progress */
    int debug_cwt = 0;
    const char *dbg = getenv("DEBUG_CWT");
    if (dbg && strcmp(dbg, "1") == 0) debug_cwt = 1;

    /* Compute CWT for each scale; OpenMP will only create a parallel region
     * if use_parallel is non-zero. */
    #pragma omp parallel for if(use_parallel) schedule(dynamic)
    for (int s = 0; s < num_scales; s++) {
        double scale = scales[s];
        if (debug_cwt) {
            fprintf(stderr, "[CWT] thread %d processing scale %d/%d (scale=%f)\n",
                    (int)omp_get_thread_num(), s, num_scales, scale);
        }
        
        /* Prepare signal and wavelet for FFT */
    fftw_complex *sig_fft = fftw_alloc_complex(length);
    fftw_complex *wav_fft = fftw_alloc_complex(length);
    fftw_complex *conv_fft = fftw_alloc_complex(length);
        
        if (!sig_fft || !wav_fft || !conv_fft) {
            fprintf(stderr, "Error: FFTW memory allocation failed\n");
            if (sig_fft) fftw_free(sig_fft);
            if (wav_fft) fftw_free(wav_fft);
            if (conv_fft) fftw_free(conv_fft);
            continue;
        }
        
        /* Copy signal */
        for (size_t i = 0; i < length; i++) {
            /* Assign real and imag parts from cplx_t to fftw_complex which
             * may be an array or C99 complex depending on FFTW build. Cast
             * the element to double* to write real/imag consistently. */
            double *p = (double *)&sig_fft[i];
            p[0] = creal(signal[i]);
            p[1] = cimag(signal[i]);
        }
        
        /* Generate wavelet at current scale */
        for (size_t i = 0; i < length; i++) {
            int t = (i <= length/2) ? (int)i : (int)i - (int)length;
            cplx_t wav = morlet_wavelet((double)t, scale);
            double *q = (double *)&wav_fft[i];
            q[0] = creal(wav);
            q[1] = cimag(wav);
        }
        
        /* Create FFT plans */
        fftw_plan plan_sig = fftw_plan_dft_1d((int)length, sig_fft, sig_fft, 
                                              FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_wav = fftw_plan_dft_1d((int)length, wav_fft, wav_fft, 
                                              FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_plan plan_inv = fftw_plan_dft_1d((int)length, conv_fft, conv_fft, 
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
        
        /* Execute FFTs */
        fftw_execute(plan_sig);
        fftw_execute(plan_wav);
        
        /* Multiply in frequency domain (convolution) */
        for (size_t i = 0; i < length; i++) {
            /* Multiply: (a+ib)*(c+id) = (ac - bd) + i(ad + bc) */
            double *ps = (double *)&sig_fft[i];
            double *pw = (double *)&wav_fft[i];
            double *pc = (double *)&conv_fft[i];
            double a = ps[0];
            double b = ps[1];
            double c = pw[0];
            double d = pw[1];
            pc[0] = a * c - b * d;
            pc[1] = a * d + b * c;
        }
        
        /* Inverse FFT */
        fftw_execute(plan_inv);
        
        /* Normalize and store results */
        for (size_t i = 0; i < length; i++) {
            /* conv_fft may be stored as real/imag pairs; construct cplx_t */
            double *pc = (double *)&conv_fft[i];
            cplx_t val = pc[0] + I * pc[1];
            features->data[s][i] = val / (double)length;
        }
        
        /* Clean up */
        fftw_destroy_plan(plan_sig);
        fftw_destroy_plan(plan_wav);
        fftw_destroy_plan(plan_inv);
        fftw_free(sig_fft);
        fftw_free(wav_fft);
        fftw_free(conv_fft);
    }
    
    free(scales);
    
    return features;
}

void free_cwt_features(CWTFeatures *features) {
    if (features) {
        if (features->data) {
            for (size_t i = 0; i < features->num_scales; i++) {
                free(features->data[i]);
            }
            free(features->data);
        }
        free(features);
    }
}
