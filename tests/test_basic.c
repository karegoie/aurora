#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <complex.h>
#include "../src/utils.h"
#include "../src/config.h"

void test_dna_to_complex() {
    printf("Testing dna_to_complex...\n");
    
    const char *seq = "ATGC";
    cplx_t *result = dna_to_complex(seq, 4);
    
    assert(result != NULL);
    
    // A → 1+i
    assert(creal(result[0]) == 1.0);
    assert(cimag(result[0]) == 1.0);
    
    // T → 1-i
    assert(creal(result[1]) == 1.0);
    assert(cimag(result[1]) == -1.0);
    
    // G → -1+i
    assert(creal(result[2]) == -1.0);
    assert(cimag(result[2]) == 1.0);
    
    // C → -1-i
    assert(creal(result[3]) == -1.0);
    assert(cimag(result[3]) == -1.0);
    
    free(result);
    printf("  PASSED\n");
}

void test_config_defaults() {
    printf("Testing config defaults...\n");
    
    AuroraConfig config;
    init_default_config(&config);
    
    assert(config.num_scales == 20);
    assert(config.window_size == 50);
    assert(config.d_model == 256);
    assert(config.nhead == 8);
    
    printf("  PASSED\n");
}

int main() {
    printf("=== Running Aurora Unit Tests ===\n\n");
    
    test_dna_to_complex();
    test_config_defaults();
    
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
