# Aurora - Ab Initio Gene Predictor

Aurora is a high-performance, high-accuracy ab initio gene predictor written in C/C++ that uses signal processing and deep reinforcement learning to identify genes in DNA sequences.

## Overview

Aurora combines three core technologies:
- **Signal Processing**: Continuous Wavelet Transform (CWT) with FFTW3 for multi-scale feature extraction
- **Deep Learning**: Transformer-based Actor-Critic architecture with LibTorch
- **Reinforcement Learning**: Proximal Policy Optimization (PPO) for learning gene structure

## Requirements

- C compiler supporting C11 (gcc/clang)
- C++ compiler supporting C++17 (g++/clang++)
- FFTW3 library (with OpenMP support)
- LibTorch (PyTorch C++ API)
- OpenMP

### Installing Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get install libfftw3-dev libfftw3-omp-dev
```

#### macOS
```bash
brew install fftw libomp
```

#### LibTorch
Download from https://pytorch.org/cplusplus/ and extract to a location on your system.

## Compilation

1. Ensure LibTorch is installed and Python 3 with PyTorch is available
2. Build the project:
```bash
make
```

This will create the `aurora` executable.

## Configuration

Aurora uses a configuration file (default: `aurora.cfg`) to set parameters. Key parameters include:

### CWT Parameters
- `num_scales`: Number of wavelet scales (default: 20)
- `min_scale`: Minimum wavelet scale (default: 1.0)
- `max_scale`: Maximum wavelet scale (default: 100.0)
- `window_size`: Context window size for the agent (default: 50)

### Training Parameters
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate for optimizer (default: 0.0001)
- `batch_size`: Batch size (default: 32)
- `gamma`: Discount factor for RL (default: 0.99)
- `clip_epsilon`: PPO clipping parameter (default: 0.2)

### Model Architecture
- `d_model`: Transformer model dimension (default: 256)
- `nhead`: Number of attention heads (default: 8)
- `num_encoder_layers`: Number of transformer layers (default: 6)
- `dim_feedforward`: Feedforward dimension (default: 1024)

### File Paths
- `train_fasta`: Path to training FASTA file
- `train_gff`: Path to training GFF3 annotation file
- `model_output`: Output path for trained model (default: aurora.pt)

## Usage

### Training a Model

Train Aurora on annotated sequences:

```bash
./aurora train aurora.cfg
```

This will:
1. Load training sequences from the FASTA file
2. Load gold-standard annotations from the GFF3 file
3. Compute CWT features for all sequences
4. Train the Transformer Actor-Critic agent using PPO
5. Save the trained model to the specified output file

### Predicting Genes

Predict genes in new sequences using a trained model:

```bash
./aurora predict aurora.pt genome.fasta aurora.cfg > predictions.gff3
```

This will:
1. Load the trained model
2. Process each sequence in the input FASTA file
3. Compute CWT features
4. Run inference with the trained agent
5. Output predictions in GFF3 format to stdout

## Input Formats

### FASTA Format
Standard FASTA format for DNA sequences:
```
>sequence_1
ATCGATCGATCG...
>sequence_2
GCTAGCTAGCTA...
```

### GFF3 Format
Standard GFF3 format for gene annotations:
```
##gff-version 3
sequence_1    source    exon    100    200    .    +    .    ID=exon1
sequence_1    source    exon    300    400    .    +    .    ID=exon2
```

## Output Format

Aurora outputs predictions in GFF3 format with the following features:
- Exons (initial, internal, terminal, single)
- Introns

Example output:
```
##gff-version 3
##sequence-region sequence_1 1 5000
sequence_1    Aurora    exon    150    250    .    +    .    ID=gene1_exon_150
sequence_1    Aurora    exon    350    450    .    +    .    ID=gene2_exon_350
```

## Architecture

Aurora follows a modular architecture:
- **config.c/h**: Configuration file parsing
- **fasta.c/h**: FASTA and GFF3 file I/O
- **utils.c/h**: DNA to complex number conversion
- **cwt.c/h**: Continuous Wavelet Transform computation
- **rl_agent.cpp/h**: Reinforcement learning environment and agent
- **model.c/h**: Model serialization
- **train.c/h**: Training pipeline
- **predict.c/h**: Prediction pipeline
- **main.c**: Command-line interface

## Performance Considerations

- CWT computation is parallelized with OpenMP for efficiency
- FFTW3 is used for fast FFT-based convolution
- The model uses GPU acceleration when available (via LibTorch)

## Limitations

This is a simplified implementation for demonstration purposes. A production-ready gene predictor would require:
- More sophisticated gene structure modeling (splice sites, start/stop codons)
- Strand-specific predictions
- Better handling of incomplete genes at sequence boundaries
- Extensive training on large annotated genomes
- Validation and benchmarking on standard datasets

## License

This project is provided as-is for educational and research purposes.

## Citation

If you use Aurora in your research, please cite:
```
Aurora: A Deep Reinforcement Learning Approach to Ab Initio Gene Prediction
```

## Contact

For questions or issues, please open an issue on GitHub.
