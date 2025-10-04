## Project "Aurora": End-to-End Implementation Brief
To: Coding Agent
From: Project Lead
Subject: [Project Aurora] Greenfield Development of a C-based ab initio Gene Predictor

Greetings. We are initiating the development of "Aurora," a high-performance, high-accuracy ab initio gene predictor, to be written in C. This document serves as the master prompt, defining the complete requirements for the project. Please follow the plan and specifications below precisely to generate the entire codebase.

### 1. Project Vision & Core Philosophy
Project Name: Aurora

Objective: To develop a state-of-the-art ab initio gene predictor that relies solely on DNA sequence information.

Core Philosophy: Signal, Structure, and Strategy

Signal: Extract rich, multi-dimensional signal features from DNA sequences using the Continuous Wavelet Transform (CWT).

Structure: Model the grammatical rules of gene structure using a Transformer-based Actor-Critic agent.

Strategy: Employ a pure Reinforcement Learning (RL) approach where the agent learns to make optimal labeling decisions from scratch, guided by a gold-standard annotation.

Technology Stack: C (C11 standard), C++ (C++17 standard, for RL components), FFTW3, cJSON, tinytoml, LibTorch, OpenMP.

### 2. Core Architectural Requirements
Strict Modularity: The codebase must be organized into clearly separated modules as defined in the file structure below. The use of global variables is strictly forbidden. All configuration and state must be passed via core data structures (e.g., AuroraConfig) as function parameters.

Memory Safety: All memory allocations (malloc, calloc) must be checked for errors. The realloc function must only be used with the safe pattern (i.e., using a temporary pointer).

Type Consistency: All sequence lengths, array sizes, and file sizes must use the size_t type.

Project File Structure:
aurora/
├── src/
│   ├── main.c
│   ├── aurora.h
│   ├── config.c, config.h
│   ├── utils.c, utils.h
│   ├── fasta.c, fasta.h
│   ├── cwt.c, cwt.h
│   ├── model.c, model.h
│   ├── train.c, train.h
│   ├── predict.c, predict.h
│   └── rl_agent.cpp, rl_agent.h  # C++ for RL Agent
├── include/
│   └── (Header files for external libs: cJSON, tinytoml, etc.)
├── tests/
│   └── (Unit test code)
├── Makefile
└── README.md
Core Data Structures (to be defined in aurora.h):
Define all core project data structures here, including AuroraConfig, FastaData, CWTFeatures, and GenePrediction.

### 3. Phased Implementation Plan
Please implement the entire project by following these phases in order.

Phase 0: Project Foundation & Setup

Create a Makefile that can compile and link C and C++ sources together. It must link against FFTW3, cJSON, tinytoml, LibTorch, and OpenMP. Use -O3 and -Wall as default compiler flags.

Define the core data structures in aurora.h.

Set up the source files for external libraries (cJSON, tinytoml) in the include and lib directories.

Phase 1: I/O & Pre-processing Modules

config.c / .h: Implement a function to parse a TOML configuration file (aurora.cfg) into the AuroraConfig struct using the tinytoml library.

fasta.c / .h: Implement a safe and efficient parser for large FASTA files.

utils.c / .h: Implement a function dna_to_complex that converts a DNA sequence into an array of complex numbers according to the rule: A → 1+i, T → 1-i, G → -1+i, C → -1-i.

tests/: Write simple unit tests to validate the parsers.

Phase 2: The CWT Signal Engine

cwt.c / .h: Implement the compute_cwt() function, which performs a Continuous Wavelet Transform using a Morlet wavelet. This function must use the FFTW3 library for high-performance FFT-based convolution. It takes the complex signal from Phase 1 as input and returns a CWTFeatures 2D matrix.

Parallel Optimization: Optimize this module by using OpenMP to parallelize the CWT computation across the different scales.

Phase 3: The Reinforcement Learning Environment

rl_agent.h: Declare the RLEnvironment C++ class. It will hold the pre-computed CWTFeatures matrix and the gold-standard label sequence.

rl_agent.cpp: Implement the RLEnvironment class with the following methods:

getState(t): Returns the state representation for the agent at position t (a window of CWT vectors + the previous action).

step(action): Takes an agent's action (a label), compares it to the gold standard to calculate the reward (+1 or -1), and advances to the next state.

Phase 4: The Transformer Actor-Critic Agent

rl_agent.h / .cpp: Implement the TransformerActorCritic C++ class, inheriting from torch::nn::Module.

Network Architecture: The network must consist of several Transformer Encoder layers, followed by two separate heads: an Actor head (policy, with Softmax) and a Critic head (value, linear).

PPO Algorithm: Implement the core logic for the Proximal Policy Optimization (PPO) algorithm using LibTorch tensors to calculate losses and update the network.

Phase 5: The Training Pipeline (aurora train)

train.c / .h: Implement the main logic for the aurora train command.

C++/C Interface: Create a C-compatible extern "C" interface function in rl_agent.cpp called run_training(), which can be called from C.

Training Process: The run_training function will:

Load the training FASTA and the corresponding gold-standard GFF files.

Pre-compute the CWT spectrogram for each sequence.

Instantiate the RLEnvironment and TransformerActorCritic.

Run the main training loop: have the agent interact with the environment, collect trajectories, and update its policy via PPO.

model.c / .h: After training, implement functions to save the trained TransformerActorCritic model weights (state_dict) to an aurora.pt file and the AuroraConfig to a JSON file.

Phase 6: The Prediction Pipeline (aurora predict)

predict.c / .h: Implement the main logic for the aurora predict command.

Prediction Process:

Load the aurora.pt model weights and configuration file.

For each sequence in the input genome.fasta, compute its CWT spectrogram.

In "inference" mode, have the agent step through the sequence, greedily choosing the action with the highest probability from the Actor's policy.

Collect the sequence of predicted labels.

Post-processing & GFF3 Output: Implement a function to convert the sequence of labels into valid GFF3-formatted gene predictions and write them to standard output.

Phase 7: Final Optimization & Documentation

Performance Profiling: Profile the entire pipeline using tools like gprof or perf to identify and eliminate any remaining bottlenecks.

README.md: Write a comprehensive README.md file that explains how to compile the project, describe the parameters in the aurora.cfg file, and provide clear examples for running aurora train and aurora predict.

Please begin generating the complete codebase for Project Aurora based on this brief. Notify me of the progress at each phase.
