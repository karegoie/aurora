#!/bin/bash
set -e

sudo apt-get update
# Install fftw (already required) and libgomp runtime
sudo apt-get install -y libfftw3-dev libgomp1

LIBTORCH_DIR="libtorch"
# Allow overriding the libtorch version via env var, default to 2.8.0
LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.8.0}"
LIBTORCH_ZIP="libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/${LIBTORCH_ZIP}"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Downloading libtorch ${LIBTORCH_VERSION} from ${LIBTORCH_URL}"
    wget "$LIBTORCH_URL" -O libtorch.zip
    unzip libtorch.zip
    rm libtorch.zip
else
    echo "$LIBTORCH_DIR already exists, skipping download."
fi

# If this repo bundles a libgomp shared object, copy it to a system library dir
# so the dynamic linker can find it (avoid overwriting existing files).
LIB_DIR="$(pwd)/lib"
if compgen -G "$LIB_DIR/libgomp*.so*" > /dev/null; then
    echo "Installing bundled libgomp libraries from $LIB_DIR to /usr/local/lib"
    for f in "$LIB_DIR"/libgomp*.so*; do
        echo "Copying $f to /usr/local/lib/"
        sudo cp -n "$f" /usr/local/lib/ || true
    done
    sudo ldconfig
else
    echo "No bundled libgomp libraries found in $LIB_DIR"
fi

# Build the project
make
