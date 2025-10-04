#!/bin/bash
set -e

# Detect available package manager and install required packages.
# Supports: apt-get (Debian/Ubuntu), dnf (Fedora/RHEL), yum (older RHEL/CentOS).
install_packages() {
    # Package names differ between Debian and Fedora/RHEL families
    packages_apt=(libfftw3-dev libgomp1)
    packages_dnf=(fftw-devel libgomp)

    if command -v apt-get >/dev/null 2>&1; then
        echo "Detected apt-get, using apt to install packages..."
        sudo apt-get update
        sudo apt-get install -y "${packages_apt[@]}"
    elif command -v dnf >/dev/null 2>&1; then
        echo "Detected dnf, using dnf to install packages..."
        # refresh cache then install
        sudo dnf makecache --refresh || true
        sudo dnf -y install "${packages_dnf[@]}"
    elif command -v yum >/dev/null 2>&1; then
        echo "Detected yum, using yum to install packages..."
        sudo yum makecache fast || true
        sudo yum -y install "${packages_dnf[@]}"
    else
        echo "No supported package manager detected (apt-get/dnf/yum). Please install FFTW and libgomp manually." >&2
        return 1
    fi
}

install_packages

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
