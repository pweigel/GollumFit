#!/bin/bash
#
# Run GPU tests for GollumFit
#
# Usage: ./run_tests.sh [test_name]
#   If test_name is provided, only that test is run
#   Otherwise, all tests are run
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../../build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "GollumFit GPU Test Runner"
echo "========================================"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please load CUDA module.${NC}"
    echo "  e.g., module load cuda/11.0"
    exit 1
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Build directory not found. Creating...${NC}"
    mkdir -p "$BUILD_DIR"
fi

# Build tests
cd "$BUILD_DIR"
echo ""
echo "Configuring with CMake..."
cmake .. -DUSE_CUDA=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

echo ""
echo "Building tests..."
make -j$(nproc) test_gpu_autodiff test_histogram_reduction test_likelihood test_spline_evaluation test_integration

echo ""
echo "========================================"
echo "Running Tests"
echo "========================================"

# List of tests
TESTS=("test_gpu_autodiff" "test_histogram_reduction" "test_likelihood" "test_spline_evaluation" "test_integration")

# If a specific test is requested
if [ -n "$1" ]; then
    TESTS=("$1")
fi

PASSED=0
FAILED=0

for test in "${TESTS[@]}"; do
    echo ""
    echo "Running: $test"
    echo "----------------------------------------"

    if [ -f "./test/cuda/$test" ]; then
        TEST_PATH="./test/cuda/$test"
    elif [ -f "./$test" ]; then
        TEST_PATH="./$test"
    else
        echo -e "${RED}Test executable not found: $test${NC}"
        FAILED=$((FAILED + 1))
        continue
    fi

    if $TEST_PATH; then
        echo -e "${GREEN}PASSED: $test${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAILED: $test${NC}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
