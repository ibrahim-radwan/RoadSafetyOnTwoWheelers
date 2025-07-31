#!/bin/bash

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to build directory
cd build

# Run cmake to configure the project
cmake ..

# Build the project
make

# Inform user about the executable location
echo ""
echo "Build complete! The executable is located at: build/fusion_replay"
echo "To run it: ./build/fusion_replay"
