#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash solution.sh <absolute path to graph> <absolute O/P file> k <no of random instances>"
    exit 1
fi

# Compile the C++ program (if not already compiled)
g++ -std=c++11 -o main main.cpp

# Run the program with the provided arguments
./main "$1" "$2" "$3" "$4"
