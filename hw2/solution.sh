#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: bash solution.sh <absolute path to graph> <absolute O/P file> k <no of random instances>"
    exit 1
fi

# Compile the C++ program with C++11 standard
g++ -std=c++11 ./src/main.cpp -o main -pthread

# Run the program with the provided arguments
./main "$1" "$2" "$3" "$4"
