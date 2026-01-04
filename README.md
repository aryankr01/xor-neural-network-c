# XOR Neural Network in Pure C

This project implements a simple neural network in **pure C** to learn the XOR function.
No ML frameworks, no libraries — everything from scratch.

## Why XOR?
XOR is a classic **non-linearly separable** problem that demonstrates why hidden layers
are required in neural networks.

## Architecture
- Hidden layer:
  - OR neuron
  - NAND neuron
- Output layer:
  - AND neuron
- Activation: Sigmoid
- Loss: Mean Squared Error
- Optimization: Gradient Descent using Finite Difference

## Features
- Manual forward pass
- Numerical gradient estimation
- Low-level implementation in C
- Demonstrates core ML concepts without abstraction

## Build & Run
```bash
gcc xor.c -o xor -lm
./xor
