# CUDA InterProcessCommunication Example with ILGPU

In this repo I collect my experiments around the question "What if the memory never leaves the GPU?" (Disclaimer: I actually never verified this)  
Possible use cases might running Machine Learning inference on data between our own calculations. One idea being a game running in C# / ILGPU and a Reinforcement Learning library running in Python.

## Current State

* I mapped the IPC methods `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` and `cudaIpcCloseMemHandle` (as per [CUDA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)) including a new struct for the `IpcMemHandle`.
* I tested `cudaIpcGetMemHandle` and was able to access the memory in python, while seeing changes on the C# side.
* With horrible uses of internal methods/state I managed to jam a handle from `cudaIpcOpenMemHandle` into a `CudaMemoryBuffer`, breaking the Disposeability during that process.

## Open Tasks
* Create a Python Sample to show the other side of my tests.
* Explore solutions around creating a useable buffer for IPC managed memory.
* See if this might result in a Pull Request for ILGPU