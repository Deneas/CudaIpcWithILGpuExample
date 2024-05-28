# CUDA InterProcessCommunication Example with ILGPU

In this repo I collect my experiments around the question "What if the memory never leaves the GPU?" (Disclaimer: I actually never verified this)  
Possible use cases might running Machine Learning inference on data between our own calculations. One idea being a game running in C# / ILGPU and a Reinforcement Learning library running in Python.

## Getting started

Disclaimer, I tested this example on Windows 10, with Python 3.8, CUDA Toolkit 12.3 and a GTX 1080 ti.
I do not know whether it will work with other configurations.

### Prerequisites

* .Net 8 (although 7 might work too)
* Python (tested with 3.8)
* The CUDA Toolkit needs to be installed and configured

### Setup

With `pip -r requirements.txt` in the `Python` directory, all necessary packages will be installed.  
It is highly recommended to use `venv` or other similar solutions.

### Executing the example

After opening the solution in the `CSharp` directory, the console program can be executed immediately.

## Current State

* I mapped the IPC methods `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle` and `cudaIpcCloseMemHandle` (as per [CUDA docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html)) including a new struct for the `IpcMemHandle`.
* I created a `CudaIpcMemoryBuffer` which can be used as a normal memory buffer, but uses `IpcOpenMemHandle` and `IpcCloseMemHandle` instead of `AllocateMemory` and `FreeMemory`
* I created the Pull Request [1235](https://github.com/m4rs-mt/ILGPU/pull/1235) to start bringing this functionality to ILGPU
* I created a script to demonstrate the IPC interop with Python
* After setting up Python the C# Program can be executed and will call the python script programatically

## Open Tasks
None