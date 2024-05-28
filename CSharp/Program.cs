using CSharp;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

static void Kernel(Index1D i, ArrayView<int> data, ArrayView<int> output)
{
    output[i] = data[i % data.Length];
}

// Initialize ILGPU.
using Context context = Context.CreateDefault();
Context.DeviceCollection<CudaDevice> cudaDevices = context.GetCudaDevices();
if (cudaDevices.Count == 0)
{
    Console.WriteLine("No CUDA devices found!");
    return;
}
using CudaAccelerator accelerator = cudaDevices[0]
    .CreateCudaAccelerator(context);

// Load the data.
MemoryBuffer1D<int, Stride1D.Dense> deviceData = accelerator.Allocate1D(new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });
MemoryBuffer1D<int, Stride1D.Dense> deviceOutput = accelerator.Allocate1D<int>(10_000);

// load / precompile the kernel
Action<Index1D, ArrayView<int>, ArrayView<int>> loadedKernel = 
    accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(Kernel);

// finish compiling and tell the accelerator to start computing the kernel
loadedKernel((int)deviceOutput.Length, deviceData.View, deviceOutput.View);

// wait for the accelerator to be finished with whatever it's doing
// in this case it just waits for the kernel to finish.
accelerator.Synchronize();

// moved output data from the GPU to the CPU for output to console
int[] hostOutput = deviceOutput.GetAsArray1D();

Console.WriteLine(String.Join(" ",hostOutput[..50]));


var cudaError = accelerator.IpcGetMemHandle(out var ipcMemHandle, deviceOutput.NativePtr);
Console.WriteLine($"{cudaError} | {ipcMemHandle}");
// CUDA_SUCCESS | 8087A46657020000E869000000000000409C0000000000000002000000020000001000000000000027000000000000000C060000000000008007004000000000

// Using a hex string for copy/paste IPC
// Replace with your own
var externalIpcMemHandle = CudaIpcMemHandle.FromHexString(
    "90ae13833a02000004160000000000009001000000000000000200000000000000100000000000002700000000000000b4010000000000004002004000000000");
// A float32 array with 100 elements was allocated in python.
try
{
    // Alternatively you could directly allocate the new buffer with
    // new CudaIpcMemoryBuffer(accelerator, externalIpcMemHandle, 100, 4)
    var externalBuffer = accelerator.MapIpc(externalIpcMemHandle, 100, 4);
    float[] cpuBuffer = new float[100];
    var arrayView = externalBuffer.AsArrayView<float>(0, 100);
    arrayView
        .CopyToCPU(cpuBuffer);
    Console.WriteLine(String.Join(" ",cpuBuffer[..5]));
    // 1 2 3 4 5
}
catch
{
    Console.WriteLine("This won't work in the same process!");
}