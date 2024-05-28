using CSharp;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

static void ClearEvenKernel(Index1D i, ArrayView<int> data, ArrayView<int> output)
{
    int isOdd = data[i] % 2;
    output[i] = isOdd * data[i];
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

// Allocate memory
int[] hostData = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
MemoryBuffer1D<int, Stride1D.Dense> deviceData = accelerator.Allocate1D<int>(hostData.Length);
MemoryBuffer1D<int, Stride1D.Dense> deviceOutput = accelerator.Allocate1D<int>(hostData.Length);

// Get IPC handles, this will zero the data!
var ipcMemHandleData = accelerator.GetIpcHandle(deviceData);
var ipcMemHandleOutput = accelerator.GetIpcHandle(deviceOutput);

// Now load the data
deviceData.CopyFromCPU(hostData);

// load / precompile the kernel
Action<Index1D, ArrayView<int>, ArrayView<int>> loadedKernel = 
    accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(ClearEvenKernel);

// finish compiling and tell the accelerator to start computing the kernel
loadedKernel((int)deviceOutput.Length, deviceData.View, deviceOutput.View);

// wait for the accelerator to be finished with whatever it's doing
// in this case it just waits for the kernel to finish.
accelerator.Synchronize();

// Show script arguments
var scriptArguments = $"""--IpcMemHandleData "{ipcMemHandleData}" """ +
                      $"""--IpcMemHandleOutput "{ipcMemHandleOutput}" """ + 
                      $"""--typestr "<i4" """ +
                      $"""--length {hostData.Length}""";
Console.WriteLine(scriptArguments);

// moved output data from the GPU to the CPU for output to console
int[] hostOutput = deviceOutput.GetAsArray1D();

Console.WriteLine(String.Join(" ",hostOutput));

// Wait so we can manually execute the python script
Console.ReadLine();

// The data only changed on the GPU, so we now have to copy it again
deviceOutput.CopyToCPU(hostOutput);
Console.WriteLine(String.Join(" ",hostOutput));