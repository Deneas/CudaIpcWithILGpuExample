using System.Diagnostics;
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

// Get the IPC handle, this will zero the data!
// For some reason, when allocating less than 2 MB total the memory shares the same space and gets all zeroed
// This means deviceData ALSO gets zeroed
var ipcMemHandle = accelerator.GetIpcHandle(deviceOutput);

// After all got zeroed NOW we copy data to the GPU
deviceData.CopyFromCPU(hostData);

// load / precompile the kernel
Action<Index1D, ArrayView<int>, ArrayView<int>> loadedClearEvenKernel = 
    accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(ClearEvenKernel);

// finish compiling and tell the accelerator to start computing the kernel
loadedClearEvenKernel((int)deviceOutput.Length, deviceData.View, deviceOutput.View);

// wait for the accelerator to be finished with whatever it's doing
// in this case it just waits for the kernel to finish.
accelerator.Synchronize();

// Show script arguments
var scriptArguments = $"""--IpcMemHandle "{ipcMemHandle}" """ + 
                      $"""--typestr "<i4" """ +
                      $"""--length {hostData.Length}""";
Console.WriteLine(scriptArguments);

// moved output data from the GPU to the CPU for output to console
int[] hostOutput = deviceOutput.GetAsArray1D();
Console.WriteLine(String.Join(" ",hostOutput));

string[] virtualPythonPaths =
[
    "../../../../Python/.venv/Scripts/python.exe",
    "../../../../Python/venv/Scripts/python.exe",
];

// try to use the system installation
string selectedPythonPath = "python.exe";
foreach (string path in virtualPythonPaths)
{
    if (File.Exists(path))
    {
        selectedPythonPath = path;
        break;
    }
}

var scriptPath = "../../../../Python/CumulativeSum.py";
var process = Process.Start(selectedPythonPath, $"{scriptPath} {scriptArguments}");
process.WaitForExit(TimeSpan.FromSeconds(10));

// The data only changed on the GPU, so we now have to copy it again
deviceOutput.CopyToCPU(hostOutput);
Console.WriteLine(String.Join(" ",hostOutput));