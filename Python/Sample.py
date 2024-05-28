import argparse
import types
import torch
from cuda import cuda
from cuda.cuda import CUipcMemHandle


def _parse_arguments():
    global args
    parser = argparse.ArgumentParser(description='Process data on CUDA-GPUs via IPC')
    parser.add_argument("--IpcMemHandleData",
                        dest="ipcMemHandleData",
                        type=str,
                        help="The IpcMemHandle of the data as Hexstring with length 128")
    parser.add_argument("--IpcMemHandleOutput",
                        dest="ipcMemHandleOutput",
                        type=str,
                        help="The IpcMemHandle of the output as Hexstring with length 128")
    parser.add_argument("--typestr",
                        dest="typestr",
                        type=str,
                        help="The datatype of the memory in __array_interface__ format.")
    parser.add_argument("--length",
                        dest="length",
                        type=int,
                        help="The length of the buffer (elements, not bytes).")
    parser.add_argument("--cuda-device",
                        dest="device_id",
                        type=int,
                        default=0,
                        help="The index of the CUDA device to use.")
    return parser.parse_args()


def throw_if_cuda_failed(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def initialize_cuda(device_id):
    # Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    throw_if_cuda_failed(err)

    # Retrieve handle for device
    err, cuDevice = cuda.cuDeviceGet(device_id)
    throw_if_cuda_failed(err)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)
    throw_if_cuda_failed(err)


def get_ipc_memory_as_tensor(hexstr, typestr, length, device_id):
    handle = CUipcMemHandle()
    handle.reserved = bytes.fromhex(hexstr)
    err, dev_ptr = cuda.cuIpcOpenMemHandle(handle, 0x1)
    throw_if_cuda_failed(err)
    cuda_dict = {
        "shape": (length,),
        "typestr": typestr,
        "data": (int(dev_ptr), False),
        "version": 3,
        "strides": None,
        "descr": None,
        "mask": None,
        "stream": None
    }

    # see https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    buffer = types.SimpleNamespace()
    setattr(buffer, "__cuda_array_interface__", cuda_dict)
    return torch.as_tensor(buffer, device=f"cuda:{device_id}")


def process_data(data: torch.Tensor, output: torch.Tensor):
    torch.cumsum(data, dim=0, out=output)


if __name__ == '__main__':
    args = _parse_arguments()
    initialize_cuda(args.device_id)
    data_memory = get_ipc_memory_as_tensor(args.ipcMemHandleData, args.typestr, args.length, args.device_id)
    output_memory = get_ipc_memory_as_tensor(args.ipcMemHandleOutput, args.typestr, args.length, args.device_id)
    print(output_memory)
    process_data(data_memory, output_memory)
    print(output_memory)

