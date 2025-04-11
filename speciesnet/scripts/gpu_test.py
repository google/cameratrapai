# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to check GPU availability for PyTorch in the current Python environment."""


def torch_test():
    """Print diagnostic information about Torch/CUDA status, including Torch/CUDA
    versions and all available CUDA device names.
    """

    try:
        import torch
    except Exception as e:
        print(
            "PyTorch unavailable, not running PyTorch tests. "
            "PyTorch import error was:\n{}".format(str(e))
        )
        return

    print("Torch version: {}".format(str(torch.__version__)))
    print("CUDA available (according to PyTorch): {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        print("CUDA version (according to PyTorch): {}".format(torch.version.cuda))  # type: ignore
        print(
            "CuDNN version (according to PyTorch): {}".format(
                torch.backends.cudnn.version()
            )
        )

    device_ids = list(range(torch.cuda.device_count()))

    if len(device_ids) > 0:
        cuda_str = "Found {} CUDA devices:".format(len(device_ids))
        print(cuda_str)

        for device_id in device_ids:
            device_name = "unknown"
            try:
                device_name = torch.cuda.get_device_name(device=device_id)
            except Exception:
                pass
            print("{}: {}".format(device_id, device_name))
    else:
        print("No GPUs reported by PyTorch")

    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            print("PyTorch reports that Metal Performance Shaders are available")
    except Exception:
        pass
    return len(device_ids)


if __name__ == "__main__":

    print("*** Running Torch tests ***\n")
    torch_test()
