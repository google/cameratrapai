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

"""Script to run the SpeciesNet model in chunks.

This is a wrapper for run_model.py.  Supports only --classifier_only mode.
Invokes run_model using python -m speciesnet.scripts.run_model, so it assumes
that the speciesnet package is available, or that the speciesnet repo root is
on the PYTHONPATH.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid

from absl import app
from absl import flags

from speciesnet import only_one_true
from speciesnet.utils import prepare_instances_dict

# Input arguments

_INSTANCES_JSON = flags.DEFINE_string(
    "instances_json",
    None,
    "Input JSON file with instances to get predictions for.",
)
_DETECTIONS_JSON = flags.DEFINE_string(
    "detections_json",
    None,
    "Input JSON file with detections from previous runs.",
)
_FILEPATHS = flags.DEFINE_list(
    "filepaths",
    None,
    "List of image filepaths to get predictions for.",
)
_FILEPATHS_TXT = flags.DEFINE_string(
    "filepaths_txt",
    None,
    "Input TXT file with image filepaths to get predictions for.",
)
_FOLDERS = flags.DEFINE_list(
    "folders",
    None,
    "List of image folders to get predictions for.",
)
_FOLDERS_TXT = flags.DEFINE_string(
    "folders_txt",
    None,
    "Input TXT file with image folders to get predictions for.",
)

# Output arguments

_PREDICTIONS_JSON = flags.DEFINE_string(
    "predictions_json",
    None,
    "Output JSON file for storing computed predictions. Unlike run_model, if this file "
    "exists, it will be over-written.",
)

# Chunking arguments

_CHUNK_SIZE = flags.DEFINE_integer(
    "chunk_size",
    2000,
    "Number of images to run in a single process.",
)

_CLASSIFIER_ONLY = flags.DEFINE_bool(
    "classifier_only",
    False,
    "Run only the classifier component. --classifier_only enables classifier-only mode, --noclassifier_only (default) disables it.",
)
_DETECTOR_ONLY = flags.DEFINE_bool(
    "detector_only",
    False,
    "Run only the detector component. --detector_only enables detector-only mode, --nodetector_only (default) disables it.",
)
_ENSEMBLE_ONLY = flags.DEFINE_bool(
    "ensemble_only",
    False,
    "Run only the ensemble component. --ensemble_only enables ensemble-only mode, --noensemble_only (default) disables it.",
)


def _execute(cmd, encoding=None, errors=None, env=None, verbose=False):
    """Run [cmd] (a single string) in a shell, yielding each line of output to the caller."""

    os.environ["PYTHONUNBUFFERED"] = "1"

    if verbose:
        if encoding is not None:
            print(
                "Launching child process with non-default encoding {}".format(encoding)
            )
        if errors is not None:
            print(
                "Launching child process with non-default text error handling {}".format(
                    errors
                )
            )
        if env is not None:
            print(
                "Launching child process with non-default environment {}".format(
                    str(env)
                )
            )

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        universal_newlines=True,
        encoding=encoding,
        errors=errors,
        env=env,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

    return return_code


def _execute_and_print(
    cmd,
    print_output=True,
    encoding=None,
    errors=None,
    env=None,
    verbose=False,
    catch_exceptions=True,
    echo_command=False,
):
    """Run [cmd] (a single string) in a shell, capturing and printing output.  Returns
    a dictionary with fields "status" and "output".
    """

    if echo_command:
        print("Running command:\n{}\n".format(cmd))

    to_return = {"status": "unknown", "output": ""}
    output = []
    try:
        for s in _execute(
            cmd, encoding=encoding, errors=errors, env=env, verbose=verbose
        ):
            output.append(s)
            if print_output:
                print(s, end="", flush=True)
        to_return["status"] = 0
    except subprocess.CalledProcessError as cpe:
        if not catch_exceptions:
            raise
        print("execute_and_print caught error: {} ({})".format(cpe.output, str(cpe)))
        to_return["status"] = cpe.returncode
    to_return["output"] = output

    return to_return


def _split_list_into_fixed_size_chunks(L, n):
    """Split the list or tuple L into chunks of size n (allowing at most one chunk with size
    less than N, i.e. len(L) does not have to be a multiple of n).

    Args:
        L (list): list to split into chunks
        n (int): preferred chunk size

    Returns:
        list: list of chunks, where each chunk is a list of length n or n-1
    """

    return [L[i * n : (i + 1) * n] for i in range((len(L) + n - 1) // n)]


def _create_argument_list_with_exceptions(all_args, exclude_args):
    """Create an argument list that excludes a subset of an original list."""

    # Filter out the excluded arguments and their values if needed
    filtered_args = []

    # Are we in the middle of an argument value list?
    skip_next = False

    for i_arg, arg in enumerate(all_args):

        # Skip this argument if flagged from previous iteration
        if skip_next:
            skip_next = False
            continue

        # Check whether this argument should be excluded
        should_exclude = False

        for exclude in exclude_args:
            # Check for exact match or key=value style argument
            if arg == exclude or arg.startswith(f"{exclude}="):
                should_exclude = True
                break

            # Check for arguments that take a value as the next argument
            if (
                (arg == exclude)
                and (i_arg < len(all_args) - 1)
                and (not all_args[i_arg + 1].startswith("-"))
            ):
                should_exclude = True
                # Skip the next value as well
                skip_next = True
                break

        if not should_exclude:
            filtered_args.append(arg)

    return filtered_args


def main(argv: str) -> None:

    if (not _CLASSIFIER_ONLY.value) or _DETECTOR_ONLY.value or _ENSEMBLE_ONLY.value:
        raise ValueError("This script only supports --classifier_only mode")

    # Check for valid inputs.
    inputs = [_INSTANCES_JSON, _FILEPATHS, _FILEPATHS_TXT, _FOLDERS, _FOLDERS_TXT]
    inputs_names = [f"--{i.name}" for i in inputs]
    inputs_values = [i.value for i in inputs]
    inputs_strings = [
        f"{name}={value}" for name, value in zip(inputs_names, inputs_values)
    ]
    if not only_one_true(*inputs_values):
        raise ValueError(
            f"Expected exactly one of [{', '.join(inputs_names)}] to be provided. "
            f"Received: [{', '.join(inputs_strings)}]."
        )

    # Extracting these out of absl flags simplifies debugging
    instances_json = _INSTANCES_JSON.value
    detections_json = _DETECTIONS_JSON.value
    filepaths = _FILEPATHS.value
    filepaths_txt = _FILEPATHS_TXT.value
    folders = _FOLDERS.value
    folders_txt = _FOLDERS_TXT.value
    chunk_size = _CHUNK_SIZE.value

    instances_dict = prepare_instances_dict(
        instances_json=instances_json,
        filepaths=filepaths,
        filepaths_txt=filepaths_txt,
        folders=folders,
        folders_txt=folders_txt,
    )

    instances = instances_dict["instances"]
    print("Loaded {} instances".format(len(instances)))

    # Split instances into chunks
    #
    # List of lists of dicts, with key "filepath"
    instance_chunks = _split_list_into_fixed_size_chunks(instances, chunk_size)
    n_chunks = len(instance_chunks)

    print("Divided instances into {} chunks".format(n_chunks))

    with open(detections_json, "r") as f:
        detections = json.load(f)

    filepath_to_detection_results = {}
    for p in detections["predictions"]:
        filepath_to_detection_results[p["filepath"]] = p

    # Split detections into chunks that match the instance chunks
    # List of lists of dicts, with keys "filepath, detections"
    detection_chunks = []

    # chunk = instance_chunks[0]
    for chunk in instance_chunks:
        detection_chunk = []
        # instance = chunk[0]
        for instance in chunk:
            p = filepath_to_detection_results[instance["filepath"]]
            detection_chunk.append(p)
        detection_chunks.append(detection_chunk)

    # Prepare inputs for each chunk
    chunk_file_base = os.path.join(
        tempfile.gettempdir(), "speciesnet_chunks", str(uuid.uuid1())
    )
    os.makedirs(chunk_file_base, exist_ok=True)

    detection_chunk_files = []
    instances_chunk_files = []

    # Double-check that we lined up the instances and detections correctly, and
    # prepare outout files
    for i_chunk, instance_chunk in enumerate(instance_chunks):
        detection_chunk = detection_chunks[i_chunk]
        instance_paths = set(instance["filepath"] for instance in instance_chunk)
        assert instance_paths == set(
            detection["filepath"] for detection in detection_chunk
        )

        detection_chunk_file = os.path.join(
            chunk_file_base, "detections_chunk_{}.json".format(str(i_chunk).zfill(4))
        )
        instance_chunk_file = os.path.join(
            chunk_file_base, "instances_chunk_{}.json".format(str(i_chunk).zfill(4))
        )

        detection_chunk_dict = {"predictions": detection_chunk}
        instances_chunk_dict = {"instances": instance_chunk}

        with open(detection_chunk_file, "w") as f:
            json.dump(detection_chunk_dict, f, indent=1)
        with open(instance_chunk_file, "w") as f:
            json.dump(instances_chunk_dict, f, indent=1)

        detection_chunk_files.append(detection_chunk_file)
        instances_chunk_files.append(instance_chunk_file)

    # Assemble commands for new processes
    classification_chunk_files = []
    classification_commands = []

    # These are the arguments we don't want to pass to child processes
    exclude_args = [
        "--predictions_json",
        "--instances_json",
        "--detections_json",
        "--filepaths",
        "--filepaths_txt",
        "--folders",
        "--chunk_size",
    ]
    all_args = sys.argv

    filtered_args = _create_argument_list_with_exceptions(all_args, exclude_args)

    for i_chunk in range(0, n_chunks):

        chunk_cmd = "python -m speciesnet.scripts.run_model"
        for arg in filtered_args:
            chunk_cmd += f" {arg}"
        instance_chunk_file = instances_chunk_files[i_chunk]
        detection_chunk_file = detection_chunk_files[i_chunk]
        classification_chunk_file = os.path.join(
            chunk_file_base,
            "classifications_chunk_{}.json".format(str(i_chunk).zfill(4)),
        )
        classification_chunk_files.append(classification_chunk_file)
        chunk_cmd += f" --instances_json {instance_chunk_file}"
        chunk_cmd += f" --detections_json {detection_chunk_file}"
        chunk_cmd += f" --predictions_json {classification_chunk_file}"
        classification_commands.append(chunk_cmd)

    # Run commands
    for i_chunk in range(0, n_chunks):
        print("Running command for chunk {}: {}".format(i_chunk, chunk_cmd))
        _execute_and_print(classification_commands[i_chunk])
        assert os.path.isfile(classification_chunk_files[i_chunk])

    # Merge results
    output_predictions = []
    for i_chunk in range(0, n_chunks):
        with open(classification_chunk_files[i_chunk], "r") as f:
            chunk_results = json.load(f)
        output_predictions.extend(chunk_results["predictions"])

    print("Read {} predictions".format(len(output_predictions)))

    output_dict = {"predictions": output_predictions}
    with open(_PREDICTIONS_JSON.value, "w") as f:
        json.dump(output_dict, f, indent=1)


if __name__ == "__main__":
    app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
