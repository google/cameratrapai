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

"""Script to convert SpeciesNet .json format to MegaDetector/Timelapse .json format."""

import argparse
import json
import sys

blank_prediction_string = "f1856211-cfb7-4a5b-9158-c0f72fd09ee6;;;;;;blank"
no_cv_result_prediction_string = "f2efdae9-efb8-48fb-8a91-eccf79ab4ffb;no cv result;no cv result;no cv result;no cv result;no cv result;no cv result"
rodent_prediction_string = (
    "90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent"
)
mammal_prediction_string = "f2d233e3-80e3-433d-9687-e29ecc7a467a;mammalia;;;;;mammal"
animal_prediction_string = "1f689929-883d-4dae-958c-3d57ab5b6c16;;;;;;animal"
human_prediction_string = "990ae9dd-7a59-4344-afcb-1b7b21368000;mammalia;primates;hominidae;homo;sapiens;human"


def invert_dictionary(d):
    """
    Creates a new dictionary that maps d.values() to d.keys().  Does not check
    uniqueness.

    Args:
        d (dict): dictionary to invert

    Returns:
        dict: inverted copy of [d]
    """

    return {v: k for k, v in d.items()}


def sort_list_of_dicts_by_key(L, k, reverse=False):
    """
    Sorts the list of dictionaries [L] by the key [k].

    Args:
        L (list): list of dictionaries to sort
        k (object, typically str): the sort key
        reverse (bool, optional): whether to sort in reverse (descending) order

    Returns:
        dict: sorted copy of [d]
    """
    return sorted(L, key=lambda d: d[k], reverse=reverse)


def is_list_sorted(L, reverse=False):
    """
    Returns True if the list L appears to be sorted, otherwise False.

    Calling is_list_sorted(L,reverse=True) is the same as calling
    is_list_sorted(L.reverse(),reverse=False).

    Args:
        L (list): list to evaluate
        reverse (bool, optional): whether to reverse the list before evaluating sort status

    Returns:
        bool: True if the list L appears to be sorted, otherwise False
    """

    if reverse:
        return all(L[i] >= L[i + 1] for i in range(len(L) - 1))
    else:
        return all(L[i] <= L[i + 1] for i in range(len(L) - 1))


def get_common_name_from_prediction_string(s):
    """
    Extract the common name from the seven-token prediction string [s], or generate
    a reasonable one (e.g. "vulpes genus").  Prediction strings look like:

    '90d950db-2106-4bd9-a4c1-777604c3eada;mammalia;rodentia;;;;rodent'

    Args:
        s (str): the string for which we should extract a common name

    Returns:
        str: the extracted common name
    """

    s = s.strip()
    tokens = s.split(";")
    assert len(tokens) == 7, "Invalid prediction string: {}".format(s)
    tokens = [token.strip() for token in tokens]

    common_name = None

    i_token = 6

    while i_token >= 0:

        token = tokens[i_token]

        if len(token) > 0:

            # Non-empty common name
            if i_token == 6:
                common_name = token
            # Non-empty species name
            elif i_token == 5:
                # If a species name is present, a genus name should always be present
                if len(tokens[4]) == 0:
                    print("Warning: invalid genus/species combination {}".format(s))
                    common_name = tokens[5]
                else:
                    # Create binomial name
                    common_name = tokens[4] + " " + tokens[5]
            # Non-empty genus name
            elif i_token == 4:
                # Not a typo, this is a convention, genus-level names end with "species"
                common_name = tokens[4] + " species"
            # Non-empty family name
            elif i_token == 3:
                common_name = tokens[3] + " family"
            # Non-empty order name
            elif i_token == 2:
                common_name = tokens[2] + " order"
            # Non-empty class name
            elif i_token == 1:
                common_name = tokens[1] + " class"
            # Non-empty ID
            elif i_token == 0:
                common_name = tokens[0] + " category"
            break

        # ...if this token is non-empty

        i_token -= 1

    # ...for each token

    if common_name is None:
        assert s == ";;;;;;"
        common_name = "empty_prediction_string"

    return common_name


def generate_md_results_from_predictions_json(
    predictions_json_file,
    md_results_file,
    base_folder=None,
    convert_homo_species_to_human=True,
):
    """
    Generate an MD-formatted .json file from a predictions.json file.  Typically,
    MD results files use relative paths, and predictions.json files use absolute paths, so
    this function optionally removes the leading string [base_folder] from all file names.

    Currently just applies the top classification category to every detection.  If the top classification
    is "blank", writes an empty detection list.

    speciesnet_to_md.py is a command-line driver for this function.

    Args:
        predictions_json_file (str): path to a predictions.json file
        md_results_file (str): path to which we should write an MD-formatted .json file
        base_folder (str, optional): leading string to remove from each path in the predictions.json file
        convert_homo_species_to_human (bool, optional): the ensemble often rolls human
            predictions up to "homo species", which isn't wrong, but looks odd.  This forces
            these back to "homo sapiens".
    """

    # Read predictions file
    with open(predictions_json_file, "r") as f:
        predictions = json.load(f)
    predictions = predictions["predictions"]
    assert isinstance(predictions, list)

    # Convert backslashes to forward slashes in both filenames and the base folder string
    for im in predictions:
        im["filepath"] = im["filepath"].replace("\\", "/")
    if base_folder is not None:
        base_folder = base_folder.replace("\\", "/")

    detection_category_id_to_name = {}
    classification_category_name_to_id = {}

    # Keep track of detections that don't have an assigned detection category; these
    # are fake detections we create for non-blank images with non-empty detection lists.
    # We need to go back later and give them a legitimate detection category ID.
    all_unknown_detections = []

    # Create the output images list
    images_out = []

    # im_in = predictions[0]
    for im_in in predictions:

        # blank_prediction_string
        im_out = {}

        fn = im_in["filepath"]
        if base_folder is not None:
            if fn.startswith(base_folder):
                fn = fn.replace(base_folder, "", 1)
            elif fn.startswith(base_folder + "/"):
                fn = fn.replace(base_folder + "/", "", 1)

        im_out["file"] = fn

        if "failures" in im_in:

            im_out["failure"] = str(im_in["failures"])
            im_out["detections"] = None

        else:

            im_out["detections"] = []

            if "detections" in im_in:

                if len(im_in["detections"]) == 0:
                    im_out["detections"] = []
                else:
                    # det_in = im_in['detections'][0]
                    for det_in in im_in["detections"]:
                        det_out = {}
                        if det_in["category"] in detection_category_id_to_name:
                            assert (
                                detection_category_id_to_name[det_in["category"]]
                                == det_in["label"]
                            )
                        else:
                            detection_category_id_to_name[det_in["category"]] = det_in[
                                "label"
                            ]
                        det_out = {}
                        for s in ["category", "conf", "bbox"]:
                            det_out[s] = det_in[s]
                        im_out["detections"].append(det_out)

            # ...if detections are present

            class_to_assign = None
            class_confidence = None

            if "classifications" in im_in:

                classifications = im_in["classifications"]
                assert len(classifications["scores"]) == len(classifications["classes"])
                assert is_list_sorted(classifications["scores"], reverse=True)
                class_to_assign = classifications["classes"][0]
                class_confidence = classifications["scores"][0]

            if "prediction" in im_in:

                class_to_assign = im_in["prediction"]
                if convert_homo_species_to_human and (
                    ("hominidae;homo" in class_to_assign)
                    or ("homo species") in class_to_assign
                ):
                    class_to_assign = human_prediction_string
                class_confidence = im_in["prediction_score"]

            if class_to_assign is not None:

                if class_to_assign == blank_prediction_string:

                    # This is a scenario that's not captured well by the MD format: a blank prediction
                    # with detections present.  But, for now, don't do anything special here, just making
                    # a note of this.
                    if len(im_out["detections"]) > 0:
                        pass

                else:

                    assert not class_to_assign.endswith("blank")

                    # This is a scenario that's not captured well by the MD format: no detections present,
                    # but a non-blank prediction.  For now, create a fake detection to handle this prediction.
                    if len(im_out["detections"]) == 0:

                        print(
                            "Warning: creating fake detection for non-blank whole-image classification"
                        )
                        det_out = {}
                        all_unknown_detections.append(det_out)

                        # We will change this to a string-int later
                        det_out["category"] = "unknown"
                        det_out["conf"] = class_confidence
                        det_out["bbox"] = [0, 0, 1, 1]
                        im_out["detections"].append(det_out)

                # ...if this is/isn't a blank classification

                # Attach that classification to each detection

                # Create a new category ID if necessary
                if class_to_assign in classification_category_name_to_id:
                    classification_category_id = classification_category_name_to_id[
                        class_to_assign
                    ]
                else:
                    classification_category_id = str(
                        len(classification_category_name_to_id)
                    )
                    classification_category_name_to_id[class_to_assign] = (
                        classification_category_id
                    )

                for det in im_out["detections"]:
                    det["classifications"] = []
                    det["classifications"].append(
                        [classification_category_id, class_confidence]
                    )

            # ...if we have some type of classification for this image

        # ...if this is/isn't a failure

        images_out.append(im_out)

    # ...for each image

    # Fix the 'unknown' category

    if len(all_unknown_detections) > 0:

        max_detection_category_id = max(
            [int(x) for x in detection_category_id_to_name.keys()]
        )
        unknown_category_id = str(max_detection_category_id + 1)
        detection_category_id_to_name[unknown_category_id] = "unknown"

        for det in all_unknown_detections:
            assert det["category"] == "unknown"
            det["category"] = unknown_category_id

    # Sort by filename

    images_out = sort_list_of_dicts_by_key(images_out, "file")

    # Prepare friendly classification names

    classification_category_descriptions = invert_dictionary(
        classification_category_name_to_id
    )
    classification_categories_out = {}
    for category_id in classification_category_descriptions.keys():
        category_name = get_common_name_from_prediction_string(
            classification_category_descriptions[category_id]
        )
        classification_categories_out[category_id] = category_name

    # Prepare the output dict

    detection_categories_out = detection_category_id_to_name
    info = {}
    info["format_version"] = 1.4
    info["detector"] = "converted_from_predictions_json"

    output_dict = {}
    output_dict["info"] = info
    output_dict["detection_categories"] = detection_categories_out
    output_dict["classification_categories"] = classification_categories_out
    output_dict["classification_category_descriptions"] = (
        classification_category_descriptions
    )
    output_dict["images"] = images_out

    with open(md_results_file, "w") as f:
        json.dump(output_dict, f, indent=1)

    # TODO: ideally we would validate the output, but this requires a lot more imports,
    # so deferring this.
    """
    validation_options = ValidateBatchResultsOptions()
    validation_options.raise_errors = True
    _ = validate_batch_results(md_results_file, options=validation_options)
    """


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions_json_file",
        action="store",
        type=str,
        help=".json file to convert from SpeciesNet predictions.json format to MD format",
    )
    parser.add_argument(
        "md_results_file",
        action="store",
        type=str,
        help="output file to write in MD format",
    )
    parser.add_argument(
        "--base_folder",
        action="store",
        type=str,
        default=None,
        help="leading string to remove from each path in the predictions.json "
        + "file (to convert from absolute to relative paths)",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    generate_md_results_from_predictions_json(
        args.predictions_json_file, args.md_results_file, args.base_folder
    )


if __name__ == "__main__":
    main()
