import os
import csv
import glob
import tqdm
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import pandas as pd


def get_test_image_names(folder_path="SrcData/Test", verbose=False):
    if not os.path.isdir(folder_path):
        print(f"ERROR: The folder '{folder_path}' does not exist.")
        return []

    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))

    image_names = [os.path.basename(f) for f in jpg_files]
    sorted_image_names = sorted(image_names, key=lambda x: int(x.split(".")[0]))
    if verbose:
        print(f"Found {len(sorted_image_names)} test images in '{folder_path}'.")
    return sorted_image_names


def resize_to_multiple_of(image, multiple, verbose=False):
    original_width, original_height = image.size

    target_width_factor = round(original_width / multiple)
    if target_width_factor == 0:
        target_width_factor = 1
    target_width = int(target_width_factor * multiple)

    target_height_factor = round(original_height / multiple)
    if target_height_factor == 0:
        target_height_factor = 1
    target_height = int(target_height_factor * multiple)
    if verbose:
        print(
            f"  Original size: ({original_width}, {original_height}) -> Target size: ({target_width}, {target_height})"
        )
    resized_image = image.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    return resized_image


def divide_into_patches(image_name, image, patch_size, final_patch_size, verbose=False):
    width, height = image.size
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patch = patch.resize(
                (final_patch_size, final_patch_size), Image.Resampling.LANCZOS
            )
            patches.append(patch)
    if verbose:
        print(
            f"  Image '{image_name}' divided into {len(patches)} patches of size {patch_size}x{patch_size}."
        )
    return patches


def test(
    test_image_folder="SrcData/Test",
    patch_size=640,
    final_patch_size=640,
    verbose=True,
    CONFIDENCE_THRESHOLD=0.5,
    NMS_IOU_THRESHOLD=0.7,
    border_margin=5,
    model_path="model_ckpt/epoch47.pt",
    output_dir="test_result",
    submission_filename="submission.csv",
):
    sea_lion_class_name_map = {
        0: "adult_males",
        1: "subadult_males",
        2: "adult_females",
        3: "juveniles",
        4: "pups",
    }
    csv_column_order = [
        "test_id",
        "adult_males",
        "subadult_males",
        "adult_females",
        "juveniles",
        "pups",
    ]

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        exit()
    if verbose:
        print(f"Loading model from '{model_path}'...")
    model = YOLO(model_path)
    if verbose:
        print("Model loaded successfully.")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Moving model to {device} and setting to evaluation mode.")
        model.to(device)
    else:
        device = torch.device("cpu")
        print(
            f"CUDA not available. Using {device} and setting model to evaluation mode."
        )
        model.to(device)
    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        model.model.eval()

    output_csv_file = os.path.join(output_dir, submission_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    with open(output_csv_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_column_order)

    image_names_list = get_test_image_names(test_image_folder, verbose=verbose)
    bar = tqdm.tqdm(image_names_list, ncols=120) if not verbose else image_names_list
    for image_name in bar:
        image_id_str = os.path.splitext(image_name)[0]
        try:
            image_id = int(image_id_str)
        except ValueError:
            print(
                f"Warning: Could not convert image name '{image_id_str}' to an integer ID. Pausing the program."
            )
            exit()
        image_path = os.path.join(test_image_folder, image_name)
        img = Image.open(image_path)
        resized_img = resize_to_multiple_of(img, patch_size, verbose=verbose)
        image_patches = divide_into_patches(
            image_name, resized_img, patch_size, final_patch_size, verbose=verbose
        )

        current_image_sea_lion_counts = {
            "adult_males": 0,
            "subadult_males": 0,
            "adult_females": 0,
            "juveniles": 0,
            "pups": 0,
        }
        for patch_idx, patch_img in enumerate(image_patches):
            results = model.predict(
                source=patch_img,
                imgsz=final_patch_size,
                conf=CONFIDENCE_THRESHOLD,
                iou=NMS_IOU_THRESHOLD,
                verbose=False,
            )
            if results and len(results) > 0:
                pred_boxes = results[0].boxes
                for box in pred_boxes:
                    cls_idx = int(box.cls.item())
                    if cls_idx in sea_lion_class_name_map:
                        class_name = sea_lion_class_name_map[cls_idx]
                        coords = box.xyxy[0].tolist()
                        xmin, ymin, xmax, ymax = coords
                        count_value = 1.0
                        edges_on_border = 0
                        if xmin < border_margin:  # Close to left edge
                            edges_on_border += 1
                        if ymin < border_margin:  # Close to top edge
                            edges_on_border += 1
                        if xmax > (
                            final_patch_size - 1 - border_margin
                        ):  # Close to right edge
                            edges_on_border += 1
                        if ymax > (
                            final_patch_size - 1 - border_margin
                        ):  # Close to bottom edge
                            edges_on_border += 1
                        if edges_on_border == 1:
                            count_value = 0.7
                        elif edges_on_border >= 2:
                            count_value = 0.5
                        current_image_sea_lion_counts[class_name] += count_value
                    else:
                        print(
                            f"Warning: Detected unknown class index {cls_idx} in patch {patch_idx} of {image_name}"
                        )

        row_data = [image_id] + [
            int(round(current_image_sea_lion_counts.get(class_name, 0.0)))
            for class_name in csv_column_order[1:]
        ]
        with open(output_csv_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)

        if verbose:
            formatted_counts = {
                k: f"{v:.1f}" for k, v in current_image_sea_lion_counts.items()
            }
            print(f"  Aggregated counts for image ID '{image_id}': {formatted_counts}")

    if verbose:
        print("All images processed successfully.")


def check_test_ids(file_path="test_result/submission.csv"):
    try:
        df = pd.read_csv(file_path)
        if "test_id" not in df.columns:
            print("ERROR: 'test_id' column not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return

        all_ids = df["test_id"].dropna().astype(int).tolist()
        existing_ids = set(all_ids)
        duplicates = []
        seen = set()

        for test_id in all_ids:
            if test_id in seen:
                if test_id not in duplicates:
                    duplicates.append(test_id)
            else:
                seen.add(test_id)

        complete_range = set(range(0, 18636))
        missing_ids = complete_range - existing_ids

        print(f"CSV file contains {len(all_ids)} total entries")
        print(f"CSV file contains {len(existing_ids)} unique test_ids")
        print(f"Should have {len(complete_range)} test_ids (0-18635)")
        print(f"Missing {len(missing_ids)} test_ids")
        print(f"Found {len(duplicates)} duplicate test_ids")

        if duplicates:
            duplicates_sorted = sorted(duplicates)
            print(f"\nDuplicate test_ids:")
            print(duplicates_sorted)

            print(f"\nDuplicate count details:")
            for dup_id in duplicates_sorted:
                count = all_ids.count(dup_id)
                print(f"  test_id {dup_id}: appears {count} times")

        if missing_ids:
            missing_list = sorted(list(missing_ids))
            print(f"\nMissing test_ids:")
            print(missing_list)

        if not missing_ids and not duplicates:
            print("\n✅ All test_ids are present and unique!")
        elif not missing_ids:
            print("\n⚠️  All test_ids are present but some are duplicated!")
        elif not duplicates:
            print("\n⚠️  No duplicates found but some test_ids are missing!")
        else:
            print("\n❌ Both missing and duplicate test_ids found!")

    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found")
        print("Please check if the file path is correct")
    except Exception as e:
        print(f"ERROR occurred: {e}")


if __name__ == "__main__":
    # test(
    #     test_image_folder="SrcData/Test",
    #     patch_size=1440,
    #     final_patch_size=640,
    #     # verbose=True,
    #     verbose=False,
    #     CONFIDENCE_THRESHOLD=0.22,
    #     NMS_IOU_THRESHOLD=0.7,
    #     border_margin=8,
    #     model_path="model_ckpt/epoch99.pt",
    #     output_dir="test_result",
    #     submission_filename="submission.csv",
    # )
    check_test_ids("test_result/submission(8).csv")
