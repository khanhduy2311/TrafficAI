import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a YOLO detection dataset with class-aware balancing."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("path/to/data_fisheye"),
        help="Dataset root containing images/, labels/, and optionally classes.txt",
    )
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--labels-dir", default="labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--dataset-yaml-name", default="dataset.yaml")
    parser.add_argument("--stats-name", default="split_stats.yaml")
    parser.add_argument("--balanced-train-name", default="train_balanced.txt")
    parser.add_argument("--balanced-dataset-yaml-name", default="dataset_balanced.yaml")
    parser.add_argument("--balance-target-ratio", type=float, default=0.8)
    parser.add_argument("--max-repeat", type=int, default=4)
    return parser.parse_args()


def validate_ratios(train_ratio, val_ratio, test_ratio):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")


def load_class_names(data_dir):
    classes_txt = data_dir / "classes.txt"
    if not classes_txt.exists():
        return []
    with classes_txt.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def find_images(images_dir):
    return sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_label_counts(label_path):
    counts = Counter()
    if not label_path.exists():
        return counts

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls_id = int(float(parts[0]))
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Invalid label format in {label_path} at line {line_number}: {raw_line!r}"
                ) from exc
            counts[cls_id] += 1
    return counts


def build_samples(images, images_dir, labels_dir):
    samples = []
    total_object_counts = Counter()

    for image_path in images:
        rel_path = image_path.relative_to(images_dir)
        label_path = (labels_dir / rel_path).with_suffix(".txt")
        class_counts = read_label_counts(label_path)
        total_object_counts.update(class_counts)
        samples.append(
            {
                "image_path": image_path,
                "class_counts": class_counts,
                "classes_present": set(class_counts),
                "object_count": sum(class_counts.values()),
            }
        )

    return samples, total_object_counts


def compute_target_image_counts(num_samples, ratios):
    split_names = list(ratios)
    raw_targets = {split: num_samples * ratio for split, ratio in ratios.items()}
    target_counts = {split: int(raw_targets[split]) for split in split_names}

    remaining = num_samples - sum(target_counts.values())
    if remaining > 0:
        ranked = sorted(
            split_names,
            key=lambda split: (raw_targets[split] - target_counts[split], ratios[split]),
            reverse=True,
        )
        for split in ranked[:remaining]:
            target_counts[split] += 1
    return target_counts


def compute_target_object_counts(total_object_counts, ratios):
    return {
        split: {cls_id: total * ratio for cls_id, total in total_object_counts.items()}
        for split, ratio in ratios.items()
    }


def rarity_score(sample, class_image_frequency):
    if not sample["classes_present"]:
        return 0.0
    return sum(1.0 / class_image_frequency[cls_id] for cls_id in sample["classes_present"])


def score_split(split_name, sample, split_state, image_targets, object_targets):
    image_capacity = image_targets[split_name] - split_state[split_name]["num_images"]
    if image_capacity <= 0:
        return float("-inf")

    score = image_capacity * 1000.0
    for cls_id, count in sample["class_counts"].items():
        current = split_state[split_name]["object_counts"][cls_id]
        target = object_targets[split_name].get(cls_id, 0.0)
        deficit = target - current
        score += min(deficit, count) * 10.0
        score -= max(0.0, current + count - target) * 2.0

    score -= split_state[split_name]["object_total"] * 0.001
    return score


def assign_samples(samples, ratios, seed):
    random_generator = random.Random(seed)
    shuffled_samples = list(samples)
    random_generator.shuffle(shuffled_samples)

    class_image_frequency = Counter()
    total_object_counts = Counter()
    for sample in shuffled_samples:
        class_image_frequency.update(sample["classes_present"])
        total_object_counts.update(sample["class_counts"])

    ordered_samples = sorted(
        shuffled_samples,
        key=lambda sample: (rarity_score(sample, class_image_frequency), sample["object_count"]),
        reverse=True,
    )

    image_targets = compute_target_image_counts(len(samples), ratios)
    object_targets = compute_target_object_counts(total_object_counts, ratios)
    split_state = {
        split: {
            "samples": [],
            "num_images": 0,
            "object_total": 0,
            "object_counts": Counter(),
        }
        for split in ratios
    }

    split_names = list(ratios)
    for sample in ordered_samples:
        ranked_splits = sorted(
            split_names,
            key=lambda split: (
                score_split(split, sample, split_state, image_targets, object_targets),
                image_targets[split] - split_state[split]["num_images"],
            ),
            reverse=True,
        )
        chosen_split = ranked_splits[0]
        state = split_state[chosen_split]
        state["samples"].append(sample)
        state["num_images"] += 1
        state["object_total"] += sample["object_count"]
        state["object_counts"].update(sample["class_counts"])

    return split_state, image_targets


def ensure_class_names(class_names, observed_class_ids):
    if not observed_class_ids:
        return class_names
    names = list(class_names)
    while len(names) <= max(observed_class_ids):
        names.append(f"class_{len(names)}")
    return names


def write_split_file(output_path, samples):
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(f"{sample['image_path'].resolve()}\n")


def compute_repeat_factors(train_samples, balance_target_ratio, max_repeat):
    train_counts = Counter()
    for sample in train_samples:
        train_counts.update(sample["class_counts"])

    if not train_counts:
        return {}, train_counts, Counter()

    max_count = max(train_counts.values())
    target_floor = max_count * balance_target_ratio
    repeat_factors = {}
    balanced_counts = Counter()

    for sample in train_samples:
        factor = 1
        desired_repeats = []
        for cls_id in sample["classes_present"]:
            current_count = train_counts[cls_id]
            if current_count < target_floor:
                desired_repeats.append(math.ceil(target_floor / current_count))
        if desired_repeats:
            factor = min(max_repeat, max(desired_repeats))
        repeat_factors[sample["image_path"]] = factor
        for cls_id, count in sample["class_counts"].items():
            balanced_counts[cls_id] += count * factor

    return repeat_factors, train_counts, balanced_counts


def write_balanced_train_file(output_path, train_samples, repeat_factors):
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in train_samples:
            for _ in range(repeat_factors.get(sample["image_path"], 1)):
                handle.write(f"{sample['image_path'].resolve()}\n")


def build_dataset_yaml(data_dir, train_file_name, class_names):
    return {
        "path": str(data_dir),
        "train": train_file_name,
        "val": "val.txt",
        "test": "test.txt",
        "names": {i: name for i, name in enumerate(class_names)},
    }


def build_stats(split_state, image_targets, class_names, balance_info):
    stats = {"splits": {}, "targets": image_targets}

    overall_counts = Counter()
    overall_images = 0
    for split_name, state in split_state.items():
        stats["splits"][split_name] = {
            "num_images": state["num_images"],
            "num_objects": state["object_total"],
            "class_counts": {
                class_names[cls_id]: state["object_counts"].get(cls_id, 0)
                for cls_id in range(len(class_names))
            },
        }
        overall_counts.update(state["object_counts"])
        overall_images += state["num_images"]

    stats["overall"] = {
        "num_images": overall_images,
        "num_objects": sum(overall_counts.values()),
        "class_counts": {
            class_names[cls_id]: overall_counts.get(cls_id, 0) for cls_id in range(len(class_names))
        },
    }

    distribution = defaultdict(dict)
    for cls_id, class_name in enumerate(class_names):
        total = overall_counts.get(cls_id, 0)
        for split_name, state in split_state.items():
            split_count = state["object_counts"].get(cls_id, 0)
            distribution[class_name][split_name] = round(split_count / total, 4) if total else 0.0
    stats["class_distribution_ratio"] = dict(distribution)

    stats["balanced_train"] = {
        "train_file": balance_info["train_file"],
        "dataset_yaml": balance_info["dataset_yaml"],
        "target_ratio": balance_info["target_ratio"],
        "max_repeat": balance_info["max_repeat"],
        "repeated_images": balance_info["repeated_images"],
        "original_num_images": balance_info["original_num_images"],
        "balanced_num_images": balance_info["balanced_num_images"],
        "class_counts_before": {
            class_names[cls_id]: balance_info["before_counts"].get(cls_id, 0)
            for cls_id in range(len(class_names))
        },
        "class_counts_after": {
            class_names[cls_id]: balance_info["after_counts"].get(cls_id, 0)
            for cls_id in range(len(class_names))
        },
    }
    return stats


def print_summary(split_state, class_names, balance_info):
    print("Split summary:")
    for split_name, state in split_state.items():
        print(f"- {split_name}: {state['num_images']} images, {state['object_total']} objects")
        for cls_id in range(len(class_names)):
            print(f"    {class_names[cls_id]}: {state['object_counts'].get(cls_id, 0)}")

    print("\nBalanced training summary:")
    print(
        f"- images: {balance_info['original_num_images']} -> {balance_info['balanced_num_images']}"
    )
    print(f"- repeated minority-class images: {balance_info['repeated_images']}")
    for cls_id in range(len(class_names)):
        print(
            f"    {class_names[cls_id]}: "
            f"{balance_info['before_counts'].get(cls_id, 0)} -> "
            f"{balance_info['after_counts'].get(cls_id, 0)}"
        )


def main():
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    data_dir = args.data_dir.resolve()
    images_dir = (data_dir / args.images_dir).resolve()
    labels_dir = (data_dir / args.labels_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = find_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    class_names = load_class_names(data_dir)
    samples, observed_object_counts = build_samples(images, images_dir, labels_dir)
    class_names = ensure_class_names(class_names, observed_object_counts.keys())

    ratios = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    split_state, image_targets = assign_samples(samples, ratios, args.seed)

    for split_name, state in split_state.items():
        write_split_file(data_dir / f"{split_name}.txt", state["samples"])

    config_dir = Path(__file__).resolve().parent / "configs"
    config_dir.mkdir(exist_ok=True)

    dataset_yaml = build_dataset_yaml(data_dir, "train.txt", class_names)
    with (data_dir / args.dataset_yaml_name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False, allow_unicode=True)
    with (config_dir / args.dataset_yaml_name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False, allow_unicode=True)

    repeat_factors, train_before_counts, train_after_counts = compute_repeat_factors(
        split_state["train"]["samples"],
        balance_target_ratio=args.balance_target_ratio,
        max_repeat=args.max_repeat,
    )

    write_balanced_train_file(
        data_dir / args.balanced_train_name,
        split_state["train"]["samples"],
        repeat_factors,
    )

    balanced_dataset_yaml = build_dataset_yaml(data_dir, args.balanced_train_name, class_names)
    with (data_dir / args.balanced_dataset_yaml_name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(balanced_dataset_yaml, handle, sort_keys=False, allow_unicode=True)
    with (config_dir / args.balanced_dataset_yaml_name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(balanced_dataset_yaml, handle, sort_keys=False, allow_unicode=True)

    balance_info = {
        "train_file": args.balanced_train_name,
        "dataset_yaml": args.balanced_dataset_yaml_name,
        "target_ratio": args.balance_target_ratio,
        "max_repeat": args.max_repeat,
        "repeated_images": sum(1 for factor in repeat_factors.values() if factor > 1),
        "original_num_images": len(split_state["train"]["samples"]),
        "balanced_num_images": sum(repeat_factors.values()),
        "before_counts": train_before_counts,
        "after_counts": train_after_counts,
    }

    stats = build_stats(split_state, image_targets, class_names, balance_info)
    with (data_dir / args.stats_name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(stats, handle, sort_keys=False, allow_unicode=True)

    print_summary(split_state, class_names, balance_info)
    print(f"Saved dataset config to: {data_dir / args.dataset_yaml_name}")
    print(f"Saved balanced dataset config to: {data_dir / args.balanced_dataset_yaml_name}")
    print(f"Saved split statistics to: {data_dir / args.stats_name}")


if __name__ == "__main__":
    main()
