from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent

SRC_IMAGES = ROOT / "RDD2022_China_Drone" / "images"
SRC_XML = ROOT / "RDD2022_China_Drone" / "annotations" / "xmls"
OUT = ROOT / "yolo_dataset"

CLASSES = [
    "Block crack",
    "D00",
    "D10",
    "D20",
    "D40",
    "Repair"
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

random.seed(42)


def find_image(stem):
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        image = SRC_IMAGES / (stem + ext)
        if image.exists():
            return image
    return None


def convert_xml(xml_file):

    root = ET.parse(xml_file).getroot()

    size = root.find("size")

    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    labels = []

    for obj in root.findall("object"):

        class_name = (obj.findtext("name") or "").strip()

        if class_name not in CLASS_TO_ID:
            print("Unknown class:", class_name)
            continue

        box = obj.find("bndbox")

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        # Keep boxes inside image
        xmin = max(0, min(xmin, width))
        xmax = max(0, min(xmax, width))
        ymin = max(0, min(ymin, height))
        ymax = max(0, min(ymax, height))

        if xmax <= xmin or ymax <= ymin:
            continue

        # Pascal VOC -> YOLO
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height

        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        class_id = CLASS_TO_ID[class_name]

        labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    return labels


def main():

    print("Starting YOLO dataset conversion...")
    print()

    if not SRC_IMAGES.exists():
        print("ERROR: Images folder not found:")
        print(SRC_IMAGES)
        return

    if not SRC_XML.exists():
        print("ERROR: XML folder not found:")
        print(SRC_XML)
        return

    xml_files = sorted(SRC_XML.glob("*.xml"))

    print("XML files found:", len(xml_files))

    pairs = []

    for xml_file in xml_files:

        image = find_image(xml_file.stem)

        if image is None:
            print("Missing image:", xml_file.name)
            continue

        pairs.append((xml_file, image))

    print("Matching image/XML pairs:", len(pairs))

    random.shuffle(pairs)

    train_count = int(len(pairs) * 0.8)

    train_pairs = pairs[:train_count]
    val_pairs = pairs[train_count:]

    # Remove old dataset if it exists
    if OUT.exists():
        shutil.rmtree(OUT)

    # Create folders
    for split in ["train", "val"]:

        (OUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    object_counts = {name: 0 for name in CLASSES}

    for split, data in [
        ("train", train_pairs),
        ("val", val_pairs)
    ]:

        print()
        print("Creating", split, "dataset...")

        for xml_file, image_file in data:

            # Copy image
            destination_image = (
                OUT / "images" / split / image_file.name
            )

            shutil.copy2(
                image_file,
                destination_image
            )

            # Convert annotation
            labels = convert_xml(xml_file)

            destination_label = (
                OUT / "labels" / split /
                (image_file.stem + ".txt")
            )

            with open(
                destination_label,
                "w",
                encoding="utf-8"
            ) as f:

                for label in labels:
                    f.write(label + "\n")

            # Count classes
            for label in labels:

                class_id = int(label.split()[0])

                object_counts[
                    CLASSES[class_id]
                ] += 1

    # Create data.yaml
    yaml_file = OUT / "data.yaml"

    with open(
        yaml_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"path: {OUT.as_posix()}\n"
        )

        f.write("train: images/train\n")
        f.write("val: images/val\n\n")

        f.write("names:\n")

        for i, name in enumerate(CLASSES):

            f.write(
                f"  {i}: {name}\n"
            )

    print()
    print("=" * 50)
    print("YOLO DATASET CREATED")
    print("=" * 50)

    print()
    print("Training images:", len(train_pairs))
    print("Validation images:", len(val_pairs))

    print()
    print("Objects:")

    for name in CLASSES:
        print(
            f"{name}: {object_counts[name]}"
        )

    print()
    print("Dataset location:")
    print(OUT)

    print()
    print("YAML:")
    print(yaml_file)

    print()
    print("Done.")


if __name__ == "__main__":
    main()