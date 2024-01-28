import argparse
import os


def main():
    parser = argparse.ArgumentParser("Split dataset into train and test set")
    parser.add_argument('annotation_path', type=str, help='path to the annotation file')
    args = parser.parse_args()
    annotation_path: str = args.annotation_path
    annotation_dir = str(os.path.dirname(annotation_path))

    lanes = [line for line in open(annotation_path, 'r')]
    # train_set = [lane for lane in lanes if not any([n in lane for n in ["2023-10-02-13-51-53", "2023-10-02-13-52-29", "2023-10-04-17-25-29"]])]
    # test_set = [lane for lane in lanes if any([n in lane for n in ["2023-10-02-13-51-53", "2023-10-02-13-52-29", "2023-10-04-17-25-29"]])]
    train_set = lanes[:int(len(lanes) * 0.8)]
    test_set = lanes[int(len(lanes) * 0.8):]
    with open(os.path.join(annotation_dir, "train_set.json"), 'w') as f:
        for line in train_set:
            f.write(line)

    with open(os.path.join(annotation_dir, "test_set.json"), 'w') as f:
        for line in test_set:
            f.write(line)


if __name__ == "__main__":
    main()
