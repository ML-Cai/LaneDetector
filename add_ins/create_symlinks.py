import os
import argparse


def main():
    argparser = argparse.ArgumentParser(description='Create symlinks for the dataset')
    argparser.add_argument('dataset_path', type=str, help='path to the dataset')
    argparser.add_argument("image_path", type=str, help="path to the images")
    args = argparser.parse_args()
    dataset_path = os.path.abspath(args.dataset_path)
    image_path = os.path.abspath(args.image_path)

    os.makedirs(os.path.join(dataset_path, "test_set"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "train_set"), exist_ok=True)

    for root, dirs, files in os.walk(image_path):
        for folder in dirs:
            os.symlink(os.path.join(root, folder), os.path.join(dataset_path, "train_set", folder))
            os.symlink(os.path.join(root, folder), os.path.join(dataset_path, "test_set", folder))
        # for folder in dirs[:int(len(dirs) * 0.8)]:  # 80% train
        #     os.symlink(os.path.join(root, folder), os.path.join(dataset_path, "train_set", folder))
        # for folder in dirs[int(len(dirs) * 0.8):]:
        #     os.symlink(os.path.join(root, folder), os.path.join(dataset_path, "test_set", folder))


if __name__ == "__main__":
    main()
