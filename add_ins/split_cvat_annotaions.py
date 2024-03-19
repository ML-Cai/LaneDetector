import xml.etree.ElementTree as ET
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', required=True, help='The path to the CVAT annotation file')
    args = parser.parse_args()
