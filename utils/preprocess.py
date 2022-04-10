import json
import pickle

import cv2
import numpy as np
from turbojpeg import TurboJPEG

images = '../mtsd_fully_annotated_images'
annotations = '../mtsd_fully_annotated_json/mtsd_v2_fully_annotated/annotations'
splits = '../mtsd_fully_annotated_json/mtsd_v2_fully_annotated/splits'


def extract_image(image_path, bbox):
    """
    Extracts a bounding box from the full JPEG image
    The bbox is then resized and returned as bytes
    :param image_path:
    :param bbox:
    :return:
    """
    jpeg = TurboJPEG()
    with open(image_path, 'rb') as _image:
        image = jpeg.decode(_image.read())
        if 'cross_boundary' in bbox:
            bbox_l = bbox['cross_boundary']['left']
            bbox_r = bbox['cross_boundary']['right']
            left = image[int(bbox_l['ymin']):int(bbox_l['ymax']), int(bbox_l['xmin']):int(bbox_l['xmax'])]
            right = image[int(bbox_r['ymin']):int(bbox_r['ymax']), int(bbox_r['xmin']):int(bbox_r['xmax'])]
            image = np.hstack((left, right))
        else:
            image = image[int(bbox['ymin']):int(bbox['ymax']), int(bbox['xmin']):int(bbox['xmax'])]

        image = cv2.resize(image, (32, 32))
        image = image[..., ::-1].copy()  # BGR to RGB
        return image


def process_annotations():
    """
    Loop through all the .txt split files.
    Go fetch the associated .json file and extract the bounding boxes from the .jpeg file
    Return everything as an array
    :return:
    """
    results = []
    labels_to_class = {}
    for split in (['train', 'test']):
        print(f'Started processing split: {split}')
        count = 0
        with open(f'{splits}/{split}.txt') as _split:
            for full_image_key in _split:
                full_image_key = full_image_key.strip()
                image_path = f'{images}.{split}/images/{full_image_key}.jpg'

                try:
                    with open(f'{annotations}/{full_image_key}.json', 'r', encoding='utf-8') as _file:
                        # 1. process json metadata
                        content = json.load(_file)
                        for obj in content['objects']:
                            try:
                                class_idx = labels_to_class[obj['label']]
                            except KeyError:
                                class_idx = len(labels_to_class)
                                labels_to_class[obj['label']] = class_idx

                            results.append({
                                'image': extract_image(image_path, obj['bbox']),
                                'class': class_idx,
                                'label': obj['label'],
                            })
                except FileNotFoundError:
                    pass  # fixme later

                if count % 100 == 0 and count > 0:
                    print(f"Processed: {count} files for split: {split}.")
                count = count + 1

        print(f"Finished processing split: {split}")
    return results


def make_splits(data):
    """
    Make train, test, val splits
    Split sizes are: 80%, 10%, 10%
    :param data:
    :return:
    """
    train_size = int(len(data) * .8)
    rest_size = int((len(data) - train_size) // 2)
    np.random.shuffle(data)

    rest = data[train_size:]  # 20% of data left
    return data[:train_size], rest[:rest_size], rest[rest_size:]


if __name__ == '__main__':
    all_data = process_annotations()
    train, test, val = make_splits(all_data)

    with open("../data/train.pkl", 'wb') as _file:
        pickle.dump(train, _file)

    with open("../data/test.pkl", 'wb') as _file:
        pickle.dump(test, _file)

    with open("../data/val.pkl", 'wb') as _file:
        pickle.dump(val, _file)