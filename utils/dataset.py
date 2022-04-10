import pickle
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class MappilaryDataset(Dataset):
    """
    Custom dataset for Mappilary Data
    """
    def __init__(self, path, transform=None, target_transform=None):
        """

        :param path: path to the .pkl file
        :param transform: transform input features
        :param target_transform: transform labels
        """
        self.images = []
        self.classes = []
        self.labels = []
        with open(path, 'rb') as _file:
            data = pickle.load(_file)
            for d in data:
                self.images.append(d['image'])
                self.classes.append(d['class'])
                self.labels.append(d['label'])

        self.transform = transform
        self.target_transform = target_transform

        self.label_to_class = {}
        self.class_to_label = {}
        for c, l in zip(self.classes, self.labels):
            self.label_to_class[l] = c
            self.class_to_label[c] = l

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.classes[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return to_tensor(image), label
