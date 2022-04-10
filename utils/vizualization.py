import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.dataset import MappilaryDataset


def log_random_images(title, writer, dataset, size=16, epoch=0):
    """
    Log some training example images to a tensorboard writer for log purposes
    :param title:
    :param writer:
    :param dataset:
    :param size:
    :param epoch:
    :return:
    """
    random_indexes = np.random.choice(len(dataset), size=size, replace=False)
    images = np.zeros((size, 3, 32, 32))
    for i, idx in enumerate(random_indexes):
        img, tag = dataset[idx]
        images[i] = img

    writer.add_images(title, images, epoch)


def log_class_distribution(title, writer, dataset):
    bins = np.array(dataset.classes)
    writer.add_histogram(title, bins)


def show_predictions(model, dataset, correctly_predicted=True):
    """
    Display the top BEST and Worst predictions by accuracy
    :param correctly_predicted:
    :param dataset: MappilaryDataset
    :return:
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    n = 6

    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)

        probs = nnf.softmax(outputs, dim=1)  # compute the in terms of probabilities (0, 1.0)
        prob, y_pred = torch.max(probs, 1)   # get the (prob, class) pair for every probability
        pred_idx = (y_pred == labels) if correctly_predicted else (y_pred != labels)

        _sorted = np.argsort(prob[pred_idx].detach().numpy())
        best_images = inputs[pred_idx][_sorted][-n:]
        best_classes_pred = y_pred[pred_idx][_sorted][-n:]
        best_classes_true = labels[pred_idx][_sorted][-n:]
        best_accuracy = prob[pred_idx][_sorted][-n:].detach().numpy()

        worst_images = inputs[pred_idx][_sorted][:n]
        worst_classes_pred = y_pred[pred_idx][_sorted][:n]
        worst_classes_true = labels[pred_idx][_sorted][:n]
        worst_accuracy = prob[pred_idx][_sorted][:n].detach().numpy()

        title_color = "green" if correctly_predicted else "red"
        plt.figure()
        for i in range(6):
            image = best_images[i].permute(1, 2, 0)
            pred_class_label = dataset.class_to_label[best_classes_pred[i].item()]
            true_class_label = dataset.class_to_label[best_classes_true[i].item()]

            plt.subplot(2, 3, i + 1)
            plt.imshow(image)
            plt.title(f"{best_accuracy[i]:.2%}", color=title_color)
            plt.text(0.5, 2, true_class_label, color="green").set_bbox(dict(facecolor='black', alpha=0.8))
            if not correctly_predicted:
                plt.text(0.5, 7, pred_class_label, color="red").set_bbox(dict(facecolor='black', alpha=0.8))

        plt.figure()
        for i in range(6):
            image = worst_images[i].permute(1, 2, 0)
            pred_class_label = dataset.class_to_label[worst_classes_pred[i].item()]
            true_class_label = dataset.class_to_label[worst_classes_true[i].item()]

            plt.subplot(2, 3, i + 1)
            plt.imshow(image)
            plt.title(f"{worst_accuracy[i]:.2%}", color=title_color)
            plt.text(0.5, 2, true_class_label, color="green").set_bbox(dict(facecolor='black', alpha=0.8))
            if not correctly_predicted:
                plt.text(0.5, 7, pred_class_label, color="red").set_bbox(dict(facecolor='black', alpha=0.8))

        plt.show()
        return


def show_weights_for_class(model, dataset, label):
    """
    Show the activation function weights for training example of a given label
    :param model: model.Model
    :param dataset:
    :param label:
    :return:
    """
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    class_idx = dataset.label_to_class[label]

    # find the image with a corresponding label
    inputs, labels = next(iter(dataloader))
    images = inputs[np.where(labels == class_idx)]
    image = images[np.random.randint(len(images))]

    # run through the first conv layer of our model
    output = model.conv1(image.reshape(1, 3, 32, 32))
    x = output.reshape(6, 1, 28, 28)

    # plot
    plt.figure()
    plt.subplot(121)
    plt.imshow(image.permute(1, 2, 0))

    plt.subplot(122)
    plt.title(label)
    plt.imshow(make_grid(x.cpu(), nrow=3, normalize=True, scale_each=True).permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    test_dataset = MappilaryDataset('../data/test.pkl')
    model = torch.jit.load('../models/latest_model.pt')
    model.eval()

    # show_predictions(model, test_dataset, correctly_predicted=True)
    # show_predictions(model, test_dataset, correctly_predicted=False)
    show_weights_for_class(model, test_dataset, label="regulatory--maximum-speed-limit-100--g1")
