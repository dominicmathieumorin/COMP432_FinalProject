# Traffic Sign Recognition
## Convolutional Neural Networks & PyTorch

Dominic Morin (40032939)

[Link to the overleaf project](https://www.overleaf.com/project/6251a2bc02c8bb986bf9baae)


### 1. Run the training
```bash
# run tensorboard so you can plot data in real time
mkdir runs
tensorboard --logdir runs

# run the training loop
python train.py
```

### 2. Plot some data

``` bash
# uncomment onf of the following lines in vizualization.py:
## show_predictions(model, test_dataset, correctly_predicted=True)
## show_predictions(model, test_dataset, correctly_predicted=False)
## show_weights_for_class(model, test_dataset, label="regulatory--maximum-speed-limit-100--g1")

# run the vizualization script
python vizualization.py
```