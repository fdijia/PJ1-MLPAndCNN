### Start Up

After downloading `pj1_savedData`, please drag all the file into the root directory.

### Attentions

- If the `beta` in `MomentGD` is `0`, then it is `SGD`
- If you define your layers by the pattern in .`hyperparameter_search.py`, you should use `Model_CNN`, otherwise you can just use `Model_MLP`.
- In training script, you can choose two training set, one is MNIST and another is argumented set from MNIST.

### Train the model.

Open `hyperparameter_search.py`, modify parameters and run `test_train_cnn.py` after apply the parameters you've changed.

You can choose two dataset to train the model.

### Test the model.

Open `test_model.py`, specify the saved model's path and the test dataset's path, then run the script, the script will output the accuracy on the test dataset.



