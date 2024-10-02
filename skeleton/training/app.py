from common.transformations import preprocess
from common.model_factory import new_model, save_model
from numpy.random import seed
from sklearn.model_selection import train_test_split
from training.load_data import load_data


def main():
    data = load_data()

    seed(1)  # Put a seed in here for repeatability.
    target_column = 'target'
    x_train, x_test, y_train, y_test = train_test_split(data.drop([target_column], axis=1), data[target_column])
    
    preprocessed_data = preprocess(x_train)
    
    model = new_model()
    model.fit(preprocessed_data, y_train, random_state=1)  # Put a seed in here for repeatability.
    #  Note that some models may not accept this parameter.

    # Add your evaluation metric here if you need to immediately see how the model performed on the test set.
    save_model(model)


if __name__ == "__main__":
    main()
