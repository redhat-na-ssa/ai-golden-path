from common.transformations import preprocess
from common.model_factory import new_model, save_model
from training.load_data import load_data


def main():
    data = load_data()

    preprocessed_data = preprocess(data)
    
    model = new_model()
    model.fit(preprocessed_data, data)  # The second argument needs to be the target variable.
    save_model(model)


if __name__ == "__main__":
    main()
