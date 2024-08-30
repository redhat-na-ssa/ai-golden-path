# {{ cookiecutter.project_name }}

{{ cookiecutter.project_description }}

## Initial Setup - to be removed after implementation

Once your code is generated, you need to make several additional steps to make this functional.

1) training.app is where the model is being fit. Your fit function may look different from the generated, and in the case of a supervised model, you will need to provide a proper target.
2) common.model_factory is where a new model gets generated. That needs to go here. Note that this project will still work if you're using a custom model as long as it is defined somewhere in common.
3) Fill in the details! (I.e. build your model pipeline)
    1) training.load_data is where you load your data
    2) common.transformations is where your pre- and post-processing go
        * Please note that these functions are used for serving, training, and evaluation, so the inputs need to match across all three of them. The implementations for these functions just take generic DataFrames with no type checks on them in the interest of usability. If you are interested in using DataFrames but want the benefit of proper type-checking on them, I would recommend that you use [pandera](https://pandera.readthedocs.io/en/stable/).
4) Update the classes serving.contract with your serving API