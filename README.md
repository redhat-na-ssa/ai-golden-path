# AI Golden Path Template

The objective of this project is to provide a simple template to get going on an AI system that enables proper model development through to production and back to development. The different modules provide ways to keep your code synchronized to give your model the best chance to integrate with your software environment.

Please note that any data orchestration must maintain synchronization between serving and training environments, and that is specific to your organization.

As is always the case for AI models, the only way to use them is if they can be integrated into the rest of your organization. While this template eases that, it is still the user's responsibility to ensure that the model system's API maintain synchronization between training and serving, and is feasible for a calling application.

## Features

This project contains all of the features you would need to integrate with infrastructure to produce an enterprise-ready AI system.

1) Development
    1) A templated Python project (see System Modules below) to enable a fully featured AI system
    2) Jupyter for notebooks in which you can run experiments.
        1) The production code is cloned and added to the PATH so you can iterate on new model versions while taking advantage of your existing work.
    3) Dev Spaces (VSCode) enabled so that you can develop and test your production code
2) CI/CD
    2) Pipelines (Tekton) for continuous integration and continuous deployment
        1) Unit tests enabled in the build images and tested during the build process
    1) ImageStreams for continuous delivery
3) Deployables
    1) Dev environment mirrored to production, using the latest tags rather than production tags.
    2) Production environment, fully deployed on Git tagging.
        1) Automated model training job scheduled with Data Science Pipelines
        2) Automated evaluation job to compare models, make decisions about which models should be serving, and monitor and alert on model performance, all scheduled with Data Science Pipelines
        3) Tightly coupled model service wrapper to handle orchestration and pre-/post-processing.
            1) Model service wrapper enabled with proper operational concerns (multiple instances, healthchecks, etc.)
4) OpenShift networking to expose your model system outside your OpenShift cluster
5) Enabled to work with models loaded into memory from a model registry or an external model serving service


## Cluster Requirements

1) Red Hat OpenShift AI
2) Red Hat Developer Hub
3) Janus ArgoCD

## System Modules

### Training

Code required for training that won't get deployed to the serving environment.

### Serving

Code required for serving that won't get deployed to the training environment.

As your orchestration gets more complicated, I would recommend pulling out the logic into its own function in a different script to keep your API clean. This can commonly occur as you start needing to pull data from e.g. a feature store.

### Common

Any code that is required for both training and serving must go in here. This typically includes your full pipeline: preprocessing, the model classes themselves, and postprocessing.

Your model registry bridges between environments and therefore must be accessible to both the training and serving environments. Alternatively, you may deploy your model in a dedicated model deployment system - this project abstracts away the interface, so you will still be able to execute the full pipeline and keep that system in sync with this one.

## Model Version

Your model version is only lightly coupled with the software version. However, specifically note that there is no way to increment the model version without changing the code version, as model version changes require code changes. As such, the only way to manage it is through the software itself as a variable. This goes in the common.__init__ file.
