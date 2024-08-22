# AI Golden Path Template

The objective of this project is to provide a simple template to get going on an AI system that enables proper model development through to production and back to development. The different modules provide ways to keep your code synchronized to give your model the best chance to integrate with your software environment.

Please note that any data orchestration must maintain synchronization between serving and training environments, and that is specific to your organization.

As is always the case for AI models, the only way to use them is if they can be integrated into the rest of your organization. While this template eases that, it is still the user's responsibility to ensure that the model system's API maintain synchronization between training and serving, and is feasible for a calling application.

## Requirements

1) Red Hat OpenShift AI installed on your OpenShift cluster
2) A StorageProvider enabled that supports the RWX access mode:

    * NFS
    * [AWS EFS](https://docs.openshift.com/container-platform/4.16/storage/container_storage_interface/persistent-storage-csi-aws-efs.html)
    * [Azure File](https://docs.openshift.com/container-platform/4.16/storage/container_storage_interface/persistent-storage-csi-azure-file.html)
    * [Google Compute Platform Filestore](https://docs.openshift.com/container-platform/4.16/storage/container_storage_interface/persistent-storage-csi-google-cloud-file.html)
    * ODF CephFS

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
