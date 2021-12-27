# Model-guided mutation (MGM)

The following subpackages are included in this repository:

1. `algorithms`: Algorithms for model-guided mutation
2. `common`: Subpackage for common classes and functions used throughout the package, such as the `Sequence` class
3. `data`: Subpackage for data generation and pre-preprocessing
4. `models`: Subpackage for relevant classification models
5. `pipelines`: Implementations of various experimental pipelines
6. `analysis`: History (Variant and VariantList) and analysis utilities
7. `testing`: Testing utilities for the `mgm` package


Notes to self - next steps:

* mgm.data.kuzmin_data.species_aware_CV() can be improved and made more generic
* run exp3