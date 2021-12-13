# Model-guided mutation (MGM)

The following subpackages are included in this repository:

1. `algorithms`: Algorithms for model-guided mutation
2. `common`: Subpackage for common classes and functions used throughout the package, such as the `Sequence` class
3. `data`: Subpackage for data generation and pre-preprocessing
4. `models`: Subpackage for relevant classification models
5. `pipelines`: Implementations of various experimental pipelines
6. `analysis`: History (Variant and VariantSet) and analysis utilities
7. `testing`: Testing utilities for the `mgm` package


Notes to self - next steps:

* Finish exp 1
* Should Kuzmin data load return aa_vocab?
* Randomized MGM - something like greedy_mgm will produce one Variant record; need a wrapper to call multiple times; create a new object in `history.py` called `VariantSet`which will store a list of Variants; pull fields that are the same across Variants into fields in the VariantSet