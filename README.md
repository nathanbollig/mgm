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

* Currently mgm-d flip flops. The initial flip is the same as hotflip, 
    but then the second flip can best minimize the objective by reverting to the original
    sequence. Essentially this feels like allowing the algorithm to pretend that the 
    original mutation never happened. This behavior can probably be avoided by
    increasing the value of lambda, but also it is unclear whether this behavior
    is acceptable.