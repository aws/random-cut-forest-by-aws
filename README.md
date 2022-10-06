# Random Cut Forest by AWS

This repository contains implementations of the Random Cut Forest (RCF) probabilistic data structure.
RCFs were originally developed at Amazon to use in a nonparametric anomaly detection algorithm for
streaming data. Later new algorithms based on RCFs were developed for density estimation, imputation,
and forecasting.

The different directories correspond to equivalent implementations in different languages, and bindings to
to those base implementations, using language specific features for greater flexibility of use. 

The package randomcutforest-examples showcases several example scenarios for using the repository. 
They also provide examples for some of the parameter settings. Many of these examples are built in tests.

## Documentation

* Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016, June). Robust random cut forest based anomaly detection on streams. In *International conference on machine learning* (pp. 2712-2721).

## Code of Conduct

This project has adopted an [Open Source Code of Conduct](https://aws.github.io/code-of-conduct).


## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.


## Licensing

See the [LICENSE](./LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.


## Copyright

Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
