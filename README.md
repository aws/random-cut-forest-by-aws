# Random Cut Forest

This package is an implementation of the Random Cut Forest probabilistic data 
structure. Random Cut Forests (RCFs) were originally developed at Amazon to use in an
anomaly detection algorithm. After anomaly detection was launched, new 
algorithms based on RCFs were developed for density estimation, imputation,
and forecasting. The goal of this library is to be easy to use and to strike
a balance between efficiency and extensibility.

The public interface to this package is the RandomCutForest class, which 
defines methods for anomaly detection, anomaly detection with attribution,
density estimation, imputation, and forecasting.

## Limitations

* Update operations in a forest are *not thread-safe*. Running concurrent updates or running an update concurrently
  with a traversal may result in errors.

## Basic operations

To create a RandomCutForest instance with all parameters set to defaults:

```java
int dimensions = 1; // The number of dimensions in the input data, required
RandomCutForest forest = RandomCutForest.defaultForest(dimensions);
```

To explicitly set optional parameters like number of trees in the forest or 
sample size, RandomCutForest provides a builder:

```java
RandomCutForest forest = RandomCutForest.builder()
    .numberOfTrees(90)
    .sampleSize(200)
    .dimensions(2) // still required!
    .lambda(0.2)
    .randomSeed(123)
    .storeSequenceIndexesEnabled(true)
    .centerOfMassEnabled(true)
    .build();
```

Typical usage of a forest is to compute a statistic on an input data point and
then update the forest with that point in a loop.

```java
Supplier<double[]> input = ...;

while (true) {
    double[] point = input.get();
    double score = forest.getAnomalyScore(point);
    forest.update(point);
    System.out.println("Anomaly Score: " + score);
}
```

## Forest Configuration

The following parameters can be configured in the RandomCutForest builder. 

| Parameter Name | Type | Description | Default Value|
| --- | --- | --- | --- |
| centerOfMassEnabled | boolean | If true, then tree nodes in the forest will compute their center of mass as part of tree update operations. | false |
| dimensions | int | The number of dimensions in the input data. | Required, no default value |
| lambda | double | The decay factor (lambda value) used by stream samplers in this forest. | 1e-5 |
| numberOfTrees | int | The number of trees in this forest. | 100 |
| outputAfter | int | The number of points required by stream samplers before results are returned. | 0.25 * sampleSize |
| parallelExecutionEnabled | boolean | If true, then the forest will create an internal threadpool. Forest updates and traversals will be submitted to this threadpool, and individual trees will be updated or traversed in parallel. | true |
| randomSeed | long | A seed value used to initialize the random number generators in this forest. | |
| sampleSize | int | The sample size used by stream samplers in this forest | 256 |
| storeSequenceIndexesEnabled | boolean | If true, then sequence indexes (ordinals indicating when a point was added to a tree) will be stored in the forest along with poitn values. | false |
| threadPoolSize | int | The number of threads to use in the internal threadpool. | Number of available processors - 1 |
| windowSize | int | An alternate way of specifying the lambda value. Using this parameter will set lambda to 1 / windowSize. | |

## Setup

1. Checkout this package from our GitHub repository.
1. Install [Apache Maven](https://maven.apache.org/) by following the direcitons on that site.
1. Set your `JAVA_HOME` environment variable to a JDK version 8 or greater.

## Build

Build this package and run the full test suite by running

```text
mvn package
```

For a faster build that excludes that long-running "functional" tests, run

```text
mvn package -DexcludedGroups=functional
```

## Build Command-line (CLI) usage

For some of the algorithms included in this package, there are CLI applications that can
be used for experiments. These applications use `String::split` to read
delimited data, and as such are **not intended for production use**. Instead,
use these applications as example code and as a way to learn about the
algorithms and their hyperparameters.

Build a local archive by running the Maven package command.

```text
% mvn package -DexcludedGroups=functional
```

You can then invoke an example CLI application by adding the resulting jar file to your classpath. For example:

```text
% java -cp target/random-cut-forest-1.0.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner --help
Usage: java -cp RandomCutForest-1.0-super.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner [options] < input_file > output_file

Compute scalar anomaly scores from the input rows and append them to the output rows.

Options:
        --delimiter, -d: The character or string used as a field delimiter. (default: ,)
        --header-row: Set to 'true' if the data contains a header row. (default: false)
        --number-of-trees, -n: Number of trees to use in the forest. (default: 100)
        --random-seed: Random seed to use in the Random Cut Forest (default: 42)
        --sample-size, -s: Number of points to keep in sample for each tree. (default: 256)
        --shingle-cyclic, -c: Set to 'true' to use cyclic shingles instead of linear shingles. (default: false)
        --shingle-size, -g: Shingle size to use. (default: 1)
        --window-size, -w: Window size of the sample or 0 for no window. (default: 0)

        --help, -h: Print this help message and exit.
```

## Code of Conduct

This project has adopted an [Open Source Code of Conduct](https://aws.github.io/code-of-conduct).


## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.


## Licensing

See the [LICENSE](./LICENSE.txt) file for our project's licensing. We will ask you to confirm the licensing of your contribution.


## Copyright

Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
