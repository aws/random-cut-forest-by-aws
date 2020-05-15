# Random Cut Forest

This package is an implementation of the Random Cut Forest probabilistic data structure. Random Cut Forests (RCFs) were 
originally developed at Amazon to use in a nonparametric anomaly detection algorithm for streaming data. Later new 
algorithms based on RCFs were developed for density estimation, imputation, and forecasting. The goal of this library 
is to be easy to use and to strike a balance between efficiency and extensibility.

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

Typical usage of a forest is to compute a statistic on an input data point and then update the forest with that point 
in a loop.

```java
Supplier<double[]> input = ...;

while (true) {
    double[] point = input.get();
    double score = forest.getAnomalyScore(point);
    forest.update(point);
    System.out.println("Anomaly Score: " + score);
}
```

## Limitations

* Update operations in a forest are *not thread-safe*. Running concurrent updates or running an update concurrently
  with a traversal may result in errors.


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

Build the modules in this package and run the full test suite by running

```text
mvn package
```

For a faster build that excludes the long-running "functional" tests, run

```text
mvn package -DexcludedGroups=functional
```

## Build Command-line (CLI) usage

For some of the algorithms included in this package, there are CLI applications that can
be used for experiments. These applications use `String::split` to read
delimited data, and as such are **not intended for production use**. Instead,
use these applications as example code and as a way to learn about the
algorithms and their hyperparameters.

Build a local archive of the core library by running the Maven package command in the randomcutforest-core module.

```text
% cd core
% mvn package -DexcludedGroups=functional
```

You can then invoke an example CLI application by adding the resulting jar file to your classpath. For example:

```text
% java -cp target/randomcutforest-core-1.0.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner --help
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

## Testing

The core library test suite is divided into unit tests and "functional" tests. By "functional", we mean tests that 
verify the expected behavior of the algorithms defined in the package. For example, a functional test for the anomaly 
detection algorithm will first train a forest on a pre-defined distribution and then verify that the forest assigns a 
high anomaly score to anomalous points (where "anomalous" is with respect to the specified distribution). Functional 
tests are indicated both in the test class name (e.g., `RandomCutForestFunctionalTest`) and in a `@Tag` annotation on 
the test class.

The full test suite including functional tests currently takes over 10 minutes to complete. If you are contributing to
this package, we recommend excluding the functional tests while actively developing, and only running the full test
suite before creating a pull request. Functional tests can be excluded from Maven build targets by passing
`-DexcludedGroups=functional` at the command line. For example:

```text
% mvn test -DexcludedGroups=functional
```

We currently have 90% line coverage with the full test suite, and 80% line coverage when running the unit tests only
(i.e., when excluding functional tests). Our goal is to reach 100% unit test coverage, and we welcome (and encourage!)
test contributions. After running tests with Maven, you can see the test coverage broken out by class by opening
`target/site/jacoco/index.html` in a web browser.

Our tests are implemented in [JUnit 5](https://junit.org/junit5/) with [Mockito](https://site.mockito.org/), [Powermock](https://github.com/powermock/powermock), and [Hamcrest](http://hamcrest.org/) for testing. 
Test dependencies will be downloaded automatically when invoking `mvn test` or `mvn package`.

## Benchmarks

The benchmark modules defines microbenchmarks using the [JMH](https://openjdk.java.net/projects/code-tools/jmh/) 
framework. Build an executable jar containing the benchmark code by running

```text
% cd benchmark
% mvn package assembly:single
```

To invoke the full benchmark suite:

```text
% java -jar target/randomcutforest-benchmark-1.0-jar-with-dependencies.jar
```

The full benchmark suite takes a long time to run. You can also pass a regex at the command-line, then only matching
benchmark methods will be executed.

```text
% java -jar target/randomcutforest-benchmark-1.0-jar-with-dependencies.jar RandomCutForestBenchmark\.updateAndGetAnomalyScore
```

## Documentation

* Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016, June). Robust random cut forest based anomaly detection on streams. In *International conference on machine learning* (pp. 2712-2721).

## Code of Conduct

This project has adopted an [Open Source Code of Conduct](https://aws.github.io/code-of-conduct).


## Security issue notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public GitHub issue.


## Licensing

See the [LICENSE](./LICENSE.txt) file for our project's licensing. We will ask you to confirm the licensing of your contribution.


## Copyright

Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
