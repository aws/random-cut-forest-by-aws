# Random Cut Forest

This directory contains a Java implementation of the Random Cut Forest data structure and algorithms
for anomaly detection, density estimation, imputation, and forecast. The goal of this library 
is to be easy to use and to strike a balance between efficiency and extensibility.

## Basic operations

To create a RandomCutForest instance with all parameters set to defaults:

```java
int dimensions = 5; // The number of dimensions in the input data, required
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
| lambda | double | The decay factor used by stream samplers in this forest. See the next section for guidance. | 1 / (10 * sampleSize) |
| numberOfTrees | int | The number of trees in this forest. | 50 |
| outputAfter | int | The number of points required by stream samplers before results are returned. | 0.25 * sampleSize |
| parallelExecutionEnabled | boolean | If true, then the forest will create an internal threadpool. Forest updates and traversals will be submitted to this threadpool, and individual trees will be updated or traversed in parallel. For larger shingle sizes, dimensions, and number of trees, parallelization may improve throughput. We recommend users benchmark against their target use case. | false |
| randomSeed | long | A seed value used to initialize the random number generators in this forest. | |
| sampleSize | int | The sample size used by stream samplers in this forest | 256 |
| storeSequenceIndexesEnabled | boolean | If true, then sequence indexes (ordinals indicating when a point was added to a tree) will be stored in the forest along with poitn values. | false |
| threadPoolSize | int | The number of threads to use in the internal threadpool. | Number of available processors - 1 |

## Choosing a `lambda` value for your application

When we submit a point to the sampler, it is included into the sample with some probability, and 
it will remain in the for some number of steps before being replaced. Call the number of steps that
a point is included in the sample the "lifetime" of the point (which may be 0). Over a finite time
window, the distribution of the lifetime of a point is approximately exponential with parameter
`lambda`. Thus, `1 / lambda` is approximately the average number of steps that a point will be included
in the sample. By default, we set `lambda` equal to `1 / (10 * sampleSize)`.

Alternatively, if you want the probability that a point survives longer than n steps to be 0.05,
you can solve for `lambda` in the equation `exp(-lambda * n) = 0.05`.

We note again that this is heuristic and not mathematically rigorous. We refer the interested reader
to [Weighted Random Sampling (2005;  Efraimidis, Spirakis)](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=BEB1FE0AB3C0129B822D2CE5EABBFD42?doi=10.1.1.591.4194&rep=rep1&type=pdf).

## Setup

1. Checkout this package from our GitHub repository.
1. Install [Apache Maven](https://maven.apache.org/) by following the directions on that site.
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

After building the project (described in the previous section), you can invoke an example CLI application by adding the
core jar file to your classpath. For example:

```text
% java -cp core/target/randomcutforest-core-1.0-alpha.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner --help
Usage: java -cp randomcutforest-core-1.0-alpha.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner [options] < input_file > output_file

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

In the core library we have 90% line coverage with the full test suite, and 80% line coverage when running the unit 
tests only (i.e., when excluding functional tests). Our goal is to reach 100% unit test coverage, and we welcome (and 
encourage!) test contributions. After running tests with Maven, you can see the test coverage broken out by class by 
opening `target/site/jacoco/index.html` in a web browser.

Our tests are implemented in [JUnit 5](https://junit.org/junit5/) with [Mockito](https://site.mockito.org/), [Powermock](https://github.com/powermock/powermock), and [Hamcrest](http://hamcrest.org/) for testing. 
Test dependencies will be downloaded automatically when invoking `mvn test` or `mvn package`.

## Benchmarks

The benchmark modules defines microbenchmarks using the [JMH](https://openjdk.java.net/projects/code-tools/jmh/) 
framework. Build an executable jar containing the benchmark code by running

```text
% # (Optional) To benchmark the code in your local repository, build and install to your local Maven repository
% # Otherwise, benchmark dependencies will be pulled from Maven central
% mvn package install -DexcludedGroups=functional
% 
% mvn -pl benchmark package assembly:single
```

To invoke the full benchmark suite:

```text
% java -jar benchmark/target/randomcutforest-benchmark-1.0-jar-with-dependencies.jar
```

The full benchmark suite takes a long time to run. You can also pass a regex at the command-line, then only matching
benchmark methods will be executed.

```text
% java -jar benchmark/target/randomcutforest-benchmark-1.0-jar-with-dependencies.jar RandomCutForestBenchmark\.updateAndGetAnomalyScore
```
