# Random Cut Forest

This directory contains a Java implementation of the Random Cut Forest data structure and algorithms
for anomaly detection, density estimation, imputation, and forecast. The goal of this library 
is to be easy to use and to strike a balance between efficiency and extensibility. Please do not forget 
to look into the ParkServices package that provide many augmented functionalities such as explicit determination 
of anomaly grade based on the first hand understanding of the core algorithm. Please also see randomcutforest-examples 
for a few detailed examples and extensions. Please do not hesitate to creat an issue for any discussion item.

## Basic operations

To create a RandomCutForest instance with all parameters set to defaults:

```java
int dimensions = 5; // The number of dimensions in the input data, required
RandomCutForest forest = RandomCutForest.defaultForest(dimensions);
```
We recommend using shingle size which correspond to contextual analysis of data, 
and RCF uses ideas not dissimilar from higher order Markov Chains to improve its 
accuracy. An option is provided to have the shingles be constructed internally. 
To explicitly set optional parameters like number of trees in the forest or 
sample size, RandomCutForest provides a builder (for example with 4 input dimensions for 
a 4-way multivariate analysis):

```java
RandomCutForest forest = RandomCutForest.builder()
        .numberOfTrees(90)
        .sampleSize(200) // use this cover the phenomenon of interest
                         // for analysis of 5 minute aggregations, a week has
                         // about 12 * 24 * 7 starting points of interest
                         // larger sample sizes will be larger models 
        .dimensions(inputDimension*4) // still required!
        .timeDecay(0.2) // determines half life of data
        .randomSeed(123)
        .internalShingleEnabled(true)
        .shingleSize(7)
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

| Parameter Name              | Type    | Description                                                                                                                                                                                                                                                                                                                                                    | Default Value                                                                         |
|-----------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| dimensions                  | int     | The number of dimensions in the input data.                                                                                                                                                                                                                                                                                                                    | Required, no default value. Should be the product of input dimensions and shingleSize |
| shingleSize                 | int     | The number of contiguous observations across all the input variables that would be used for analysis                                                                                                                                                                                                                                                           | Strongly recommended for contextual anomalies. Required for Forecast/Extrapolate      |
| lambda                      | double  | The decay factor used by stream samplers in this forest. See the next section for guidance.                                                                                                                                                                                                                                                                    | 1 / (10 * sampleSize)                                                                 |
| numberOfTrees               | int     | The number of trees in this forest.                                                                                                                                                                                                                                                                                                                            | 50                                                                                    |
| outputAfter                 | int     | The number of points required by stream samplers before results are returned.                                                                                                                                                                                                                                                                                  | 0.25 * sampleSize                                                                     |
| internalShinglingEnabled    | boolean | Whether the shingling is performed by RCF itself since it has already seen previous values.                                                                                                                                                                                                                                                                    | false (for historical reasons). Recommended : true, will result in smaller models.    |
| parallelExecutionEnabled    | boolean | If true, then the forest will create an internal threadpool. Forest updates and traversals will be submitted to this threadpool, and individual trees will be updated or traversed in parallel. For larger shingle sizes, dimensions, and number of trees, parallelization may improve throughput. We recommend users benchmark against their target use case. | false                                                                                 |
| randomSeed                  | long    | A seed value used to initialize the random number generators in this forest.                                                                                                                                                                                                                                                                                   |                                                                                       |
| sampleSize                  | int     | The sample size used by stream samplers in this forest                                                                                                                                                                                                                                                                                                         | 256                                                                                   |
| centerOfMassEnabled         | boolean | If true, then tree nodes in the forest will compute their center of mass as part of tree update operations.                                                                                                                                                                                                                                                    | false                                                                                 |
| storeSequenceIndexesEnabled | boolean | If true, then sequence indexes (ordinals indicating when a point was added to a tree) will be stored in the forest along with poitn values.                                                                                                                                                                                                                    | false                                                                                 |
| threadPoolSize              | int     | The number of threads to use in the internal threadpool.                                                                                                                                                                                                                                                                                                       | Number of available processors - 1                                                    |

The above parameters are the most common and historical. Please use the issues to request additions/discussions of other parameters of interest.

RandomCutForest primarily provides an estimation (say anomaly score, or extrapolation over a forecast horizon) and using that raw estimation can be challenging. The ParkServices package provides 
several capabilities (ThresholdedRandomCutForest, RCFCaster, respectively) for distilling the scores to a determination of 
anomaly/otherwise (an assesment of grade) or calibrated conformal forecasts. These have natural parameter choices that are different 
from the core RandomCutForest -- for example internalShinglingEnabled defaults to true since that is more natural in those contexts.
The package examples provides a collection of examples and uses of parameters, we draw the attention to ThresholdedMultiDimensionalExample 
and RCFCasterExample. If one is interested in sequential analysis of a series of consecutive inputs, check out SequentialAnomalyExample. 
ParkServices also exposes many other functionalities of RCF which were covert, such as clustering (including multi-centroid representations) 
-- see NumericGLADExample for instance. 

## Choosing a `timeDecay` value for your application

When we submit a point to the sampler, it is included into the sample with some probability, and 
it will remain in the for some number of steps before being replaced. Call the number of steps that
a point is included in the sample the "lifetime" of the point (which may be 0). Over a finite time
window, the distribution of the lifetime of a point is approximately exponential with parameter
`lambda`. Thus, `1 / timmeDecay` is approximately the average number of steps that a point will be included
in the sample. By default, we set `timeDecay` equal to `1 / (10 * sampleSize)`.

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

> **Important.** The CLI applications use `String::split` to read delimited data
> and as such are **not intended for production use**.

For some of the algorithms included in this package there are CLI applications
that can be used for experimentation as well as a way to learn about these
algorithms and their hyperparameters. After building the project you can invoke
an example CLI application by adding the core jar file to your classpath.

In the example below we train and score a Random Cut Forest model on the
three-dimensional data shown in Figure 3 in the original RCF paper.
([PDF][rcf-paper]) These example data can be
found at `../example-data/rcf-paper.csv`:

```text
$ tail data/example.csv
-5.0074,-0.0038,-0.0237
-5.0029,0.0170,-0.0057
-4.9975,-0.0102,-0.0065
4.9878,0.0136,-0.0087
5.0118,0.0098,-0.0057
0.0158,0.0061,0.0091
5.0167,0.0041,0.0054
-4.9947,0.0126,-0.0010
-5.0209,0.0004,-0.0033
4.9923,-0.0142,0.0030
```

(Note that there is one data point above that is not like the others.) The
`AnomalyScoreRunner` application reads in each line of the input data as a
vector data point, scores the data point, and then updates the model with this
point. The program output appends a column of anomaly scores to the input:

```text
$ java -cp core/target/randomcutforest-core-3.5.0.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner < ../example-data/rcf-paper.csv > example_output.csv
$ tail example_output.csv
-5.0029,0.0170,-0.0057,0.8129401629464965
-4.9975,-0.0102,-0.0065,0.6591046054520615
4.9878,0.0136,-0.0087,0.8552217070518414
5.0118,0.0098,-0.0057,0.7224686064066762
0.0158,0.0061,0.0091,2.8299054033889814
5.0167,0.0041,0.0054,0.7571453322237215
-4.9947,0.0126,-0.0010,0.7259960347128676
-5.0209,0.0004,-0.0033,0.9119498264685114
4.9923,-0.0142,0.0030,0.7310102658466711
Done.
```

(As you can see the anomalous data point was given large anomaly score.) You can
read additional usage instructions, including options for setting model
hyperparameters, using the `--help` flag:

```text
$ java -cp core/target/randomcutforest-core-3.5.0.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner --help
Usage: java -cp target/random-cut-forest-3.5.0.jar com.amazon.randomcutforest.runner.AnomalyScoreRunner [options] < input_file > output_file

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

Other CLI applications are available in the `com.amazon.randomcutforest.runner`
package.

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
% java -jar benchmark/target/randomcutforest-benchmark-3.5.0-jar-with-dependencies.jar
```

The full benchmark suite takes a long time to run. You can also pass a regex at the command-line, then only matching
benchmark methods will be executed.

```text
% java -jar benchmark/target/randomcutforest-benchmark-3.5.0-jar-with-dependencies.jar RandomCutForestBenchmark\.updateAndGetAnomalyScore
```

[rcf-paper]: http://proceedings.mlr.press/v48/guha16.pdf
