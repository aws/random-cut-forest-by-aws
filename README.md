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

## Command-line usage

For each algorithm included in this package there is CLI application that can
be used for experiments. These applications use `String::split` to read
delimited data, and as such are **not intended for production use**. Instead,
use these applications as example code and as a way to learn about the
algorithms and their hyperparameters.

You can build a local archive by running the Maven package command. Use the "excludedGroups" flag to disable the
long-running "functional" tests, which take about 10 minutes to complete.

```text
% mvn package -DexcludedGroups=functional
```

You can then invoke an example CLI application by adding the superjar to your classpath. For example:

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

