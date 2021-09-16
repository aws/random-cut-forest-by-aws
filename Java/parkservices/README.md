# Park Services

Park services includes tools for perfoming thresholded anomaly detection. The
random cut forest model outputs an anomaly score, such that the larger the score
the more anomalous the data point. By contrast, a thresholded random cut forest
classifies data points as anomalous or not. It also outputs detailed information
about an anomaly such as its severity.

## Command-line (CLI) Usage

> **Important.** The CLI applications use `String::split` to read delimited data
> and as such are **not intended for production use**.

For ease of use, begin by building park services with dependencies included. Otherwise,
you will have to explicitly link to the other packages in this library.

```
$ mvn clean
$ mvn -pl parkservices package assembly:single
```

Then, run the thresholded random cut forest runner on input data. The data is
provided via STDIN similar to the usage of `AnomalyScoreRunner` and other runners
in the `core` package.

```
$ java -cp parkservices/target/randomcutforest-parkservices-2.0.1-jar-with-dependencies.jar com.amazon.randomcutforest.parkservices.runner.ThresholdedRandomCutForestRunner < ../example-data/rcf-paper.csv > ./example_output.csv
$ tail example_output.csv
-5.0074,-0.0038,-0.0237,1.05251420280967,0.0
-5.0029,0.0170,-0.0057,0.779527152203384,0.0
-4.9975,-0.0102,-0.0065,0.6555765135757349,0.0
4.9878,0.0136,-0.0087,0.7926342873023955,0.0
5.0118,0.0098,-0.0057,0.7398851897159503,0.0
0.0158,0.0061,0.0091,2.757692377161313,1.0
5.0167,0.0041,0.0054,0.7490870040523756,0.0
-4.9947,0.0126,-0.0010,0.6807435706579732,0.0
-5.0209,0.0004,-0.0033,0.8978880069845551,0.0
4.9923,-0.0142,0.0030,0.7360101409592287,0.0

```

The thresholded random cut forest appends two columns to the original data. The
first column is the anomaly score determined by the random cut forest model. The
second column is the anomaly grade: a non-zero grade indicates that the thresholder
detected an anomaly.

Note that there is one data point that is not like the others. The large anomaly
score and non-zero anomaly grade indicate this! You can read additional usage
instructions, including options for setting model hyperparameters, using the
`--help` flag. There are several thresholding-specific hyperparameters that can
be set in addition to the usual random cut forest hyperparameters.

```
$ java -cp parkservices/target/randomcutforest-parkservices-2.0.1-jar-with-dependencies.jar com.amazon.randomcutforest.parkservices.runner.ThresholdedRandomCutForestRunner --help
Usage: java -cp randomcutforest-core-1.0.jar com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForest [options] < input_file > output_file

Streaming anomaly detection on input rows. Appends anomaly score and anomaly grade to output rows.

Options:
        --anomaly-rate: Approximate expected anomaly rate. Controls anomaly threshold decay rate. (default: 0.01)
        --delimiter, -d: The character or string used as a field delimiter. (default: ,)
        --header-row: Set to 'true' if the data contains a header row. (default: false)
        --horizon: Mixture factor between using scores and score differences for thresholding. Value of 1.0 means the thresholder only uses score values. Value of 0.0 means the thresholder only uses score differences. (default: 0.5)
        --lower-threshold: Anomaly score threshold for marking a potential anomaly. Affects thresholder sensitivity. (default: 1.0)
        --number-of-trees, -n: Number of trees to use in the forest. (default: 100)
        --random-seed: Random seed to use in the Random Cut Forest (default: 42)
        --sample-size, -s: Number of points to keep in sample for each tree. (default: 256)
        --shingle-cyclic, -c: Set to 'true' to use cyclic shingles instead of linear shingles. (default: false)
        --shingle-size, -g: Shingle size to use. (default: 1)
        --window-size, -w: Window size of the sample or 0 for no window. (default: 0)
        --zfactor: Z-score threshold for marking a potential anomaly. Affects thresholder sensitivity. (default: 2.5)

        --help, -h: Print this help message and exit.
```