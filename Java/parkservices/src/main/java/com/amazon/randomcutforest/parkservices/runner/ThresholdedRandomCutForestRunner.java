/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest.parkservices.runner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.function.Function;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.runner.LineTransformer;
import com.amazon.randomcutforest.runner.SimpleRunner;
import com.amazon.randomcutforest.util.ShingleBuilder;

/**
 * A simple command line application for performing thresholded anomaly
 * detection on multi-dimensional data. Creates an instance of a
 * ThresholdedRandomCutForest, reads values from STDIN, and writes results to
 * STDOUT.
 */
public class ThresholdedRandomCutForestRunner extends SimpleRunner {

    // overwrite the base class initializer to use a ThresholdedRandomCutForest
    protected final Function<ThresholdedRandomCutForest, LineTransformer> thresholdedAlgorithmInitializer;
    protected final ThresholdedArgumentParser thresholdedArgumentParser;

    public ThresholdedRandomCutForestRunner() {
        // modify the default argument parser to accept additional,
        // thresholding-specific arguments
        // before instantiating the runner
        super(new ThresholdedArgumentParser(ThresholdedRandomCutForest.class.getName(),
                "Streaming anomaly detection on input rows. Appends anomaly score and anomaly grade to output rows."),
                null);

        thresholdedAlgorithmInitializer = AnomalyDescriptorTransformer::new;
        thresholdedArgumentParser = (ThresholdedArgumentParser) argumentParser;
    }

    public static void main(String... args) throws IOException {
        ThresholdedRandomCutForestRunner runner = new ThresholdedRandomCutForestRunner();
        runner.parse(args);
        runner.run(new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
                new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8)));
    }

    /**
     * Set up the internal ThresholdedRandomCutForest instance and line transformer.
     * 
     * The SimpleRunner class assumes that the input algorithm is a RandomCutForest.
     * This method needs to be overwritten so that we initialize a
     * ThresholdedRandomCutForest, instead.
     * 
     * @param dimensions The number of dimensions in the input data.
     */
    @Override
    protected void prepareAlgorithm(int dimensions) {
        pointBuffer = new double[dimensions];
        shingleBuilder = new ShingleBuilder(dimensions, argumentParser.getShingleSize(),
                argumentParser.getShingleCyclic());
        shingleBuffer = new double[shingleBuilder.getShingledPointSize()];

        // need to set internal shingling to false for SimpleRunner compatibility
        ThresholdedRandomCutForest model = ThresholdedRandomCutForest.builder()
                .numberOfTrees(argumentParser.getNumberOfTrees()).sampleSize(argumentParser.getSampleSize())
                .dimensions(dimensions).internalShinglingEnabled(false).shingleSize(argumentParser.getShingleSize())
                .timeDecay(argumentParser.getTimeDecay()).randomSeed(argumentParser.getRandomSeed())
                .anomalyRate(thresholdedArgumentParser.getAnomalyRate()).build();

        model.getThresholder().setHorizon(thresholdedArgumentParser.getHorizon());
        model.getThresholder().setLowerThreshold(thresholdedArgumentParser.getLowerThreshold());
        model.getThresholder().setZfactor(thresholdedArgumentParser.getZfactor());

        algorithm = thresholdedAlgorithmInitializer.apply(model);
    }

    public static class AnomalyDescriptorTransformer implements LineTransformer {
        private final ThresholdedRandomCutForest model;
        private long timestamp;

        public AnomalyDescriptorTransformer(ThresholdedRandomCutForest model) {
            this.model = model;
            this.timestamp = 0;
        }

        @Override
        public List<String> getResultValues(double... point) {
            // assumes that every input point is consecutive
            AnomalyDescriptor result = model.process(point, timestamp);
            String score = Double.toString(result.getRcfScore());
            String grade = Double.toString(result.getAnomalyGrade());
            timestamp += 1;
            return List.of(score, grade);
        }

        @Override
        public List<String> getEmptyResultValue() {
            return List.of("NA", "NA");
        }

        @Override
        public List<String> getResultColumnNames() {
            return List.of("anomaly score", "anomaly grade");
        }

        @Override
        public RandomCutForest getForest() {
            return model.getForest();
        }
    }
}
