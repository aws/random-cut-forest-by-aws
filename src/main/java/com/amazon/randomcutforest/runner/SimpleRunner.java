/*
 * Copyright <2019> Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.runner;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;
import java.util.function.Function;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.util.ShingleBuilder;

public class SimpleRunner {

    protected final ArgumentParser argumentParser;
    protected final Function<RandomCutForest, LineTransformer> algorithmInitializer;
    protected LineTransformer algorithm;
    protected ShingleBuilder shingleBuilder;
    protected double[] pointBuffer;
    protected double[] shingleBuffer;
    protected int lineNumber;

    public SimpleRunner(String runnerClass, String runnerDescription,
                        Function<RandomCutForest, LineTransformer> algorithmInitializer) {
        this(new ArgumentParser(runnerClass, runnerDescription), algorithmInitializer);
    }

    public SimpleRunner(ArgumentParser argumentParser, Function<RandomCutForest, LineTransformer> algorithmInitializer) {
        this.argumentParser = argumentParser;
        this.algorithmInitializer = algorithmInitializer;
    }

    public void parse(String... arguments) {
        argumentParser.parse(arguments);
    }

    public void run(BufferedReader in, PrintWriter out) throws IOException {
        String line;
        while ((line = in.readLine()) != null) {
            lineNumber++;
            String[] values = line.split(argumentParser.getDelimiter());

            if (pointBuffer == null) {
                prepareAlgorithm(values.length);
            }

            if (lineNumber == 1 && argumentParser.getHeaderRow()) {
                writeHeader(values, out);
                continue;
            }

            processLine(values, out);
        }

        finish(out);
        out.flush();
    }

    protected void prepareAlgorithm(int dimensions) {
        pointBuffer = new double[dimensions];
        shingleBuilder = new ShingleBuilder(dimensions, argumentParser.getShingleSize(),
            argumentParser.getShingleCyclic());
        shingleBuffer = new double[shingleBuilder.getShingledPointSize()];

        RandomCutForest forest = RandomCutForest.builder()
            .numberOfTrees(argumentParser.getNumberOfTrees())
            .sampleSize(argumentParser.getSampleSize())
            .dimensions(shingleBuilder.getShingledPointSize())
            .lambda(argumentParser.getLambda())
            .randomSeed(argumentParser.getRandomSeed())
            .build();

        algorithm = algorithmInitializer.apply(forest);
    }

    protected void writeHeader(String[] values, PrintWriter out) {
        StringJoiner joiner = new StringJoiner(argumentParser.getDelimiter());
        Arrays.stream(values).forEach(joiner::add);
        algorithm.getResultColumnNames().forEach(joiner::add);
        out.println(joiner.toString());
    }

    protected void processLine(String[] values, PrintWriter out) {
        if (values.length != pointBuffer.length) {
            throw new IllegalArgumentException(
                String.format("Wrong number of values on line %d. Exected %d but found %d.",
                    lineNumber, pointBuffer.length, values.length));
        }

        parsePoint(values);
        shingleBuilder.addPoint(pointBuffer);

        List<String> result;
        if (shingleBuilder.isFull()) {
            shingleBuilder.getShingle(shingleBuffer);
            result = algorithm.getResultValues(shingleBuffer);
        } else {
            result = algorithm.getEmptyResultValue();
        }

        StringJoiner joiner = new StringJoiner(argumentParser.getDelimiter());
        Arrays.stream(values).forEach(joiner::add);
        result.forEach(joiner::add);

        out.println(joiner.toString());
    }

    protected void parsePoint(String... stringValues) {
        for (int i = 0; i < pointBuffer.length; i++) {
            pointBuffer[i] = Double.parseDouble(stringValues[i]);
        }
    }

    protected void finish(PrintWriter out) {

    }

    protected int getPointSize() {
        return pointBuffer != null ? pointBuffer.length : 0;
    }

    protected int getShingleSize() {
        return shingleBuffer != null ? shingleBuffer.length : 0;
    }
}
