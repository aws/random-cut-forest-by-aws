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

/**
 * A simple command-line application that parses command-line arguments, creates
 * a RandomCutForest instance based on those arguments, reads values from STDIN
 * and writes results to STDOUT.
 */
public class SimpleRunner {

    protected final ArgumentParser argumentParser;
    protected final Function<RandomCutForest, LineTransformer> algorithmInitializer;
    protected LineTransformer algorithm;
    protected ShingleBuilder shingleBuilder;
    protected double[] pointBuffer;
    protected double[] shingleBuffer;
    protected int lineNumber;

    /**
     * Create a new SimpleRunner.
     * 
     * @param runnerClass          The name of the runner class. This will be
     *                             displayed in the help text.
     * @param runnerDescription    A description of the runner class. This will be
     *                             displayed in the help text.
     * @param algorithmInitializer A factory method to create a new LineTransformer
     *                             instance from a RandomCutForest.
     */
    public SimpleRunner(String runnerClass, String runnerDescription,
            Function<RandomCutForest, LineTransformer> algorithmInitializer) {
        this(new ArgumentParser(runnerClass, runnerDescription), algorithmInitializer);
    }

    /**
     * Create a new SimpleRunner.
     * 
     * @param argumentParser       A argument parser that will be used by this
     *                             runner to parse command-line arguments.
     * @param algorithmInitializer A factory method to create a new LineTransformer
     *                             instance from a RandomCutForest.
     */
    public SimpleRunner(ArgumentParser argumentParser,
            Function<RandomCutForest, LineTransformer> algorithmInitializer) {
        this.argumentParser = argumentParser;
        this.algorithmInitializer = algorithmInitializer;
    }

    /**
     * Parse the given command-line arguments.
     * 
     * @param arguments An array of command-line arguments.
     */
    public void parse(String... arguments) {
        argumentParser.parse(arguments);
    }

    /**
     * Read data from an input stream, apply the desired transformation, and write
     * the result to an output stream.
     * 
     * @param in  An input stream where input values will be read.
     * @param out An output stream where the result values will be written.
     * @throws IOException if IO errors are encountered during reading or writing.
     */
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

    /**
     * Set up the internal RandomCutForest instance and line transformer.
     * 
     * @param dimensions The number of dimensions in the input data.
     */
    protected void prepareAlgorithm(int dimensions) {
        pointBuffer = new double[dimensions];
        shingleBuilder = new ShingleBuilder(dimensions, argumentParser.getShingleSize(),
                argumentParser.getShingleCyclic());
        shingleBuffer = new double[shingleBuilder.getShingledPointSize()];

        RandomCutForest forest = RandomCutForest.builder().numberOfTrees(argumentParser.getNumberOfTrees())
                .sampleSize(argumentParser.getSampleSize()).dimensions(shingleBuilder.getShingledPointSize())
                .lambda(argumentParser.getLambda()).randomSeed(argumentParser.getRandomSeed()).build();

        algorithm = algorithmInitializer.apply(forest);
    }

    /**
     * Write a header row to the output stream.
     * 
     * @param values The array of values that are used to create the header. These
     *               values will be joined together using the user-specified
     *               delimiter.
     * @param out    The output stream where the header will be written.
     */
    protected void writeHeader(String[] values, PrintWriter out) {
        StringJoiner joiner = new StringJoiner(argumentParser.getDelimiter());
        Arrays.stream(values).forEach(joiner::add);
        algorithm.getResultColumnNames().forEach(joiner::add);
        out.println(joiner.toString());
    }

    /**
     * Process a single line of input data and write the result to the output
     * stream.
     * 
     * @param values An array of string values taken from the input stream. These
     *               values will be parsed into an array of doubles before being
     *               transformed and written to the output stream.
     * @param out    The output stream where the transformed line will be written.
     */
    protected void processLine(String[] values, PrintWriter out) {
        if (values.length != pointBuffer.length) {
            throw new IllegalArgumentException(
                    String.format("Wrong number of values on line %d. Exected %d but found %d.", lineNumber,
                            pointBuffer.length, values.length));
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

    /**
     * Parse the array of string values into doubles and write them to an internal
     * buffer.
     * 
     * @param stringValues An array of string-encoded double values.
     */
    protected void parsePoint(String[] stringValues) {
        for (int i = 0; i < pointBuffer.length; i++) {
            pointBuffer[i] = Double.parseDouble(stringValues[i]);
        }
    }

    /**
     * This method is used to write any final output to the output stream after the
     * input stream has beeen fully processed.
     * 
     * @param out The output stream where additional output text may be written.
     */
    protected void finish(PrintWriter out) {

    }

    /**
     * @return the size of the internal point buffer.
     */
    protected int getPointSize() {
        return pointBuffer != null ? pointBuffer.length : 0;
    }

    /**
     * @return the size of the internal shingled point buffer.
     */
    protected int getShingleSize() {
        return shingleBuffer != null ? shingleBuffer.length : 0;
    }
}
