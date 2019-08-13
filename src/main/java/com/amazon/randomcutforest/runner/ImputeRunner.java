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
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.StringJoiner;

public class ImputeRunner extends SimpleRunner {

    private String missingValueMarker;
    private int numberOfMissingValues;
    private int[] missingIndexes;
    public ImputeRunner() {
        super(new ImputeArgumentParser(), UpdateOnlyTransformer::new);
    }

    public static void main(String... args) throws IOException {
        ImputeRunner runner = new ImputeRunner();
        runner.parse(args);
        System.out.println("Reading from stdin... (Ctrl-c to exit)");
        runner.run(
            new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
            new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))
        );
        System.out.println("Done.");
    }

    @Override
    protected void prepareAlgorithm(int dimensions) {
        super.prepareAlgorithm(dimensions);
        missingIndexes = new int[dimensions];
        missingValueMarker = ((ImputeArgumentParser) argumentParser).getMissingValueMarker();
    }

    @Override
    protected void processLine(String[] values, PrintWriter out) {

        numberOfMissingValues = 0;
        for (int i = 0; i < getPointSize(); i++) {
            if (missingValueMarker.equals(values[i])) {
                missingIndexes[numberOfMissingValues++] = i;
                values[i] = "0";
            }
        }

        if (numberOfMissingValues > 0) {
            parsePoint(values);
            double[] imputedPoint = algorithm.getForest().imputeMissingValues(pointBuffer, numberOfMissingValues,
                missingIndexes);
            StringJoiner joiner = new StringJoiner(argumentParser.getDelimiter());
            Arrays.stream(imputedPoint)
                .mapToObj(Double::toString)
                .forEach(joiner::add);
            out.println(joiner.toString());
        } else {
            super.processLine(values, out);
        }
    }

    public static class ImputeArgumentParser extends ArgumentParser {

        private final StringArgument missingValueMarker;

        public ImputeArgumentParser() {
            super(
                ImputeRunner.class.getName(),
                "Read rows with missing values and write rows with missing values imputed."
            );

            missingValueMarker = new StringArgument(
                null, "--missing-value-marker",
                "String used to represent a missing value in the data.",
                "NA"
            );

            addArgument(missingValueMarker);

            removeArgument("--shingle-size");
            removeArgument("--shingle-cyclic");
        }

        public String getMissingValueMarker() {
            return missingValueMarker.getValue();
        }
    }
}
