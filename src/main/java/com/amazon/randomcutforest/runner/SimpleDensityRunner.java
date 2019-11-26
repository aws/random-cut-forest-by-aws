/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import java.util.ArrayList;
import java.util.List;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.returntypes.DiVector;

/**
 * A command-line application that computes directional density. Points are read from STDIN and output is written to
 * STDOUT. Output consists of the original input point with the directional density vector appended.
 */
public class SimpleDensityRunner extends SimpleRunner {

    public SimpleDensityRunner() {
        super(
            SimpleDensityRunner.class.getName(),
            "Compute directional density vectors from the input rows and append them to the output rows.",
            SimpleDensityRunner.SimpleDensityTransformer::new
        );
    }

    public static void main(String... args) throws IOException {
        SimpleDensityRunner runner = new SimpleDensityRunner();
        runner.parse(args);
        System.out.println("Reading from stdin... (Ctrl-c to exit)");
        runner.run(
            new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
            new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))
        );
        System.out.println("Done.");
    }

    public static class SimpleDensityTransformer implements LineTransformer {
        private final RandomCutForest forest;

        public SimpleDensityTransformer(RandomCutForest forest) {
            this.forest = forest;
        }

        @Override
        public List<String> getResultValues(double... point) {
            DiVector densityFactors = forest.getSimpleDensity(point).getDirectionalDensity();
            forest.update(point);

            List<String> result = new ArrayList<>(2 * forest.getDimensions());
            for (int i = 0; i < forest.getDimensions(); i++) {
                result.add(String.format("%f", densityFactors.high[i]));
                result.add(String.format("%f", densityFactors.low[i]));
            }
            return result;
        }

        @Override
        public List<String> getEmptyResultValue() {
            List<String> result = new ArrayList<>(2 * forest.getDimensions());
            for (int i = 0; i < 2 * forest.getDimensions(); i++) {
                result.add("NA");
            }
            return result;
        }

        @Override
        public List<String> getResultColumnNames() {
            List<String> result = new ArrayList<>(2 * forest.getDimensions());
            for (int i = 0; i < forest.getDimensions(); i++) {
                result.add(String.format("prob_mass_%d_up", i));
                result.add(String.format("prob_mass_%d_down", i));
            }
            return result;
        }

        @Override
        public RandomCutForest getForest() {
            return forest;
        }
    }
}
