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
import java.util.Collections;
import java.util.List;

import com.amazon.randomcutforest.RandomCutForest;

public class AnomalyScoreRunner extends SimpleRunner {

    public AnomalyScoreRunner() {
        super(
            AnomalyScoreRunner.class.getName(),
            "Compute scalar anomaly scores from the input rows and append them to the output rows.",
            AnomalyScoreTransformer::new
        );
    }

    public static void main(String... args) throws IOException {
        AnomalyScoreRunner runner = new AnomalyScoreRunner();
        runner.parse(args);
        System.out.println("Reading from stdin... (Ctrl-c to exit)");
        runner.run(
            new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8)),
            new PrintWriter(new OutputStreamWriter(System.out, StandardCharsets.UTF_8))
        );
        System.out.println("Done.");
    }

    public static class AnomalyScoreTransformer implements LineTransformer {
        private final RandomCutForest forest;

        public AnomalyScoreTransformer(RandomCutForest forest) {
            this.forest = forest;
        }

        @Override
        public List<String> getResultValues(double... point) {
            double score = forest.getAnomalyScore(point);
            forest.update(point);
            return Collections.singletonList(Double.toString(score));
        }

        @Override
        public List<String> getEmptyResultValue() {
            return Collections.singletonList("NA");
        }

        @Override
        public List<String> getResultColumnNames() {
            return Collections.singletonList("anomaly_score");
        }

        @Override
        public RandomCutForest getForest() {
            return forest;
        }
    }
}
