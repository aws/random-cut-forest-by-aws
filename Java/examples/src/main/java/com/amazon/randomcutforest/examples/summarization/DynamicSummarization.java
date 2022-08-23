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

package com.amazon.randomcutforest.examples.summarization;

import static com.amazon.randomcutforest.summarization.Summarizer.DEFAULT_SEPARATION_RATIO_FOR_MERGE;
import static com.amazon.randomcutforest.testutils.ExampleDataSets.rotateClockWise;
import static java.lang.Math.PI;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;
import com.amazon.randomcutforest.util.Weighted;

/**
 * Summarized representation of the stored points provide a convenient view into
 * the "current state" of the stream seen/sampled by an RCF. However since RCFs
 * provide a generic sketch for multple different scenrios
 * https://opensearch.org/blog/odfe-updates/2019/11/random-cut-forests/ the
 * summarization can be used repeatedly to provide a dynamic clustering a
 * numeric data stream as shown in the example below.
 *
 * The summarization is based on a well-scattered multi-centroid representation
 * as in CURE https://en.wikipedia.org/wiki/CURE_algorithm and distance based
 * clustering as in https://en.wikipedia.org/wiki/Data_stream_clustering
 *
 * The example corresponds to a wheel like arrangement -- where numberOfBlades
 * determine the number of spokes. For many settings of the parameter the spokes
 * are closer to each other near the center than the extremity at the rim. Thus
 * a centroidal representation cannot conceptually capture each spoke as a
 * cluster, and multi-centroid approach is necessary. Note that the input to the
 * summarization is not the same as the numberOfBladed; the maxAllowed number
 * corresponds to the maximum number of clusters which can be much larger. In a
 * clustering application, the number of clusters are typically not known
 * apriori.
 *
 * The pointset is generated once and are input to RCF with rotations. As the
 * "blades are running", the output clusters can be colored and we can visualize
 * the clusters produced. For the parameters below, simplistic plotting
 * functions such as gnuplot using do for [i = 0:359] { plot [-15:15][-15:15]
 * "sum" index i u 1:2:3:4 w circles fill solid noborder fc palette z t "" }
 * would show the rotating clusters where the representatives corresponding to
 * the same cluster has the same color. We note that the visualizations is
 * neither polished nor complete, since the goal is to highlight the
 * functionality of summarization in RCFs.
 */
public class DynamicSummarization implements Example {

    public static void main(String[] args) throws Exception {
        new DynamicSummarization().run();
    }

    @Override
    public String command() {
        return "dynamic_summarization";
    }

    @Override
    public String description() {
        return "shows a potential use of dynamic clustering/summarization";
    }

    @Override
    public void run() throws Exception {
        int newDimensions = 2;
        long randomSeed = 123;
        int dataSize = 1350;
        int numberOfBlades = 3;

        RandomCutForest newForest = RandomCutForest.builder().numberOfTrees(100).sampleSize(256)
                .dimensions(newDimensions).randomSeed(randomSeed).timeDecay(1.0 / 800).centerOfMassEnabled(true)
                .build();
        String name = "dynamic_summarization_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));
        double[][] data = getData(dataSize, 0, numberOfBlades);

        boolean printData = true;
        boolean printClusters = false;

        List<ICluster<float[]>> oldSummary = null;
        int[] oldColors = null;

        int count = 0;
        int sum = 0;
        for (int degree = 0; degree < 360; degree += 1) {
            for (double[] datum : data) {
                double[] vec = rotateClockWise(datum, -2 * PI * degree / 360);
                if (printData) {
                    file.append(vec[0] + " " + vec[1] + "\n");
                }
                newForest.update(vec);
            }
            if (printData) {
                file.append("\n");
                file.append("\n");
            }

            double shrinkage = 1.0 / (2 * numberOfBlades);
            // if the maxAllowed is increased, decrease the shrikage ratio to get
            // counterbalance the test for
            // sumary.size() == numberOfFans
            // for example,
            // List<ICluster<float[]>> summary =
            // newForest.summarize(5*numberOfFans,0.1,5,0.5,oldSummary);

            List<ICluster<float[]>> summary = newForest.summarize(2 * numberOfBlades, shrinkage, 5,
                    DEFAULT_SEPARATION_RATIO_FOR_MERGE, oldSummary);
            sum += summary.size();
            if (summary.size() == numberOfBlades) {
                ++count;
            }
            int[] colors = align(summary, oldSummary, oldColors);

            for (int i = 0; i < summary.size(); i++) {
                double weight = summary.get(i).getWeight();
                for (Weighted<float[]> representative : summary.get(i).getRepresentatives()) {
                    double t = representative.weight / weight;
                    if (t > 0.05 && printClusters) {
                        file.append(representative.index[0] + " " + representative.index[1] + " " + t + " " + colors[i]
                                + "\n");
                    }
                }
            }
            if (summary.size() == numberOfBlades) {
                oldSummary = summary;
                oldColors = colors;
            }
            if (printClusters) {
                file.append("\n");
                file.append("\n");
            }

        }
        System.out.println("Exact detection :" + ((float) Math.round(count / 3.6) * 0.01)
                + " fraction, average number of clusters " + ((float) Math.round(sum / 3.6) * 0.01));
        file.close();
    }

    public double[][] getData(int dataSize, int seed, int fans) {
        Random prg = new Random(0);
        NormalMixtureTestData generator = new NormalMixtureTestData(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        int newDimensions = 2;
        double[][] data = generator.generateTestData(dataSize, newDimensions, seed);

        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale, make an ellipse
            data[i][1] *= 1.0 / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += 5.0 + fans / 2;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    int[] align(List<ICluster<float[]>> current, List<ICluster<float[]>> previous, int[] oldColors) {
        int[] nearest = new int[current.size()];

        if (previous == null || previous.size() == 0) {
            for (int i = 0; i < current.size(); i++) {
                nearest[i] = i;
            }
        } else {
            Arrays.fill(nearest, previous.size() + 1);
            for (int i = 0; i < current.size(); i++) {
                double dist = previous.get(0).distance(current.get(i), Summarizer::L1distance);
                nearest[i] = oldColors[0];
                for (int j = 1; j < previous.size(); j++) {
                    double t = previous.get(j).distance(current.get(i), Summarizer::L1distance);
                    if (t < dist) {
                        dist = t;
                        nearest[i] = oldColors[j];
                    }
                }
            }
        }
        return nearest;
    }
}
