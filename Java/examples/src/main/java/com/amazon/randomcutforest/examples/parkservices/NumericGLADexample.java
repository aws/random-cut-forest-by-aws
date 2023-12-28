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

package com.amazon.randomcutforest.examples.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.testutils.ExampleDataSets.rotateClockWise;
import static java.lang.Math.PI;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;

import com.amazon.randomcutforest.examples.Example;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.GlobalLocalAnomalyDetector;
import com.amazon.randomcutforest.parkservices.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.config.ScoringStrategy;
import com.amazon.randomcutforest.parkservices.returntypes.GenericAnomalyDescriptor;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.testutils.NormalMixtureTestData;

/**
 * The following example demonstrates clustering based anomaly detection for
 * numeric vectors. The clustering can use an arbitrary distance metric (but it
 * has no mechanism to verify if the function provided is a metric beyond
 * checking that distances are non-negative; improper implementations of
 * distances can produce uninterpretable results). The clustering corresponds to
 * clustering a recency biased sample of points (using the exact same as RCF)
 * and clustering using multi-centroid method (CURE algorithm).
 *
 * There is a natural question that given that this is the RCF library, how does
 * this clustering based algorithm perform vis-a-vis RCF. First, RCF is
 * preferred/natural for shingled/sequenced data, e.g., in analysis of time
 * series. Simple clustering of shingles do not seem to provide similar benefit.
 * In fact, even for shinglesize 1, which correponds to time dependent
 * population analysis, the recursive decomposition provided by RCF can provide
 * a richer detail (even though RCF naturally considers the L1/Manhattan
 * metric). That recursive decomposition can be viewed as a (randomized) partion
 * based clustering. That distance function is used to compute the DensityOutput
 * in RCF. Multilevel clustering is known to be more useful than simple
 * clustering in many applications. Here we show such an application which
 *
 * (i) shows an example use of GlobalLocalAnomalyDetector (GLAD) for dynamic
 * data as well as
 *
 * (ii) a comparable use using a new ForestMode.DISTANCE exposed for RCF.
 *
 * RCF seems to perform better for this simple two dimensional dynamic case. At
 * the same time, the new clusering based algorithm works for generic types with
 * just a distance function. In applications where distances are meaningful and
 * key, such geo etc., user-defined distance based anomalies can be extremely
 * beneficial. If the data can be mapped to explicit vectors then perhaps RCF
 * and its multi-level partitioning can provide more useful insights.
 *
 * Try the following in a visualizer. For example in vanilla gnuplot try
 *
 * set terminal gif transparent animate delay 5
 *
 * set size square
 *
 * set output "test.gif"
 *
 * do for [i = 0:359] { plot [-15:15][-15:15] "clustering_example" i i u 1:2:3 w
 * p palette pt 7 t "" }
 *
 *
 * Try the above/equivalent for setting printFlaggedGLAD = true (setting
 * printFlaggedRCF = false), or to see the data, printData = true. Try changing
 * the number of blades in the fan, the zFactor setting etc.
 */
public class NumericGLADexample implements Example {

    public static void main(String[] args) throws Exception {
        new NumericGLADexample().run();
    }

    @Override
    public String command() {
        return "An example of Global-Local Anomaly Detector on numeric vectors";
    }

    @Override
    public String description() {
        return "An example of Global-Local Anomaly Detector on numeric vectors";
    }

    @Override
    public void run() throws Exception {
        long randomSeed = new Random().nextLong();
        System.out.println("Seed " + randomSeed);
        // we would be sending dataSize * 360 vectors
        int dataSize = 2000;
        double range = 10.0;
        int numberOfFans = 3;
        // corresponds to number of clusters
        double[][] data = shiftedEllipse(dataSize, 7, range / 2, numberOfFans);
        int truePos = 0;
        int falsePos = 0;
        int falseNeg = 0;

        int truePosRCF = 0;
        int falsePosRCF = 0;
        int falseNegRCF = 0;

        int reservoirSize = dataSize;

        // this ensures that the points are flushed out (albeit randomly) duting the
        // rotation
        double timedecay = 1.0 / reservoirSize;

        GlobalLocalAnomalyDetector<float[]> reservoir = GlobalLocalAnomalyDetector.builder().randomSeed(42)
                .numberOfRepresentatives(3).timeDecay(timedecay).capacity(reservoirSize).build();
        reservoir.setGlobalDistance(Summarizer::L2distance);

        double zFactor = 6.0; // six sigma deviation; seems to work best
        reservoir.setZfactor(zFactor);

        ThresholdedRandomCutForest test = ThresholdedRandomCutForest.builder().dimensions(2).shingleSize(1)
                .randomSeed(77).timeDecay(timedecay).scoringStrategy(ScoringStrategy.DISTANCE).build();
        test.setZfactor(zFactor); // using the zFactor for same apples to apples comparison

        String name = "clustering_example";
        BufferedWriter file = new BufferedWriter(new FileWriter(name));
        boolean printData = true;
        boolean printAnomalies = false;
        // use one or the other prints below
        boolean printFlaggedRCF = false;
        boolean printFlaggedGLAD = true;

        Random noiseGen = new Random(randomSeed + 1);
        for (int degree = 0; degree < 360; degree += 1) {
            int index = 0;
            while (index < data.length) {
                boolean injected = false;
                float[] vec;
                if (noiseGen.nextDouble() < 0.005) {
                    injected = true;
                    double[] candAnomaly = new double[2];
                    // generate points along x axis
                    candAnomaly[0] = (range / 2 * noiseGen.nextDouble() + range / 2);
                    candAnomaly[1] = 0.1 * (2.0 * noiseGen.nextDouble() - 1.0);
                    int antiFan = noiseGen.nextInt(numberOfFans);
                    // rotate to be 90-180 degrees away -- these are decidedly anomalous
                    vec = toFloatArray(rotateClockWise(candAnomaly,
                            -2 * PI * (degree + 180 * (1 + 2 * antiFan) / numberOfFans) / 360));
                    if (printAnomalies) {
                        file.append(vec[0] + " " + vec[1] + " " + 0.0 + "\n");
                    }
                } else {
                    vec = toFloatArray(rotateClockWise(data[index], -2 * PI * degree / 360));
                    if (printData) {
                        file.append(vec[0] + " " + vec[1] + " " + 0.0 + "\n");
                    }
                    ++index;
                }

                GenericAnomalyDescriptor<float[]> result = reservoir.process(vec, 1.0f, null, true);

                AnomalyDescriptor res = test.process(toDoubleArray(vec), 0L);
                double grade = res.getAnomalyGrade();

                if (injected) {
                    if (result.getAnomalyGrade() > 0) {
                        ++truePos;
                    } else {
                        ++falseNeg;
                    }
                    if (grade > 0) {
                        ++truePosRCF;
                    } else {
                        ++falseNegRCF;
                    }
                } else {
                    if (result.getAnomalyGrade() > 0) {
                        ++falsePos;
                    }
                    if (grade > 0) {
                        ++falsePosRCF;
                    }
                }
                if (printFlaggedRCF && grade > 0) {
                    file.append(vec[0] + " " + vec[1] + " " + grade + "\n");
                } else if (printFlaggedGLAD && result.getAnomalyGrade() > 0) {
                    file.append(vec[0] + " " + vec[1] + " " + result.getAnomalyGrade() + "\n");
                }
            }
            if (printAnomalies || printData || printFlaggedRCF || printFlaggedGLAD) {
                file.append("\n");
                file.append("\n");
            }

            if (falsePos + truePos == 0) {
                throw new IllegalStateException("");
            }

            checkArgument(falseNeg + truePos == falseNegRCF + truePosRCF, " incorrect accounting");
            System.out.println(" at degree " + degree + " injected " + (truePos + falseNeg));
            System.out.print("Precision = " + precision(truePos, falsePos));
            System.out.println(" Recall = " + recall(truePos, falseNeg));
            System.out.print("RCF Distance Mode Precision = " + precision(truePosRCF, falsePosRCF));
            System.out.println(" RCF Distance Mode Recall = " + recall(truePosRCF, falseNegRCF));

        }
    }

    public double[][] shiftedEllipse(int dataSize, int seed, double shift, int fans) {
        NormalMixtureTestData generator = new NormalMixtureTestData(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
        double[][] data = generator.generateTestData(dataSize, 2, seed);
        Random prg = new Random(0);
        for (int i = 0; i < dataSize; i++) {
            int nextFan = prg.nextInt(fans);
            // scale
            data[i][1] *= 1.0 / fans;
            data[i][0] *= 2.0;
            // shift
            data[i][0] += shift + 1.0 / fans;
            data[i] = rotateClockWise(data[i], 2 * PI * nextFan / fans);
        }

        return data;
    }

    double precision(int truePos, int falsePos) {
        return (truePos + falsePos > 0) ? 1.0 * truePos / (truePos + falsePos) : 1.0;
    }

    double recall(int truePos, int falseNeg) {
        return (truePos + falseNeg > 0) ? 1.0 * truePos / (truePos + falseNeg) : 1.0;
    }

}
