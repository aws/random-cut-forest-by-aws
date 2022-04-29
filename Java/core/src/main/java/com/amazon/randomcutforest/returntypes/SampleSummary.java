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

package com.amazon.randomcutforest.returntypes;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.toFloatArray;
import static com.amazon.randomcutforest.util.WeightedIndex.prefixPick;
import static java.lang.Math.max;
import static java.util.stream.Collectors.toCollection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.amazon.randomcutforest.util.WeightedIndex;

public class SampleSummary {

    /**
     * a collection of summarized points (reminiscent of typical sets from the
     * perspective of information theory, Cover and Thomas, Chapter 3) which should
     * be the mean/median of a spatially continuous distribution with central
     * tendency. If the input is a collection of samples that correspond to an union
     * of two such well separated distributions, for example as in the example data
     * of RCF paper then the output should be the two corresponding central points.
     */
    public float[][] summaryPoints;

    /**
     * a measure of comparison among the typical points;
     */
    public float[] relativeWeight;

    /**
     * number of samples, often the number of summary
     */
    public double weightOfSamples;
    /**
     * the global mean
     */
    public float[] mean;

    public float[] median;

    /**
     * This is the global deviation, without any filtering on the TreeSamples
     */
    public float[] deviation;

    public SampleSummary(double weightOfSamples, float[][] typicalPoints, float[] relativeLikelihood, float[] median,
            float[] mean, float[] deviation) {
        checkArgument(typicalPoints.length == relativeLikelihood.length, "incorrect lengths of fields");
        this.weightOfSamples = weightOfSamples;
        this.summaryPoints = typicalPoints;
        this.relativeWeight = relativeLikelihood;
        this.mean = mean;
        this.median = median;
        this.deviation = deviation;
    }

    public SampleSummary(int dimensions) {
        this.weightOfSamples = 0;
        this.summaryPoints = new float[1][];
        this.summaryPoints[0] = new float[dimensions];
        this.relativeWeight = new float[] { 0.0f };
        this.median = new float[dimensions];
        this.mean = new float[dimensions];
        this.deviation = new float[dimensions];
    }

    // for older tests
    public SampleSummary(double[] point) {
        this.weightOfSamples = 0;
        this.summaryPoints = new float[1][];
        this.summaryPoints[0] = new float[point.length];
        this.relativeWeight = new float[] { 0.0f };
        this.median = toFloatArray(point);
        this.mean = toFloatArray(point);
        this.deviation = new float[point.length];
    }

    public SampleSummary(int dimension, float[][] typicalPoints, float[] relativeLikelihood) {
        this.addTypical(dimension, typicalPoints, relativeLikelihood);
    }

    public void addTypical(int dimension, float[][] typicalPoints, float[] relativeLikelihood) {
        checkArgument(typicalPoints.length == relativeLikelihood.length, "incorrect lengths of fields");
        this.summaryPoints = new float[typicalPoints.length][];
        for (int i = 0; i < typicalPoints.length; i++) {
            checkArgument(dimension == typicalPoints[i].length, " incorrect length points");
            this.summaryPoints[i] = Arrays.copyOf(typicalPoints[i], dimension);
        }
        this.relativeWeight = Arrays.copyOf(relativeLikelihood, relativeLikelihood.length);
    }

    public SampleSummary(List<WeightedIndex<float[]>> points, float[][] typicalPoints, float[] relativeLikelihood) {
        this(points);
        this.addTypical(points.get(0).index.length, typicalPoints, relativeLikelihood);
    }

    public SampleSummary(List<WeightedIndex<float[]>> points) {
        checkArgument(points.size() > 0, "point list cannot be empty");
        int dimension = points.get(0).index.length;
        double[] coordinateSum = new double[dimension];
        double[] coordinateSumSquare = new double[dimension];
        double totalWeight = 0;
        for (WeightedIndex<float[]> e : points) {
            checkArgument(e.index.length == dimension, "points have to be of same length");
            float weight = e.weight;
            checkArgument(weight >= 0, "weights have to be non-negative");
            totalWeight += weight;
            for (int i = 0; i < dimension; i++) {
                checkArgument(!Float.isNaN(weight) && Float.isFinite(weight),
                        " weights must be finite, non-NaN values ");
                checkArgument(!Float.isNaN(e.index[i]) && Float.isFinite(e.index[i]),
                        " improper input, in coordinate " + i + ", must be finite, non-NaN values");
                coordinateSum[i] += e.index[i] * weight;
                coordinateSumSquare[i] += e.index[i] * e.index[i] * weight;
            }
        }
        checkArgument(totalWeight > 0, " weights cannot all be 0");
        this.weightOfSamples = totalWeight;
        this.mean = new float[dimension];
        this.deviation = new float[dimension];
        this.median = new float[dimension];

        for (int i = 0; i < dimension; i++) {
            this.mean[i] = (float) (coordinateSum[i] / totalWeight);
            this.deviation[i] = (float) Math.sqrt(max(0.0, coordinateSumSquare[i] / totalWeight - mean[i] * mean[i]));
        }
        for (int i = 0; i < dimension; i++) {
            int index = i;
            ArrayList<WeightedIndex<Float>> list = points.stream()
                    .map(e -> new WeightedIndex<>(e.index[index], e.weight)).collect(toCollection(ArrayList::new));
            list.sort((o1, o2) -> Float.compare(o1.index, o2.index));
            this.median[i] = prefixPick(list, totalWeight / 2.0).index;
        }
    }

}
