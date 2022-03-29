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

}
