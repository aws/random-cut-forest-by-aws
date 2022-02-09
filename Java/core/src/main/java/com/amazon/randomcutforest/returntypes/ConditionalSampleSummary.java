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

public class ConditionalSampleSummary {

    /**
     * a collection of typical points (in the shingled, input space) seen from the
     * sampling
     */
    public float[][] typicalPoints;

    /**
     * a measure of comparison among the typical points;
     */
    public float[] relativeLikelihood;

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

    public ConditionalSampleSummary(double weightOfSamples, float[][] typicalPoints, float[] relativeLikelihood,
            float[] median, float[] mean, float[] deviation) {
        checkArgument(typicalPoints.length == relativeLikelihood.length, "incorrect lengths of fields");
        this.weightOfSamples = weightOfSamples;
        this.typicalPoints = typicalPoints;
        this.relativeLikelihood = relativeLikelihood;
        this.mean = mean;
        this.median = median;
        this.deviation = deviation;
    }

    public ConditionalSampleSummary(int dimensions) {
        this.weightOfSamples = 0;
        this.typicalPoints = new float[1][];
        typicalPoints[0] = new float[dimensions];
        this.relativeLikelihood = new float[] { 0.0f };
        this.median = new float[dimensions];
        this.mean = new float[dimensions];
        this.deviation = new float[dimensions];
    }

    // for older tests
    public ConditionalSampleSummary(double[] point) {
        this.weightOfSamples = 0;
        this.typicalPoints = new float[1][];
        typicalPoints[0] = new float[point.length];
        this.relativeLikelihood = new float[] { 0.0f };
        this.median = toFloatArray(point);
        this.mean = toFloatArray(point);
        this.deviation = new float[point.length];
    }

}
