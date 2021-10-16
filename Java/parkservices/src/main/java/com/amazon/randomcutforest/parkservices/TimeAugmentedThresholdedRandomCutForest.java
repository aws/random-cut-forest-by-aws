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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.preprocessor.Preprocessor;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;

/**
 * This class provides a combined RCF and thresholder, both of which operate in
 * a streaming manner and respect the arrow of time.
 */
@Getter
@Setter
public class TimeAugmentedThresholdedRandomCutForest extends ThresholdedRandomCutForest {

    public TimeAugmentedThresholdedRandomCutForest(Builder<?> builder) {
        super(builder);
        checkArgument(builder.forestMode == ForestMode.TIME_AUGMENTED, "incorrect invocation");
        forestMode = ForestMode.TIME_AUGMENTED;
    }

    // for mappers
    public TimeAugmentedThresholdedRandomCutForest(RandomCutForest forest, BasicThresholder thresholder,
            Preprocessor preprocessor) {
        super(forest, thresholder, preprocessor);
        checkArgument(forest.isInternalShinglingEnabled(), " time augmentation requires internal shingling");
        forestMode = ForestMode.TIME_AUGMENTED;
    }

    /**
     * a single call that prepreprocesses data, compute score/grade and updates
     * state
     *
     * @param inputPoint current input point
     * @param timestamp  time stamp of input
     * @return anomalydescriptor for the current input point
     */
    public AnomalyDescriptor process(double[] inputPoint, long timestamp) {

        boolean ifZero = (forest.getBoundingBoxCacheFraction() == 0);
        if (ifZero) { // turn caching on temporarily
            forest.setBoundingBoxCacheFraction(1.0);
        }

        double[] scaledInput = preprocessor.preProcess(inputPoint, timestamp, forest, lastAnomalyTimeStamp,
                lastExpectedPoint);
        if (scaledInput == null) {
            return new AnomalyDescriptor();
        }

        // only internal shingling
        double[] point = forest.transformToShingledPoint(scaledInput);

        // score anomalies
        AnomalyDescriptor description = getAnomalyDescription(point, timestamp, inputPoint);

        // add expected value, update state
        AnomalyDescriptor result = preprocessor.postProcess(description, inputPoint, timestamp, forest);
        if (ifZero) { // turn caching off
            forest.setBoundingBoxCacheFraction(0);
        }
        return result;

    }

    /**
     * a first stage corrector that attempts to fix the after effects of a previous
     * anomaly which may be in the shingle, or just preceding the shingle
     * 
     * @param point          the current (transformed) point under evaluation
     * @param gap            the relative position of the previous anomaly being
     *                       corrected
     * @param shingleSize    size of the shingle
     * @param baseDimensions number of dimensions in each shingle
     * @return the score of the corrected point
     */
    @Override
    double[] applyBasicCorrector(double[] point, int gap, int shingleSize, int baseDimensions) {
        double[] correctedPoint = super.applyBasicCorrector(point, gap, shingleSize, baseDimensions);
        if (lastRelativeIndex == 0 && transformMethod != TransformMethod.DIFFERENCE
                && transformMethod != TransformMethod.NORMALIZE_DIFFERENCE) {
            // definitely correct the time dimension which is always differenced
            // this applies to the non-differenced cases
            correctedPoint[point.length - (gap - 1) * baseDimensions - 1] += lastAnomalyPoint[point.length - 1]
                    - lastExpectedPoint[point.length - 1];

        }
        return correctedPoint;
    }

}
