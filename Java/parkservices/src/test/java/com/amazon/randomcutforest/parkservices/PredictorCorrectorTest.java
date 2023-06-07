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

import static com.amazon.randomcutforest.config.TransformMethod.NORMALIZE;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.returntypes.DiVector;

public class PredictorCorrectorTest {

    @Test
    void AttributorTest() {
        int sampleSize = 256;
        int baseDimensions = 10;
        int shingleSize = 10;
        int dimensions = baseDimensions * shingleSize;
        ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder().compact(true).dimensions(dimensions)
                .precision(Precision.FLOAT_32).randomSeed(0L).forestMode(ForestMode.STANDARD).shingleSize(shingleSize)
                .anomalyRate(0.01).transformMethod(NORMALIZE).build();
        DiVector test = new DiVector(baseDimensions * shingleSize);
        assert (forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null) == null);
        assertThrows(IllegalArgumentException.class, () -> forest.predictorCorrector.setNumberOfAttributors(-1));
        forest.predictorCorrector.setNumberOfAttributors(baseDimensions);
        assertThrows(NullPointerException.class,
                () -> forest.predictorCorrector.getExpectedPoint(test, 0, baseDimensions, null, null));
    }

}
