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

package com.amazon.randomcutforest.parkservices.preprocessor.transform;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class EmptyTransformer implements ITransformer {

    public EmptyTransformer() {
    }

    @Override
    public double[] getWeights() {
        return null;
    }

    @Override
    public void setWeights(double[] weights) {
    }

    @Override
    public Deviation[] getDeviations() {
        return null;
    }

    @Override
    public double[] invert(double[] values, double[] correspondingInput) {
        return Arrays.copyOf(values, values.length);
    }

    @Override
    public void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput) {

    }

    @Override
    public void updateDeviation(double[] inputPoint, double[] lastInput) {
    }

    @Override
    public double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] lastInput, double[] factors,
            double clipFactor) {
        return Arrays.copyOf(inputPoint, inputPoint.length);
    }
}
