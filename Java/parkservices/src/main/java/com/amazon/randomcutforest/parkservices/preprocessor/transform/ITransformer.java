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

import com.amazon.randomcutforest.parkservices.statistics.Deviation;
import com.amazon.randomcutforest.returntypes.RangeVector;

public interface ITransformer {

    double[] getWeights();

    void setWeights(double[] weights);

    Deviation[] getDeviations();

    double[] invert(double[] values, double[] correspondingInput);

    void invertForecastRange(RangeVector ranges, int baseDimension, double[] lastInput);

    void updateDeviation(double[] inputPoint, double[] lastInput);

    double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] lastInput, double[] factors,
            double clipFactor);
}