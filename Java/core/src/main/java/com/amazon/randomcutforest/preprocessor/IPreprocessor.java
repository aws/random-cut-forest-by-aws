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

package com.amazon.randomcutforest.preprocessor;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.ImputationMethod;
import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.returntypes.RangeVector;
import com.amazon.randomcutforest.returntypes.SampleSummary;

public interface IPreprocessor {

    int getShingleSize();

    int getInputLength();

    float[] getLastShingledPoint();

    double[] getShift();

    double[] getScale();

    double[] getSmoothedDeviations();

    int getInternalTimeStamp();

    int getValuesSeen();

    ImputationMethod getImputationMethod();

    double dataQuality();

    float[] getScaledInput(float[] point, long timestamp);

    float[] getScaledShingledInput(double[] point, long timestamp, int[] missing, RandomCutForest forest);

    SampleSummary invertSummary(SampleSummary summary, int numberOfMissing, int[] missingIndices, float[] point);

    void update(double[] point, float[] rcfPoint, long timestamp, int[] missing, RandomCutForest forest);

    boolean isForecastReasonable(boolean internalShingling);

    double[] getExpectedValue(int relativeBlockIndex, double[] reference, float[] point, float[] newPoint);

    double[] getShingledInput(int index);

    long getTimeStamp(int index);

    long inverseMapTime(double gap, int relativePosition);

    double getTransformDecay();

    int numberOfImputes(long timestamp);

    TimedRangeVector invertForecastRange(RangeVector ranges, long lastTimeStamp, double[] delta, boolean useExpected,
            long expectedTimeStamp);

}
