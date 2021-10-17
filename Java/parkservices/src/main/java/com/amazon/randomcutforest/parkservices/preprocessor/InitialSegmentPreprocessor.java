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

package com.amazon.randomcutforest.parkservices.preprocessor;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.config.TransformMethod;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.statistics.Deviation;

@Getter
@Setter
public class InitialSegmentPreprocessor extends Preprocessor {

    public InitialSegmentPreprocessor(Builder<?> builder) {
        super(builder);
        initialValues = new double[startNormalization][];
        initialTimeStamps = new long[startNormalization];
    }

    /**
     * A core function of the preprocessor. It can augment time values (with
     * normalization) or impute missing values on the fly using the forest.
     *
     * @param inputPoint the actual input
     * @param timestamp  timestamp of the point
     * @param forest     RCF
     * @return a scaled/normalized tuple that can be used for anomaly detection
     */
    @Override
    public AnomalyDescriptor preProcess(double[] inputPoint, long timestamp, RandomCutForest forest,
            long lastAnomalyTimeStamp, double[] lastExpectedValue) {

        if (valuesSeen < startNormalization) {
            storeInitial(inputPoint, timestamp);
            return null;
        } else if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        return super.preProcess(inputPoint, timestamp, forest, lastAnomalyTimeStamp, lastExpectedValue);
    }

    /**
     * stores initial data for normalization
     *
     * @param inputPoint input data
     * @param timestamp  timestamp
     */
    protected void storeInitial(double[] inputPoint, long timestamp) {
        initialTimeStamps[valuesSeen] = timestamp;
        initialValues[valuesSeen] = Arrays.copyOf(inputPoint, inputPoint.length);
        ++valuesSeen;
    }

    /**
     * an execute once block which first computes the multipliers for normalization
     * and then processes each of the stored inputs
     */
    protected void dischargeInitial(RandomCutForest forest) {
        Deviation tempTimeDeviation = new Deviation();
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            tempTimeDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double timeFactor = tempTimeDeviation.getDeviation();
        double[] factors = null;
        if (requireInitialSegment(false, transformMethod)) {
            Deviation[] tempList = new Deviation[inputLength];
            for (int j = 0; j < inputLength; j++) {
                tempList[j] = new Deviation(deviationList[j].getDiscount());
            }
            for (int i = 0; i < initialValues.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    double value;
                    if (transformMethod == TransformMethod.NORMALIZE) {
                        value = initialValues[i][j];
                    } else {
                        value = (i == 0) ? 0 : initialValues[i][j] - initialValues[i - 1][j];
                    }
                    tempList[j].update(value);
                }
            }
            factors = new double[inputLength];
            for (int j = 0; j < inputLength; j++) {
                factors[j] = tempList[j].getDeviation();
            }
        }

        for (int i = 0; i < valuesSeen; i++) {
            double[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], factors, timeFactor);
            if (internalTimeStamp > 0) {
                timeStampDeviation.update(initialTimeStamps[i] - previousTimeStamps[shingleSize - 1]);
            }
            updateState(initialValues[i], scaledInput, initialTimeStamps[i]);
            forest.update(scaledInput);
        }

        initialTimeStamps = null;
        initialValues = null;
    }

}
