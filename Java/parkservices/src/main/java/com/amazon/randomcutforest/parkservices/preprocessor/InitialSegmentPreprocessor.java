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

import static com.amazon.randomcutforest.parkservices.preprocessor.transform.WeightedTransformer.NUMBER_OF_STATS;

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.IRCFComputeDescriptor;
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
     * a modified preprocessing block which buffers the initial number of points
     * (startNormalization) and then switches to streaming transformation
     *
     * @param description           the description of the input point
     * @param lastAnomalyDescriptor the descriptor of the last anomaly
     * @param forest                RCF
     * @return an AnomalyDescriptor object to be used in anomaly detection
     */
    @Override
    public AnomalyDescriptor preProcess(AnomalyDescriptor description, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        initialSetup(description, lastAnomalyDescriptor, forest);

        if (valuesSeen < startNormalization) {
            storeInitial(description.getCurrentInput(), description.getInputTimestamp());
            return description;
        }

        return super.preProcess(description, lastAnomalyDescriptor, forest);
    }

    // same for post process
    @Override
    public AnomalyDescriptor postProcess(AnomalyDescriptor description, IRCFComputeDescriptor lastAnomalyDescriptor,
            RandomCutForest forest) {

        AnomalyDescriptor answer = super.postProcess(description, lastAnomalyDescriptor, forest);

        if (valuesSeen == startNormalization) {
            dischargeInitial(forest);
        }

        return answer;
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

    // computes the normalization statistics
    protected Deviation[] getDeviations() {
        if (requireInitialSegment(false, transformMethod)) {
            Deviation[] tempList = new Deviation[NUMBER_OF_STATS * inputLength];
            for (int j = 0; j < NUMBER_OF_STATS * inputLength; j++) {
                tempList[j] = new Deviation(transformDecay);
            }
            for (int i = 0; i < initialValues.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    tempList[j].update(initialValues[i][j]);
                    double value = (i == 0) ? 0 : initialValues[i][j] - initialValues[i - 1][j];
                    tempList[j + inputLength].update(value);
                }
            }
            for (int i = 0; i < initialValues.length; i++) {
                for (int j = 0; j < inputLength; j++) {
                    tempList[j + 2 * inputLength].update(tempList[j].getMean());
                }
            }
            return tempList;
        }
        return null;
    }

    /**
     * a block which executes once; it first computes the multipliers for
     * normalization and then processes each of the stored inputs
     */
    protected void dischargeInitial(RandomCutForest forest) {
        Deviation tempTimeDeviation = new Deviation();
        for (int i = 0; i < initialTimeStamps.length - 1; i++) {
            tempTimeDeviation.update(initialTimeStamps[i + 1] - initialTimeStamps[i]);
        }
        double timeFactor = tempTimeDeviation.getDeviation();
        Deviation[] deviations = getDeviations();
        for (int i = 0; i < valuesSeen; i++) {
            double[] scaledInput = getScaledInput(initialValues[i], initialTimeStamps[i], deviations, timeFactor);
            updateState(initialValues[i], scaledInput, initialTimeStamps[i], previousTimeStamps[shingleSize - 1]);
            dataQuality[0].update(1.0);
            forest.update(scaledInput);
        }

        initialTimeStamps = null;
        initialValues = null;
    }

}
