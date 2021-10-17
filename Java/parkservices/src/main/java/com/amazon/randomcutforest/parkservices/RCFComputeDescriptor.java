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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class RCFComputeDescriptor {

    // sequence index (the number of updates to RCF) -- it is possible in imputation
    // that
    // the number of updates more than the input tuples seen by the overall program
    long totalUpdates;

    // internal timestamp (basically a sequence index, but can be scaled and
    // jittered as in
    // the example);
    // kept as long for potential future use
    long internalTimeStamp;

    // number of trees in the forest
    int numberOfTrees;

    // current values
    double[] currentInput;

    // input timestamp
    long inputTimestamp;

    // potential number of imputes before processing current point
    int numberOfImputes;

    // actual, potentially transformed point on which compute occurs
    double[] RCFPoint;

    // expected RCFPoint for the current point
    double[] expectedRCFPoint;

    // internal timestamp of last anomaly
    long lastAnomalyInternalTimestamp;

    // expected point at anomaly
    double[] lastExpectedPoint;

    public void setCurrentInput(double[] currentValues) {
        this.currentInput = copyIfNotnull(currentValues);
    }

    public double[] getCurrentInput() {
        return copyIfNotnull(currentInput);
    }

    public void setExpectedRCFPoint(double[] point) {
        expectedRCFPoint = copyIfNotnull(point);
    }

    public double[] getExpectedRCFPoint() {
        return copyIfNotnull(expectedRCFPoint);
    }

    public void setRCFPoint(double[] point) {
        RCFPoint = copyIfNotnull(point);
    }

    public double[] getRCFPoint() {
        return copyIfNotnull(RCFPoint);
    }

    public void setLastExpectedPoint(double[] point) {
        lastExpectedPoint = copyIfNotnull(point);
    }

    public double[] getLastExpectedPoint() {
        return copyIfNotnull(lastExpectedPoint);
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }
}
