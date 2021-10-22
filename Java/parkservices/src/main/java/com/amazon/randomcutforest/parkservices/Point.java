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

/**
 * a basic class that defines a proto-point
 */
@Getter
@Setter
public class Point {

    // current values
    double[] currentInput;

    // input timestamp
    long inputTimestamp;

    public Point(double[] input, long inputTimestamp) {
        this.currentInput = copyIfNotnull(input);
        this.inputTimestamp = inputTimestamp;
    }

    public void setCurrentInput(double[] currentValues) {
        this.currentInput = copyIfNotnull(currentValues);
    }

    public double[] getCurrentInput() {
        return copyIfNotnull(currentInput);
    }

    protected double[] copyIfNotnull(double[] array) {
        return array == null ? null : Arrays.copyOf(array, array.length);
    }

    // an explicit copy operation to control the stored state
    public Point copyOf() {
        return new Point(currentInput, inputTimestamp);
    }
}
