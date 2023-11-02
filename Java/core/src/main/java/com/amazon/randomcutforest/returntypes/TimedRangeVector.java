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

package com.amazon.randomcutforest.parkservices.returntypes;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Arrays;

import com.amazon.randomcutforest.returntypes.RangeVector;

/**
 * ThresholdedRandomCutForests handle time internally and thus the forecast of
 * values also correpond to the next sequential timestamps. The RangeVector
 * corresponds to the forecast from RCF (based on the inverse of the
 * transformation applied by TRCF as it invokes RCF). The timeStamps correspond
 * to the predicted timestamps The upper and lower ranges are also present
 * similar to RangeVector
 *
 * Note that if the timestamps cannot be predicted meaningfully (for example in
 * STREAMING_IMPUTE mode), then those entries would be 0
 */
public class TimedRangeVector {

    public final RangeVector rangeVector;

    public final long[] timeStamps;

    public final long[] upperTimeStamps;

    public final long[] lowerTimeStamps;

    public TimedRangeVector(int dimensions, int horizon) {
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        checkArgument(horizon > 0, "horizon must be greater than 0");
        checkArgument(dimensions % horizon == 0, "horizon should divide dimensions");
        rangeVector = new RangeVector(dimensions);
        timeStamps = new long[horizon];
        upperTimeStamps = new long[horizon];
        lowerTimeStamps = new long[horizon];
    }

    public TimedRangeVector(RangeVector rangeVector, long[] timestamps, long[] upperTimeStamps,
            long[] lowerTimeStamps) {
        checkArgument(rangeVector.values.length % timestamps.length == 0,
                " dimensions must be be divisible by horizon");
        checkArgument(timestamps.length == upperTimeStamps.length && upperTimeStamps.length == lowerTimeStamps.length,
                "horizon must be equal");
        this.rangeVector = new RangeVector(rangeVector);
        for (int i = 0; i < timestamps.length; i++) {
            checkArgument(upperTimeStamps[i] >= timestamps[i] && timestamps[i] >= lowerTimeStamps[i],
                    "incorrect semantics");
        }
        this.timeStamps = Arrays.copyOf(timestamps, timestamps.length);
        this.lowerTimeStamps = Arrays.copyOf(lowerTimeStamps, lowerTimeStamps.length);
        this.upperTimeStamps = Arrays.copyOf(upperTimeStamps, upperTimeStamps.length);
    }

    public TimedRangeVector(TimedRangeVector base) {
        this.rangeVector = new RangeVector(base.rangeVector);
        this.timeStamps = Arrays.copyOf(base.timeStamps, base.timeStamps.length);
        this.lowerTimeStamps = Arrays.copyOf(base.lowerTimeStamps, base.lowerTimeStamps.length);
        this.upperTimeStamps = Arrays.copyOf(base.upperTimeStamps, base.upperTimeStamps.length);
    }

    /**
     * Create a deep copy of the base RangeVector.
     *
     * @param base The RangeVector to copy.
     */
    public TimedRangeVector(RangeVector base, int horizon) {
        checkArgument(base.values.length % horizon == 0, "incorrect lengths");
        this.rangeVector = new RangeVector(base);
        this.timeStamps = new long[horizon];
        this.upperTimeStamps = new long[horizon];
        this.lowerTimeStamps = new long[horizon];
    }

    public void shiftTime(int i, long shift) {
        checkArgument(i >= 0 && i < timeStamps.length, "incorrect index");
        timeStamps[i] += shift;
        // managing precision
        upperTimeStamps[i] = max(timeStamps[i], upperTimeStamps[i] + shift);
        lowerTimeStamps[i] = min(timeStamps[i], lowerTimeStamps[i] + shift);
    }

    public void scaleTime(int i, double weight) {
        checkArgument(i >= 0 && i < timeStamps.length, "incorrect index");
        checkArgument(weight > 0, " negative weight not permitted");
        timeStamps[i] = (long) (timeStamps[i] * weight);
        // managing precision
        upperTimeStamps[i] = max((long) (upperTimeStamps[i] * weight), timeStamps[i]);
        lowerTimeStamps[i] = min((long) (lowerTimeStamps[i] * weight), timeStamps[i]);
    }

}
