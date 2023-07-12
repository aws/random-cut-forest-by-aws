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

import java.util.Arrays;

import lombok.Getter;
import lombok.Setter;

import com.amazon.randomcutforest.parkservices.returntypes.TimedRangeVector;
import com.amazon.randomcutforest.returntypes.DiVector;
import com.amazon.randomcutforest.returntypes.RangeVector;

@Getter
@Setter
public class ForecastDescriptor extends AnomalyDescriptor {

    // all the following objects will be of length (forecast horizon x the number of
    // input variables)

    /**
     * basic forecast field, with the time information to be used for TIME_AUGMENTED
     * mode in the future
     */
    TimedRangeVector timedForecast;

    /**
     * the distribution of errors -- for an algorithm that self-calibrates, this
     * information has to be computed exposing the error can be of use for the user
     * to audit the results. The distributions will use interpolations and will not
     * adhere to specific quantile values -- thereby allowing for better
     * generalization.
     */
    RangeVector observedErrorDistribution;

    /**
     * typically RMSE is a single vector -- however unlike standard literature, we
     * would not be limited to zero mean time series; in fact converting a time
     * series to a zero mean series in an online manner is already challenging.
     * Moreover, it is often the case that errors have a typical distribution skew;
     * in the current library we have partitioned many of the explainabilty aspects
     * (e.g., attribution in anomaly detection, directionality in density
     * estimation, etc.) based on high/low; when the actual value being observed is
     * correspondingly higher/lower than some (possibly implicit) baseline. We split
     * the same for error.
     */
    DiVector errorRMSE;

    /**
     * mean error corresponding to the forecast horizon x the number of input
     * variables This is not used in the current intervalPrecision -- we use the
     * median value from the error distribution.
     */
    float[] errorMean;

    /**
     * in the forecast horizon x the number of input variables this corresponds to
     * the fraction of variables \predicted correctly over the error horizon. A
     * value of 1.0 is terrific.
     */
    float[] intervalPrecision;

    public ForecastDescriptor(double[] input, long inputTimeStamp, int horizon) {
        super(input, inputTimeStamp);
        int forecastLength = input.length * horizon;
        this.timedForecast = new TimedRangeVector(forecastLength, horizon);
        this.observedErrorDistribution = new RangeVector(forecastLength);
        Arrays.fill(this.observedErrorDistribution.lower, -Float.MAX_VALUE);
        Arrays.fill(this.observedErrorDistribution.upper, Float.MAX_VALUE);
        this.errorMean = new float[forecastLength];
        this.errorRMSE = new DiVector(forecastLength);
        this.intervalPrecision = new float[forecastLength];
    }

    void setObservedErrorDistribution(RangeVector base) {
        checkArgument(base.values.length == this.observedErrorDistribution.values.length, " incorrect length");
        System.arraycopy(base.values, 0, this.observedErrorDistribution.values, 0, base.values.length);
        System.arraycopy(base.upper, 0, this.observedErrorDistribution.upper, 0, base.upper.length);
        System.arraycopy(base.lower, 0, this.observedErrorDistribution.lower, 0, base.lower.length);
    }

    void setIntervalPrecision(float[] calibration) {
        System.arraycopy(calibration, 0, this.intervalPrecision, 0, calibration.length);
    }

    @Deprecated
    float[] getCalibration() {
        return Arrays.copyOf(intervalPrecision, intervalPrecision.length);
    }

    void setErrorMean(float[] errorMean) {
        System.arraycopy(errorMean, 0, this.errorMean, 0, errorMean.length);
    }

    void setErrorRMSE(DiVector errorRMSE) {
        checkArgument(this.errorRMSE.getDimensions() == errorRMSE.getDimensions(), " incorrect input");
        System.arraycopy(errorRMSE.high, 0, this.errorRMSE.high, 0, errorRMSE.high.length);
        System.arraycopy(errorRMSE.low, 0, this.errorRMSE.low, 0, errorRMSE.low.length);
    }
}
