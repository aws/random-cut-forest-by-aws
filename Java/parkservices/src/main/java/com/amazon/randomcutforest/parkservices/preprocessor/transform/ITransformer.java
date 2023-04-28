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

/**
 * ThresholdedRCF allows transformers that transform the data in a streaming
 * manner; invoke RCF on the transformed data; and invert the results to the
 * original input space. A typical examples are differencing,
 * (streaming/stochastic) normalization, etc.
 *
 * This interface class spells out the operations required from such
 * transformers. Some of the operations below are specific to the existing
 * implementation and required by the mappers to produce state classes.
 */

public interface ITransformer {

    // required by the mapper; this corresponds to providing each input
    // column/attribute a weight
    // different from 1.0 -- changing these weights can alter the RCF predictions
    // significantly
    // these weights should be informed by the domain and the intent of the overall
    // computation

    double[] getWeights();

    // reverse of the above, used in mappers

    void setWeights(double[] weights);

    // used in mappers stores basic discounted averages and discounted (single step)
    // differenced average

    Deviation[] getDeviations();

    // If the RCF expects values described by values[] corresponding to the
    // correspondingInput[]
    // then what should be alternative input that would have been transformed into
    // values[]

    double[] invert(double[] values, double[] previousInput);

    // similar to invert() but applies to a forecast provided by RangeVector over an
    // input length (number of variables in a multivariate analysis) baseDimension
    // and
    // previousInput[] corresponds to the last observed values of those input.

    void invertForecastRange(RangeVector ranges, int baseDimension, double[] previousInput);

    // update the internal data structures based on the current (multivariate) input
    // inputPoint
    // previousInput[] is the corresponding values of the last observed values

    void updateDeviation(double[] inputPoint, double[] previousInput);

    // transforms inputPoint[] to RCF space, non-null values of initials[] are
    // used in normalization
    // and are specific to this implementation, internalStamp corresponds to the
    // sequence number of the
    // input and clipFactor is a parameter that clips any normalization

    double[] transformValues(int internalTimeStamp, double[] inputPoint, double[] previousInput, Deviation[] initials,
            double clipFactor);

    default double[] getShift() {
        return null;
    };

    default double[] getScale() {
        return null;
    }
}