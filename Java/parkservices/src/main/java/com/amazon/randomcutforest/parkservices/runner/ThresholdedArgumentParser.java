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

package com.amazon.randomcutforest.parkservices.runner;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import com.amazon.randomcutforest.runner.ArgumentParser;

/**
 * An argument parser for ThresholdedRandomCutForests.
 * 
 * A ThresholdedRandomCutForest takes all of the same initialization arguments
 * as a RandomCutForest plus several additional thresholding-specific
 * parameters. This ArgumentParser includes several of these additional
 * parameters.
 */
public class ThresholdedArgumentParser extends ArgumentParser {
    private final DoubleArgument anomalyRate;
    private final DoubleArgument horizon;
    private final DoubleArgument lowerThreshold;
    private final DoubleArgument zFactor;

    public ThresholdedArgumentParser(String runnerClass, String runnerDescription) {
        super(runnerClass, runnerDescription);

        anomalyRate = new DoubleArgument(null, "--anomaly-rate",
                "Approximate expected anomaly rate. Controls anomaly threshold decay rate.", 0.01);
        addArgument(anomalyRate);

        horizon = new DoubleArgument(null, "--horizon",
                "Mixture factor between using scores and score differences for thresholding. Value of 1.0 means the thresholder only uses score values. Value of 0.0 means the thresholder only uses score differences.",
                0.5, n -> checkArgument(n >= 0.0 && n <= 1.0, "Horizon should be between 0.0 and 1.0"));
        addArgument(horizon);

        lowerThreshold = new DoubleArgument(null, "--lower-threshold",
                "Anomaly score threshold for marking a potential anomaly. Affects thresholder sensitivity.", 1.0,
                n -> checkArgument(n > 0.0, "Lower threshold must be greater than 0.0"));
        addArgument(lowerThreshold);

        zFactor = new DoubleArgument(null, "--zfactor",
                "Z-score threshold for marking a potential anomaly. Affects thresholder sensitivity.", 2.5,
                n -> checkArgument(n > 0.0, "Zfactor must be greater than 0.0"));
        addArgument(zFactor);
    }

    /**
     * @return the user-specified value of the anomaly rate
     */
    public double getAnomalyRate() {
        return anomalyRate.getValue();
    }

    /**
     * @return the user-specified value of the horizon
     */
    public double getHorizon() {
        return horizon.getValue();
    }

    /**
     * @return the user-specified value of the lower threshold
     */
    public double getLowerThreshold() {
        return lowerThreshold.getValue();
    }

    /**
     * @return the user-specified value of the zFactor
     */
    public double getZfactor() {
        return zFactor.getValue();
    }
}
