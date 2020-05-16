/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package com.amazon.randomcutforest.returntypes;

/**
 * DensityOutput extends InterpolationMeasure with methods for computing density
 * estimates.
 */
public class DensityOutput extends InterpolationMeasure {

    /**
     * Default scaling factor (q) to use in the getDensity method.
     */
    public static final double DEFAULT_SUM_OF_POINTS_SCALING_FACTOR = 0.001;

    /**
     * Create a new DensityOutput object with the given number of spatial
     * dimensions. Note that the number of half-dimensions will be 2 * dimensions.
     *
     * @param dimensions The number of spatial dimensions.
     * @param sampleSize The samplesize of each tree in forest, which may be used
     *                   for normalization.
     */
    public DensityOutput(int dimensions, int sampleSize) {
        super(dimensions, sampleSize);
    }

    /**
     * A copy constructor that creates a deep copy of the base DensityOutput.
     *
     * @param base An InterpolationMeasure instance that we want to copy.
     */
    public DensityOutput(InterpolationMeasure base) {
        super(base);
    }

    /**
     * Compute a scalar density estimate. The scaling factor q is multiplied by the
     * sum of points measure and added to the denominator in the density expression
     * to prevent divide-by-0 errors.
     *
     * @param q                 A scaling factor applied to the sum of points in the
     *                          measure.
     * @param manifoldDimension The number of dimensions of the submanifold on which
     *                          we are estimating a density.
     * @return a scalar density estimate.
     */
    public double getDensity(double q, int manifoldDimension) {
        double sumOfPts = measure.getHighLowSum() / sampleSize;

        if (sumOfPts <= 0.0) {
            return 0.0;
        }

        double sumOfFactors = 0;

        for (int i = 0; i < dimensions; i++) {
            double t = probMass.getHighLowSum(i) > 0 ? distances.getHighLowSum(i) / probMass.getHighLowSum(i) : 0;
            if (t > 0) {
                t = Math.exp(Math.log(t) * manifoldDimension) * probMass.getHighLowSum(i);
            }
            sumOfFactors += t;
        }

        return sumOfPts / (q * sumOfPts + sumOfFactors);
    }

    /**
     * Compute a scalar density estimate. This method uses the default scaling
     * factor and the full number of dimensions.
     *
     * @return a scalar density estimate.
     */
    public double getDensity() {
        return getDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, dimensions);
    }

    /**
     * Compute a directional density estimate. The scaling factor q is multiplied by
     * the sum of points measure and added to the denominator in the density
     * expression to prevent divide-by-0 errors.
     *
     * @param q                 A scaling factor applied to the sum of points in the
     *                          measure.
     * @param manifoldDimension The number of dimensions of the submanifold on which
     *                          we are estimating a density.
     * @return a directional density estimate.
     */
    public DiVector getDirectionalDensity(double q, int manifoldDimension) {
        double density = getDensity(q, manifoldDimension);
        double sumOfPts = measure.getHighLowSum(); // normalization not performed since this would be used in a ratio
        DiVector factors = new DiVector(super.getDimensions());

        if (sumOfPts > 0) {
            factors = measure.scale(density / sumOfPts);
        }

        return factors;
    }

    /**
     * Compute a directional density estimate. This method uses the default scaling
     * factor and the full number of dimensions.
     *
     * @return a scalar density estimate.
     */
    public DiVector getDirectionalDensity() {
        return getDirectionalDensity(DEFAULT_SUM_OF_POINTS_SCALING_FACTOR, dimensions);
    }
}
