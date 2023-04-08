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

package com.amazon.randomcutforest.store;

import java.util.List;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.summarization.ICluster;

/**
 * A view of the PointStore that forces a read only access to the store.
 */
public interface IPointStoreView<Point> {

    int getDimensions();

    int getCapacity();

    float[] getNumericVector(int index);

    float[] getInternalShingle();

    long getNextSequenceIndex();

    float[] transformToShingledPoint(Point input);

    boolean isInternalRotationEnabled();

    boolean isInternalShinglingEnabled();

    int getShingleSize();

    int[] transformIndices(int[] indexList);

    /**
     * Prints the point given the index, irrespective of the encoding of the point.
     * Used in exceptions and error messages
     * 
     * @param index index of the point in the store
     * @return a string that can be printed
     */
    String toString(int index);

    /**
     * a function that exposes an L1 clustering of the points stored in pointstore
     * 
     * @param maxAllowed              the maximum number of clusters one is
     *                                interested in
     * @param shrinkage               a parameter used in CURE algorithm that can
     *                                produce a combination of behaviors (=1
     *                                corresponds to centroid clustering, =0
     *                                resembles robust Minimum Spanning Tree)
     * @param numberOfRepresentatives another parameter used to control the
     *                                plausible (potentially non-spherical) shapes
     *                                of the clusters
     * @param separationRatio         a parameter that controls how aggressively we
     *                                go below maxAllowed -- this is often set to a
     *                                DEFAULT_SEPARATION_RATIO_FOR_MERGE
     * @param distance                a distance function
     * @param previous                a (possibly null) list of previous clusters
     *                                which can be used to seed the current clusters
     *                                to ensure some smoothness
     * @return a list of clusters
     */

    List<ICluster<float[]>> summarize(int maxAllowed, double shrinkage, int numberOfRepresentatives,
            double separationRatio, BiFunction<float[], float[], Double> distance, List<ICluster<float[]>> previous);

}
