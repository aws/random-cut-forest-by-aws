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

package com.amazon.randomcutforest;

import java.util.ArrayList;
import java.util.List;

import com.amazon.randomcutforest.sampler.Weighted;

/**
 * A class that encapsulates most of the data used in a RandomCutForest such
 * that the forest can be serialized and deserialized.
 */
public class ForestData {

    public int typeOfForest = 0;

    public long entreesSeen;

    public double lambda;

    public int numberOfTrees;

    public int sampleSize;

    public int dimensions;

    public int outputAfter;

    public boolean storeSequenceIndexesEnabled;

    public boolean compactEnabled;

    public boolean centerOfMassEnabled;

    public boolean parallelExecutionEnabled;

    public int threadPoolSize;

    public ArrayList<List<Weighted<?>>> simpleSamplerData;

    public ForestData(RandomCutForest forest) {
        this.numberOfTrees = forest.getNumberOfTrees();
        this.dimensions = forest.getDimensions();
        this.lambda = forest.getLambda();
        this.sampleSize = forest.getSampleSize();
        this.centerOfMassEnabled = forest.centerOfMassEnabled();
        this.outputAfter = forest.getOutputAfter();
        this.parallelExecutionEnabled = forest.parallelExecutionEnabled();
        this.threadPoolSize = forest.getThreadPoolSize();
        this.storeSequenceIndexesEnabled = forest.storeSequenceIndexesEnabled();
        this.simpleSamplerData = forest.getSimpleSamplerData();

        this.entreesSeen = forest.getTotalUpdates();
        this.compactEnabled = forest.isCompactEnabled();
    }
}
