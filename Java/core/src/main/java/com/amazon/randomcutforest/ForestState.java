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

import com.amazon.randomcutforest.executor.Sequential;
import com.amazon.randomcutforest.sampler.CompactSamplerData;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.store.PointStoreDoubleData;
import com.amazon.randomcutforest.tree.TreeData;

/**
 * A class that encapsulates most of the data used in a RandomCutForest such
 * that the forest can be serialized and deserialized.
 */
public class ForestState {

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

    public boolean saveTreeData;

    public PointStoreDoubleData pointStoreDoubleData;

    public ArrayList<CompactSamplerData> compactSamplerData;

    public ArrayList<List<Sequential<double[]>>> sequentialSamplerData;

    public ArrayList<List<Weighted<double[]>>> smallSamplerData;

    public ArrayList<TreeData> treeData;

    public ForestState(RandomCutForest forest) {
        this.numberOfTrees = forest.getNumberOfTrees();
        this.dimensions = forest.getDimensions();
        this.lambda = forest.getLambda();
        this.sampleSize = forest.getSampleSize();
        this.centerOfMassEnabled = forest.centerOfMassEnabled();
        this.outputAfter = forest.getOutputAfter();
        this.parallelExecutionEnabled = forest.parallelExecutionEnabled();
        this.threadPoolSize = forest.getThreadPoolSize();
        this.storeSequenceIndexesEnabled = forest.storeSequenceIndexesEnabled();
        this.entreesSeen = forest.getTotalUpdates();
        this.compactEnabled = forest.compactEnabled();
        this.saveTreeData = forest.saveTreeData();

        if (!compactEnabled) {
            /**
             * In this case there is no pointstore and we onle have a basic serialization
             * where the samples are stored and the trees are rebuilt from the samples.
             */
            this.pointStoreDoubleData = null;
            if (storeSequenceIndexesEnabled) {
                this.sequentialSamplerData = forest.updateExecutor.getSequentialSamples();
                this.smallSamplerData = null;
            } else {
                this.smallSamplerData = forest.updateExecutor.getWeightedSamples();
                this.sequentialSamplerData = null;
            }
            this.compactSamplerData = null;
        } else {
            this.pointStoreDoubleData = forest.updateExecutor.getPointStoredata();
            if (this.saveTreeData) {
                this.treeData = forest.updateExecutor.getTreeData();
            } else {
                this.treeData = null;
            }
            this.compactSamplerData = forest.updateExecutor.getCompactSamplerData();
            this.sequentialSamplerData = null;
            this.smallSamplerData = null;
        }
        this.entreesSeen = forest.getTotalUpdates();
    }
}
