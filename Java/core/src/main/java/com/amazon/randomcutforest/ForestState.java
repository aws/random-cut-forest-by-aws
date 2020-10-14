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

import com.amazon.randomcutforest.executor.Sequential;
import com.amazon.randomcutforest.sampler.CompactSamplerData;
import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.store.PointStoreDoubleData;
import com.amazon.randomcutforest.tree.TreeData;

import java.util.ArrayList;
import java.util.List;

/**
 * A class that encapsulates most of the data used in a RandomCutForest such
 * that the forest can be serialized and deserialized.
 */
public class ForestState {

    public String typeOfForest = null;

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

}
