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

package com.amazon.randomcutforest.serialize.json.v1;

/**
 * Serialized RCF for internal use only.
 */
public class V1SerializedRandomCutForest {

    public Random getRng() {
        return rng;
    }

    public void setRng(Random rng) {
        this.rng = rng;
    }

    public int getDimensions() {
        return dimensions;
    }

    public void setDimensions(int dimensions) {
        this.dimensions = dimensions;
    }

    public int getSampleSize() {
        return sampleSize;
    }

    public void setSampleSize(int sampleSize) {
        this.sampleSize = sampleSize;
    }

    public int getOutputAfter() {
        return outputAfter;
    }

    public void setOutputAfter(int outputAfter) {
        this.outputAfter = outputAfter;
    }

    public int getNumberOfTrees() {
        return numberOfTrees;
    }

    public void setNumberOfTrees(int numberOfTrees) {
        this.numberOfTrees = numberOfTrees;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public boolean isStoreSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    public void setStoreSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
        this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
    }

    public boolean isCenterOfMassEnabled() {
        return centerOfMassEnabled;
    }

    public void setCenterOfMassEnabled(boolean centerOfMassEnabled) {
        this.centerOfMassEnabled = centerOfMassEnabled;
    }

    public boolean isParallelExecutionEnabled() {
        return parallelExecutionEnabled;
    }

    public void setParallelExecutionEnabled(boolean parallelExecutionEnabled) {
        this.parallelExecutionEnabled = parallelExecutionEnabled;
    }

    public int getThreadPoolSize() {
        return threadPoolSize;
    }

    public void setThreadPoolSize(int threadPoolSize) {
        this.threadPoolSize = threadPoolSize;
    }

    public Executor getExecutor() {
        return executor;
    }

    public void setExecutor(Executor executor) {
        this.executor = executor;
    }

    private static class Random {
    }

    private static class Tree {
        private boolean storeSequenceIndexesEnabled;
        private boolean centerOfMassEnabled;
        private Random random;

        public boolean isStoreSequenceIndexesEnabled() {
            return storeSequenceIndexesEnabled;
        }

        public void setStoreSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
        }

        public boolean isCenterOfMassEnabled() {
            return centerOfMassEnabled;
        }

        public void setCenterOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
        }

        public Random getRandom() {
            return random;
        }

        public void setRandom(Random random) {
            this.random = random;
        }
    }

    static class WeightedSamples {
        private double[] point;
        private double weight;
        private long sequenceIndex;

        public double[] getPoint() {
            return point;
        }

        public void setPoint(double[] point) {
            this.point = point;
        }

        public double getWeight() {
            return weight;
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }

        public long getSequenceIndex() {
            return sequenceIndex;
        }

        public void setSequenceIndex(long sequenceIndex) {
            this.sequenceIndex = sequenceIndex;
        }
    }

    static class Sampler {
        private WeightedSamples[] weightedSamples;
        private int sampleSize;
        private double lambda;
        private Random random;
        private long entriesSeen;

        public WeightedSamples[] getWeightedSamples() {
            return weightedSamples;
        }

        public void setWeightedSamples(WeightedSamples[] weightedSamples) {
            this.weightedSamples = weightedSamples;
        }

        public int getSampleSize() {
            return sampleSize;
        }

        public void setSampleSize(int sampleSize) {
            this.sampleSize = sampleSize;
        }

        public double getLambda() {
            return lambda;
        }

        public void setLambda(double lambda) {
            this.lambda = lambda;
        }

        public Random getRandom() {
            return random;
        }

        public void setRandom(Random random) {
            this.random = random;
        }

        public long getEntriesSeen() {
            return entriesSeen;
        }

        public void setEntriesSeen(long entriesSeen) {
            this.entriesSeen = entriesSeen;
        }
    }

    static class TreeUpdater {
        public Sampler getSampler() {
            return sampler;
        }

        public void setSampler(Sampler sampler) {
            this.sampler = sampler;
        }

        public Tree getTree() {
            return tree;
        }

        public void setTree(Tree tree) {
            this.tree = tree;
        }

        private Sampler sampler;
        private Tree tree;
    }

    static class Exec {
        private TreeUpdater[] treeUpdaters;
        private long totalUpdates;
        private int threadPoolSize;

        public TreeUpdater[] getTreeUpdaters() {
            return treeUpdaters;
        }

        public void setTreeUpdaters(TreeUpdater[] treeUpdaters) {
            this.treeUpdaters = treeUpdaters;
        }

        public long getTotalUpdates() {
            return totalUpdates;
        }

        public void setTotalUpdates(long totalUpdates) {
            this.totalUpdates = totalUpdates;
        }

        public int getThreadPoolSize() {
            return threadPoolSize;
        }

        public void setThreadPoolSize(int threadPoolSize) {
            this.threadPoolSize = threadPoolSize;
        }
    }

    static class Executor {
        private String executor_type;
        private Exec executor;

        public String getExecutor_type() {
            return executor_type;
        }

        public void setExecutor_type(String executor_type) {
            this.executor_type = executor_type;
        }

        public Exec getExecutor() {
            return executor;
        }

        public void setExecutor(Exec executor) {
            this.executor = executor;
        }
    }

    private Random rng;
    private int dimensions;
    private int sampleSize;
    private int outputAfter;
    private int numberOfTrees;
    private double lambda;
    private boolean storeSequenceIndexesEnabled;
    private boolean centerOfMassEnabled;
    private boolean parallelExecutionEnabled;
    private int threadPoolSize;
    private Executor executor;
}
