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

package com.amazon.randomcutforest.executor;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.sampler.Weighted;
import com.amazon.randomcutforest.state.sampler.CompactSamplerState;
import com.amazon.randomcutforest.state.store.PointStoreDoubleState;
import com.amazon.randomcutforest.state.tree.CompactRandomCutTreeState;

/**
 * The class transforms input points into the form expected by internal models,
 * and submits transformed points to individual models for updating.
 *
 * @param <P> The point representation used by model data structures.
 */
public abstract class AbstractForestUpdateExecutor<P> {

    protected final IUpdateCoordinator<P> updateCoordinator;
    protected final ArrayList<IUpdatable<P>> models;
    protected long currentIndex;

    /**
     * Create a new AbstractForestUpdateExecutor.
     * 
     * @param updateCoordinator The update coordinater that will be used to
     *                          transform points and process deleted points if
     *                          needed.
     * @param models            A list of models to update.
     */
    protected AbstractForestUpdateExecutor(IUpdateCoordinator<P> updateCoordinator, ArrayList<IUpdatable<P>> models) {
        this.updateCoordinator = updateCoordinator;
        this.models = models;
        currentIndex = 0L;
    }

    /**
     * @return the total number of times that an update has been completed.
     */
    public long getCurrentIndex() {
        return currentIndex;
    }

    /**
     * @param seen sets the "clock" of the updater needed for time dependent
     *             sampling
     */
    public void setCurrentIndex(long seen) {
        currentIndex = seen;
    }

    /**
     * Update the forest with the given point. The point is submitted to each
     * sampler in the forest. If the sampler accepts the point, the point is
     * submitted to the update method in the corresponding Random Cut Tree.
     *
     * @param point The point used to update the forest.
     */
    public void update(double[] point) {
        double[] pointCopy = cleanCopy(point);
        ++currentIndex;
        P updateInput = updateCoordinator.initUpdate(pointCopy, currentIndex);
        List<Optional<UpdateReturn<P>>> results = update(updateInput, currentIndex);
        updateCoordinator.completeUpdate(results, updateInput);
    }

    /**
     * Internal update method which submits the given input value to
     * {@link IUpdatable#update} for each model managed by this executor.
     *
     * @param updateInput  Input value that will be submitted to the update method
     *                     for each tree.
     * @param currentIndex the timestamp
     * @return a list of points that were deleted from the model as part of the
     *         update.
     */
    protected abstract List<Optional<UpdateReturn<P>>> update(P updateInput, long currentIndex);

    /**
     * Returns a clean deep copy of the point.
     *
     * Current clean-ups include changing negative zero -0.0 to positive zero 0.0.
     *
     * @param point The original data point.
     * @return a clean deep copy of the original point.
     */
    protected double[] cleanCopy(double[] point) {
        double[] pointCopy = Arrays.copyOf(point, point.length);
        for (int i = 0; i < point.length; i++) {
            if (pointCopy[i] == 0.0) {
                pointCopy[i] = 0.0;
            }
        }
        return pointCopy;
    }

    /**
     *
     * @return the weighted samples (without sequential information)
     */
    public ArrayList<List<Weighted<P>>> getWeightedSamples() {
        ArrayList<List<Weighted<P>>> result = new ArrayList<>();
        for (IUpdatable<P> t : models) {
            result.add(t.getWeightedSamples());
        }
        ;
        return result;
    }

    /**
     *
     * @return the sequential samples; if that is not stored then a dummy value is
     *         placed
     */
    public ArrayList<List<Sequential<P>>> getSequentialSamples() {
        ArrayList<List<Sequential<P>>> result = new ArrayList<>();
        for (IUpdatable<P> t : models) {
            result.add(t.getSequentialSamples());
        }
        ;
        return result;
    }

    /**
     * Gets the tree data for compact RCF
     * 
     * @return the tree data for the specifica sampler combination
     */
    public ArrayList<CompactRandomCutTreeState> getTreeData() {
        ArrayList<CompactRandomCutTreeState> result = new ArrayList<>();
        for (IUpdatable<P> t : models) {
            result.add(((SamplerPlusTree) t).getTreeData());
        }
        return result;
    }

    /**
     * gets the sampler data for compact RCF
     * 
     * @return
     */
    public ArrayList<CompactSamplerState> getCompactSamplerData() {
        ArrayList<CompactSamplerState> result = new ArrayList<>();
        for (IUpdatable<P> t : models) {
            result.add(((SamplerPlusTree) t).getCompactSamplerData());
        }
        return result;
    }

    /**
     * gets the pointstore information, currently for Integer refs and double[]
     * input
     * 
     * @return pointstore information,
     */
    public PointStoreDoubleState getPointStoredata() {
        return updateCoordinator.getPointStoreState();
    }

    /**
     * initializes the models (sampler + plus) based on samples
     * 
     * @param samplerData           data without sequence information
     * @param sequentialSamplerData data with sequence information Exactly one of
     *                              the arguments is required.
     */

    public void initializeModels(List<List<Weighted<P>>> samplerData, List<List<Sequential<P>>> sequentialSamplerData) {
        checkArgument(samplerData != null || sequentialSamplerData != null, "error, need one");
        checkArgument(!(samplerData != null && sequentialSamplerData != null), "need exactly one");
        if (sequentialSamplerData == null) {
            checkArgument(samplerData.size() == models.size(), " Mismatch ");
            int componentNum = 0;
            for (List<Weighted<P>> singleList : samplerData) {
                models.get(componentNum).initialize(singleList, null);
                ++componentNum;
            }
        } else {
            checkArgument(sequentialSamplerData.size() == models.size(), " Mismatch ");
            int componentNum = 0;
            for (List<Sequential<P>> singleList : sequentialSamplerData) {
                models.get(componentNum).initialize(null, singleList);
                ++componentNum;
            }
        }
    }

    /**
     * Initializes the compact RCF model
     * 
     * @param samplerData  data for the samplers, can have sequence information
     * @param treeDataList data for the trees; can be null, in which case the trees
     *                     would be rebuilt from the sampler information
     */

    public void initializeCompact(List<CompactSamplerState> samplerData, List<CompactRandomCutTreeState> treeDataList) {
        checkArgument(samplerData.size() == models.size(), " Mismatch ");
        if (treeDataList != null) {
            checkArgument(samplerData.size() == treeDataList.size(), " Mismatch ");
        }
        int componentNum = 0;
        for (CompactSamplerState singleSampler : samplerData) {
            if (treeDataList != null) {
                ((SamplerPlusTree) models.get(componentNum)).initializeCompact(singleSampler,
                        treeDataList.get(componentNum));
            } else {
                ((SamplerPlusTree) models.get(componentNum)).initializeCompact(singleSampler, null);
            }
            ++componentNum;
        }
    }

}
