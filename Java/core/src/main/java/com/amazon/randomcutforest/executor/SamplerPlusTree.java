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

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Optional;
import java.util.function.BiFunction;

import lombok.Getter;

import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.IMultiVisitorFactory;
import com.amazon.randomcutforest.IVisitorFactory;
import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.sampler.ISampled;
import com.amazon.randomcutforest.sampler.IStreamSampler;
import com.amazon.randomcutforest.tree.ITree;
import com.amazon.randomcutforest.tree.NewRandomCutTree;

/**
 * A SamplerPlusTree corresponds to a combination of sampler and tree where the
 * information is passed via P and the tree can seek explicit point information
 * of type Q
 *
 * @param <P> The internal point representation expected by the component models
 *            in this list.
 * @param <Q> The explicit data type of points being passed
 */
@Getter
public class SamplerPlusTree<P, Q> implements IComponentModel<P, Q> {

    private ITree<P, Q> tree;
    private IStreamSampler<P> sampler;

    /**
     * Constructor of a pair of sampler + tree. The sampler is the driver's seat
     * because it aceepts/rejects independently of the tree and the tree has to
     * remain consistent.
     *
     * @param sampler the sampler
     * @param tree    the corresponding tree
     */
    public SamplerPlusTree(IStreamSampler<P> sampler, ITree<P, Q> tree) {
        checkNotNull(sampler, "sampler must not be null");
        checkNotNull(tree, "tree must not be null");
        this.sampler = sampler;
        this.tree = tree;
    }

    /**
     * This is main function that maintains the coordination between the sampler and
     * the tree. The sampler proposes acceptance (by setting the weight in
     * queueEntry) and in that case the evictedPoint is set. That evictedPoint is
     * removed from the tree and in that case its reference deleteRef of type T is
     * noted. The point is then added to the tree where the tree may propose a new
     * reference newRef because the point is already present in the tree. The
     * sampler entry is modified and added to the sampler. The pair of the newRef
     * and deleteRef are returned for plausible bookkeeping in update executors.
     *
     * @param point         point in consideration for updating the sampler plus
     *                      tree
     * @param sequenceIndex a time stamp that is used to generate weight in the
     *                      timed sampling
     * @return the pair of (newRef,deleteRef) with potential Optional.empty()
     */

    @Override
    public UpdateResult<P> update(P point, long sequenceIndex) {
        P deleteRef = null;
        if (sampler.acceptPoint(sequenceIndex)) {
            Optional<ISampled<P>> deletedPoint = sampler.getEvictedPoint();
            if (deletedPoint.isPresent()) {
                ISampled<P> p = deletedPoint.get();
                deleteRef = p.getValue();
                tree.deletePoint(deleteRef, p.getSequenceIndex());
            }

            // the tree may choose to return a reference to an existing point
            // whose value is equal to `point`
            P addedPoint = tree.addPoint(point, sequenceIndex);
            sampler.addPoint(addedPoint);
            return UpdateResult.<P>builder().addedPoint(addedPoint).deletedPoint(deleteRef).build();
        }
        return UpdateResult.noop();
    }

    @Override
    public <R> R traverse(double[] point, IVisitorFactory<R> visitorFactory) {
        return tree.traverse(point, visitorFactory);
    }

    @Override
    public <R> R traverseMulti(double[] point, IMultiVisitorFactory<R> visitorFactory) {
        return tree.traverseMulti(point, visitorFactory);
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            tree.setConfig(name, value, clazz);
        } else if (Config.TIME_DECAY.equals(name)) {
            sampler.setConfig(name, value, clazz);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public <T> T getConfig(String name, Class<T> clazz) {
        checkNotNull(clazz, "clazz must not be null");
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            return tree.getConfig(name, clazz);
        } else if (Config.TIME_DECAY.equals(name)) {
            return sampler.getConfig(name, clazz);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public boolean isOutputReady() {
        return tree.isOutputReady();
    }

    // the following is for testing RCF3.0 alongside RCF2.0 and will be merged into
    // getAnomalyScore()
    public double scalarScore(double[] point, int ignoreMass, BiFunction<Double, Double, Double> scoreSeen,
            BiFunction<Double, Double, Double> scoreUnseen, BiFunction<Double, Double, Double> damp,
            BiFunction<Double, Double, Double> normalizer) {
        return ((NewRandomCutTree) tree).scalarScore(point, ignoreMass, scoreSeen, scoreUnseen, damp, normalizer);
    }

}
