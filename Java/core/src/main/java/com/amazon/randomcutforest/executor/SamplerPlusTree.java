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
import java.util.function.Function;

import com.amazon.randomcutforest.IComponentModel;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.sampler.IStreamSampler;
import com.amazon.randomcutforest.tree.ITree;

public class SamplerPlusTree<P> implements IComponentModel<P> {

    private ITree<P> tree;
    private IStreamSampler<P> sampler;

    /**
     * Constructor of a pair of sampler + tree. The sampler is the driver's seat
     * because it aceepts/rejects independently of the tree and the tree has to
     * remain consistent.
     *
     * @param sampler the sampler
     * @param tree    the corresponding tree
     */
    public SamplerPlusTree(IStreamSampler<P> sampler, ITree<P> tree) {
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
     * @param point  point in consideration for updating the sampler plus tree
     * @param seqNum a time stamp that is used to generate weight in the timed
     *               sampling
     * @return the pair of (newRef,deleteRef) with potential Optional.empty()
     */

    @Override
    public Optional<UpdateReturn<P>> update(P point, long seqNum) {
        P deleteRef = null;
        Optional<Float> presampleResult = sampler.acceptSample(seqNum);
        if (presampleResult.isPresent()) {
            Optional<Sequential<P>> deletedPoint = sampler.getEvictedPoint();
            if (deletedPoint.isPresent()) {
                tree.deletePoint(deletedPoint.get());
                deleteRef = deletedPoint.get().getValue();
            }
            P newRef = (P) tree.addPoint(new Sequential(point, presampleResult.get(), seqNum));

            sampler.addSample(newRef, presampleResult.get(), seqNum);
            return Optional.ofNullable(new UpdateReturn(newRef, Optional.ofNullable(deleteRef)));
        }
        return Optional.empty();
    }

    /**
     *
     * @return the sampler
     */
    public IStreamSampler<P> getSampler() {
        return sampler;
    }

    /**
     *
     * @return the tree
     */
    public ITree<P> getTree() {
        return tree;
    }

    /**
     * traversal methods
     * 
     * @param point          to be evaluated
     * @param visitorFactory the visitor implementation that corresponds to the
     *                       function being evaluated
     * @param <R>            return type
     * @return
     */
    @Override
    public <R> R traverse(double[] point, Function<ITree<?>, Visitor<R>> visitorFactory) {
        return tree.traverse(point, visitorFactory);
    }

    /**
     *
     * @param point          to be evaluated
     * @param visitorFactory MultiVisitor, used in extrapolation
     * @param <R>            return type
     * @return
     */
    @Override
    public <R> R traverseMulti(double[] point, Function<ITree<?>, MultiVisitor<R>> visitorFactory) {
        return tree.traverseMulti(point, visitorFactory);
    }

}
