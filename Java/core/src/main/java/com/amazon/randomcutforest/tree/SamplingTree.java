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

package com.amazon.randomcutforest.tree;

import static com.amazon.randomcutforest.CommonUtils.checkNotNull;

import java.util.Optional;

import com.amazon.randomcutforest.IUpdatable;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Sequential;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.sampler.IStreamSampler;

public class SamplingTree<P> implements ITree<Sequential<P>>, IUpdatable<Sequential<P>> {

    private final ITree<Sequential<P>> tree;
    private final IStreamSampler<P> sampler;

    public SamplingTree(IStreamSampler<P> sampler, ITree<Sequential<P>> tree) {
        checkNotNull(sampler, "sampler must not be null");
        checkNotNull(tree, "tree must not be null");
        this.sampler = sampler;
        this.tree = tree;
    }

    @Override
    public Optional<Sequential<P>> update(Sequential<P> point) {
        if (sampler.sample(point)) {
            Optional<Sequential<P>> evictedPoint = sampler.getEvictedPoint();
            evictedPoint.ifPresent(this::deletePoint);
            addPoint(point);
            return evictedPoint;
        }
        return Optional.empty();
    }

    @Override
    public int getMass() {
        return tree.getMass();
    }

    @Override
    public void addPoint(Sequential<P> point) {
        tree.addPoint(point);
    }

    @Override
    public void deletePoint(Sequential<P> point) {
        tree.deletePoint(point);
    }

    @Override
    public <R> R traverse(double[] point, Visitor<R> visitor) {
        return tree.traverse(point, visitor);
    }

    @Override
    public <R> R traverseMulti(double[] point, MultiVisitor<R> visitor) {
        return tree.traverseMulti(point, visitor);
    }
}
