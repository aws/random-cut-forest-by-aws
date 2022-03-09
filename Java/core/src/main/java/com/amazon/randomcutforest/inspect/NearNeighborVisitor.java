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

package com.amazon.randomcutforest.inspect;

import static com.amazon.randomcutforest.CommonUtils.toDoubleArray;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.returntypes.Neighbor;
import com.amazon.randomcutforest.tree.INodeView;

/**
 * A visitor that returns the leaf node in a traversal if the distance between
 * the leaf point and the query point is less than a given threshold.
 */
public class NearNeighborVisitor implements Visitor<Optional<Neighbor>> {

    private final float[] queryPoint;
    private final double distanceThreshold;
    private Neighbor neighbor;

    /**
     * Create a NearNeighborVisitor for the given query point.
     *
     * @param queryPoint        The point whose neighbors we are looking for.
     * @param distanceThreshold Leaf points whose distance from the query point is
     *                          less than this value are considered near neighbors.
     */
    public NearNeighborVisitor(float[] queryPoint, double distanceThreshold) {
        this.queryPoint = queryPoint;
        this.distanceThreshold = distanceThreshold;
        neighbor = null;
    }

    /**
     * Create a NearNeighborVisitor which always returns the leaf point in the
     * traversal. The distance threshold is set to positive infinity.
     *
     * @param queryPoint The point whose neighbors we are looking for.
     */
    public NearNeighborVisitor(float[] queryPoint) {
        this(queryPoint, Double.POSITIVE_INFINITY);
    }

    /**
     * Near neighbors are identified in the {@link #acceptLeaf} method, hence this
     * method does nothing.
     *
     * @param node        the node being visited
     * @param depthOfNode the depth of the node being visited
     */
    @Override
    public void accept(INodeView node, int depthOfNode) {
    }

    /**
     * Check to see whether the Euclidean distance between the leaf point and the
     * query point is less than the distance threshold. If it is, then this visitor
     * will return an {@link java.util.Optional} containing this leaf point
     * (converted to a {@link Neighbor} object). Otherwise, this visitor will return
     * an empty Optional.
     *
     * @param leafNode    the leaf node being visited
     * @param depthOfNode the depth of the leaf node
     */
    @Override
    public void acceptLeaf(INodeView leafNode, int depthOfNode) {
        float[] leafPoint = leafNode.getLiftedLeafPoint();
        double distanceSquared = 0.0;
        for (int i = 0; i < leafPoint.length; i++) {
            double diff = queryPoint[i] - leafPoint[i];
            distanceSquared += diff * diff;
        }

        if (Math.sqrt(distanceSquared) < distanceThreshold) {
            List<Long> sequenceIndexes = new ArrayList<>(leafNode.getSequenceIndexes().keySet());

            neighbor = new Neighbor(toDoubleArray(leafPoint), Math.sqrt(distanceSquared), sequenceIndexes);
        }
    }

    /**
     * @return an {@link Optional} containing the leaf point (converted to a
     *         {@link Neighbor} if the Euclidean distance between the leaf point and
     *         the query point is less than the distance threshold. Otherwise return
     *         an empty Optional.
     */
    @Override
    public Optional<Neighbor> getResult() {
        return Optional.ofNullable(neighbor);
    }

    @Override
    public boolean isConverged() {
        return true;
    }

}
