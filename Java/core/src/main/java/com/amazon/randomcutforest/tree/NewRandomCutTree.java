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

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static com.amazon.randomcutforest.CommonUtils.checkState;

import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.Stack;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.IMultiVisitorFactory;
import com.amazon.randomcutforest.IVisitorFactory;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.config.Config;
import com.amazon.randomcutforest.store.IPointStoreView;

/**
 * A Compact Random Cut Tree is a tree data structure whose leaves represent
 * points inserted into the tree and whose interior nodes represent regions of
 * space defined by Bounding Boxes and Cuts. New nodes and leaves are added to
 * the tree by making random cuts.
 *
 * The offsets are encoded as follows: an offset greater or equal maxSize
 * corresponds to a leaf node of offset (offset - maxSize) otherwise the offset
 * corresponds to an internal node
 *
 * The main use of this class is to be updated with points sampled from a
 * stream, and to define traversal methods. Users can then implement a
 * {@link Visitor} which can be submitted to a traversal method in order to
 * compute a statistic from the tree.
 */
public class NewRandomCutTree implements ITree<Integer, float[]> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */

    private Random testRandom;
    private long randomSeed;
    protected int root;
    protected IPointStoreView<float[]> pointStoreView;
    protected int treeMass;
    protected int capacity;
    protected AbstractNodeStore nodeStore;
    protected double nodeCacheFraction = 1.0;
    protected int outputAfter;

    protected NewRandomCutTree(Builder<?> builder) {
        pointStoreView = builder.pointStoreView;
        capacity = builder.capacity;
        randomSeed = builder.randomSeed;
        outputAfter = builder.outputAfter.orElse(capacity / 4);
        if (capacity <= 256 && pointStoreView.getDimensions() <= 256) {
            nodeStore = new NodeStoreSmall(capacity, pointStoreView.getDimensions(), builder.nodeCacheFraction);
        } else if (capacity <= Character.MAX_VALUE && pointStoreView.getDimensions() <= Character.MAX_VALUE) {
            nodeStore = new NodeStoreMedium(capacity, pointStoreView.getDimensions(), builder.nodeCacheFraction);
        } else {
            nodeStore = new NodeStoreLarge(capacity, pointStoreView.getDimensions(), builder.nodeCacheFraction);
        }
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    String.format("Setting '%s' must be a double value", name));
            setNodeCacheFraction((Double) value);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    @Override
    public <T> T getConfig(String name, Class<T> clazz) {
        checkNotNull(clazz, "clazz must not be null");
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(clazz.isAssignableFrom(Double.class),
                    String.format("Setting '%s' must be a double value", name));
            return clazz.cast(nodeCacheFraction);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    // dynamically change the fraction of the new nodes which caches their bounding
    // boxes
    // 0 would mean less space usage, but slower throughput
    // 1 would imply larger space but better throughput
    public void setNodeCacheFraction(double fraction) {
        checkArgument(0 <= fraction && fraction <= 1, "incorrect parameter");
        nodeCacheFraction = fraction;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param factor A random cut
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    protected Cut randomCut(double factor, float[] point, BoundingBoxFloat box) {
        double range = 0.0;

        for (int i = 0; i < point.length; i++) {
            float minValue = (float) box.getMinValue(i);
            float maxValue = (float) box.getMaxValue(i);
            if (point[i] < minValue) {
                minValue = point[i];
            } else if (point[i] > maxValue) {
                maxValue = point[i];
            }
            range += maxValue - minValue;
        }

        double breakPoint = factor * range;

        for (int i = 0; i < box.getDimensions(); i++) {
            float minValue = (float) box.getMinValue(i);
            float maxValue = (float) box.getMaxValue(i);
            if (point[i] < minValue) {
                minValue = point[i];
            } else if (point[i] > maxValue) {
                maxValue = point[i];
            }
            double gap = maxValue - minValue;
            if (breakPoint <= gap) {
                float cutValue = (float) (minValue + breakPoint);

                // Random cuts have to take a value in the half-open interval [minValue,
                // maxValue) to ensure that a
                // Node has a valid left child and right child.
                if ((cutValue >= maxValue) && (minValue < maxValue)) {
                    cutValue = Math.nextAfter((float) maxValue, minValue);
                }

                return new Cut(i, cutValue);
            }
            breakPoint -= gap;
        }

        throw new IllegalStateException("The break point did not lie inside the expected range");
    }

    public Integer addPoint(Integer pointIndex, long sequenceIndex) {

        if (root == 0) {
            root = nodeStore.addLeaf(0, pointIndex, 1);
            treeMass = 1;
            return pointIndex;
        } else {
            float[] point = pointStoreView.get(pointIndex);
            Stack<int[]> leafPath = nodeStore.getPath(root, point, false);
            int[] first = leafPath.pop();
            int leafNode = first[0];
            int leafSavedSibling = first[1];
            int sibling = leafSavedSibling;
            int leafPointIndex = nodeStore.getPointIndex(leafNode);
            float[] oldPoint = pointStoreView.get(leafPointIndex);
            int savedParent = 0;
            if (leafPath.size() != 0) {
                savedParent = leafPath.lastElement()[0];
            }
            Stack<int[]> parentPath = new Stack<>();
            treeMass += 1;
            if (Arrays.equals(point, oldPoint)) {
                nodeStore.increaseLeafMass(leafNode);
                checkArgument(!nodeStore.freeNodeManager.isEmpty(), "incorrect/impossible state");
                nodeStore.manageAncestorsAdd(leafPath, point, pointStoreView);
                return leafPointIndex;
            } else {
                int node = leafNode;
                int savedNode = node;
                int parent = savedParent;
                float savedCutValue = (float) 0.0;
                BoundingBoxFloat currentBox = new BoundingBoxFloat(oldPoint, oldPoint);
                BoundingBoxFloat savedBox = currentBox.newCopy();
                int savedDim = Integer.MAX_VALUE;
                Random rng = new Random(randomSeed);
                randomSeed = rng.nextLong();
                while (true) {
                    double factor = rng.nextDouble();
                    Cut cut = randomCut(factor, point, currentBox);
                    int dim = cut.getDimension();
                    float value = (float) cut.getValue();

                    boolean separation = ((point[dim] <= value && value < currentBox.getMinValue(dim)
                            || point[dim] > value && value >= currentBox.getMaxValue(dim)));

                    if (separation) {
                        savedCutValue = value;
                        savedDim = dim;
                        savedParent = parent;
                        savedNode = node;
                        savedBox = currentBox.newCopy();
                        parentPath.clear();
                    } else {
                        parentPath.push(new int[] { node, sibling });
                    }

                    if (savedDim == Integer.MAX_VALUE) {
                        randomCut(factor, point, currentBox);
                        throw new IllegalStateException(" cut failed ");
                    }
                    if (currentBox.contains(point) || parent == 0) {
                        break;
                    } else {
                        nodeStore.growNodeBox(currentBox, pointStoreView, parent, sibling);
                        int[] next = leafPath.pop();
                        node = next[0];
                        sibling = next[1];
                        if (leafPath.size() != 0) {
                            parent = leafPath.lastElement()[0];
                        } else {
                            parent = 0;
                        }
                    }
                }
                int newLeafNode = nodeStore.addLeaf(0, pointIndex, 1);
                int newMass = nodeStore.getMass(savedNode) + 1;
                int mergedNode = (point[savedDim] <= savedCutValue)
                        ? nodeStore.addNode(savedParent, newLeafNode, savedNode, savedDim, savedCutValue, newMass)
                        : nodeStore.addNode(savedParent, savedNode, newLeafNode, savedDim, savedCutValue, newMass);

                savedBox.addPoint(point);
                nodeStore.addBox(mergedNode, savedBox);

                if (savedParent != 0) {
                    // add the new node
                    nodeStore.spliceEdge(savedParent, savedNode, mergedNode);
                    while (!parentPath.isEmpty()) {
                        leafPath.push(parentPath.pop());
                    }
                    assert (leafPath.lastElement()[0] == savedParent);
                    nodeStore.manageAncestorsAdd(leafPath, point, pointStoreView);
                } else {
                    root = mergedNode;
                }
            }
            return pointIndex;
        }
    }

    public Integer deletePoint(Integer pointIndex, long sequenceIndex) {
        if (root == 0) {
            throw new IllegalStateException(" deleting from an empty tree");
        }

        treeMass = treeMass - 1;
        float[] point = pointStoreView.get(pointIndex);
        // System.out.println(" new delete " + Arrays.toString(point));
        Stack<int[]> leafPath = nodeStore.getPath(root, point, false);
        int[] first = leafPath.pop();
        int leafSavedSibling = first[1];
        int leafNode = first[0];
        int leafPointIndex = nodeStore.getPointIndex(leafNode);
        float[] oldPoint = pointStoreView.get(leafPointIndex);

        if (!Arrays.equals(point, oldPoint)) {
            throw new IllegalStateException(" deleting wrong node " + leafPointIndex + " instead of " + pointIndex);
        }

        if (nodeStore.decreaseLeafMass(leafNode) == 0) {
            if (leafPath.size() == 0) {
                root = 0;
            } else {
                int parent = leafPath.pop()[0];
                int grandParent = 0;
                if (leafPath.size() != 0) {
                    grandParent = leafPath.lastElement()[0];
                }

                if (grandParent == 0) {
                    root = leafSavedSibling;
                    nodeStore.setRoot(root);
                } else {
                    nodeStore.replaceParentBySibling(grandParent, parent, leafNode);
                    assert (leafPath.lastElement()[0] == grandParent);
                    nodeStore.manageAncestorsDelete(leafPath, point, pointStoreView);
                }
                nodeStore.deleteInternalNode(parent);
            }
        }
        return leafPointIndex;
    }

    public double dynamicScore(double[] point, int ignoreMass, BiFunction<Double, Double, Double> scoreSeen,
            BiFunction<Double, Double, Double> scoreUnseen, BiFunction<Double, Double, Double> damp,
            BiFunction<Double, Double, Double> normalizer) {
        Function<Double, Double> treeDamp = x -> damp.apply(x, treeMass * 1.0);
        return normalizer.apply(
                nodeStore.dynamicScore(root, ignoreMass, point, pointStoreView, scoreSeen, scoreUnseen, treeDamp),
                treeMass * 1.0);
    }

    /**
     * Starting from the root, traverse the canonical path to a leaf node and visit
     * the nodes along the path. The canonical path is determined by the input
     * point: at each interior node, we select the child node by comparing the
     * node's {@link Cut} to the corresponding coordinate value in the input point.
     * The method recursively traverses to the leaf node first and then invokes the
     * visitor on each node in reverse order. That is, if the path to the leaf node
     * determined by the input point is root, node1, node2, ..., node(N-1), nodeN,
     * leaf; then we will first invoke visitor::acceptLeaf on the leaf node, and
     * then we will invoke visitor::accept on the remaining nodes in the following
     * order: nodeN, node(N-1), ..., node2, node1, and root.
     *
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    @Override
    public <R> R traverse(double[] point, IVisitorFactory<R> visitorFactory) {
        checkState(root != 0, "this tree doesn't contain any nodes");
        Visitor<R> visitor = visitorFactory.newVisitor(this, point);
        nodeStore.traversePathToLeafAndVisitNodes(projectToTree(point), visitor, root, pointStoreView,
                this::liftFromTree);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    /**
     * This is a traversal method which follows the standard traversal path (defined
     * in {@link #traverse(double[], IVisitorFactory)}) but at Node in checks to see
     * whether the visitor should split. If a split is triggered, then independent
     * copies of the visitor are sent down each branch of the tree and then merged
     * before propagating the result.
     *
     * @param point          A point which determines the traversal path from the
     *                       root to a leaf node.
     * @param visitorFactory A visitor that will be invoked for each node on the
     *                       path.
     * @param <R>            The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */

    @Override
    public <R> R traverseMulti(double[] point, IMultiVisitorFactory<R> visitorFactory) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        checkState(root != 0, "this tree doesn't contain any nodes");
        MultiVisitor<R> visitor = visitorFactory.newVisitor(this, point);
        nodeStore.traverseTreeMulti(projectToTree(point), visitor, root, pointStoreView, this::liftFromTree);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    /**
     *
     * @return the mass of the tree
     */
    @Override
    public int getMass() {
        return treeMass;
    }

    /**
     *
     * @return the root of the tree
     */

    public Integer getRoot() {
        return (int) root;
    }

    /**
     * returns the number of samples that needs to be seen before returning
     * meaningful inference
     */
    public int getOutputAfter() {
        return outputAfter;
    }

    @Override
    public boolean isOutputReady() {
        return getMass() >= outputAfter;
    }

    public double[] projectToTree(double[] point) {
        return Arrays.copyOf(point, point.length);
    }

    public double[] liftFromTree(double[] result) {
        return Arrays.copyOf(result, result.length);
    }

    public int[] projectMissingIndices(int[] list) {
        return Arrays.copyOf(list, list.length);
    }

    public long getRandomSeed() {
        return randomSeed;
    }

    public static class Builder<T extends Builder<T>> {
        protected double nodeCacheFraction = RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected long randomSeed = new Random().nextLong();
        protected Random random = null;
        protected int capacity = RandomCutForest.DEFAULT_SAMPLE_SIZE;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected int dimension;
        protected IPointStoreView<float[]> pointStoreView;

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T nodeCacheFraction(double boundingBoxCacheFraction) {
            this.nodeCacheFraction = boundingBoxCacheFraction;
            return (T) this;
        }

        public T pointStoreView(IPointStoreView<float[]> pointStoreView) {
            this.pointStoreView = pointStoreView;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return (T) this;
        }

        public T outputAfter(int outputAfter) {
            this.outputAfter = Optional.of(outputAfter);
            return (T) this;
        }

        public T dimension(int dimension) {
            this.dimension = dimension;
            return (T) this;
        }

        public NewRandomCutTree build() {
            return new NewRandomCutTree(this);
        }
    }
}
