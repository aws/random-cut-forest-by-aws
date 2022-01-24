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
import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;

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
public class RandomCutTree implements ITree<Integer, float[]> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */

    private Random testRandom;
    protected boolean storeSequenceIndexesEnabled;
    protected boolean centerOfMassEnabled;
    private long randomSeed;
    protected int root;
    protected IPointStoreView<float[]> pointStoreView;
    protected int numberOfLeaves;
    protected AbstractNodeStore nodeStore;
    protected double boundingBoxCacheFraction;
    protected int outputAfter;
    protected int dimension;

    protected RandomCutTree(Builder<?> builder) {
        pointStoreView = builder.pointStoreView;
        numberOfLeaves = builder.capacity;
        randomSeed = builder.randomSeed;
        testRandom = builder.random;
        outputAfter = builder.outputAfter.orElse(numberOfLeaves / 4);
        dimension = (builder.dimension != 0) ? builder.dimension : pointStoreView.getDimensions();
        nodeStore = (builder.nodeStore != null) ? builder.nodeStore
                : AbstractNodeStore.builder().capacity(numberOfLeaves - 1).dimensions(dimension)
                        .boundingBoxCacheFraction(builder.boundingBoxCacheFraction).pointStoreView(pointStoreView)
                        .centerOfMassEnabled(builder.centerOfMassEnabled)
                        .storeSequencesEnabled(builder.storeSequenceIndexesEnabled).build();
        // note the number of internal nodes is one less than sampleSize
        // the RCF V2_0 states used this notion
        this.boundingBoxCacheFraction = builder.boundingBoxCacheFraction;
        this.storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        this.centerOfMassEnabled = builder.centerOfMassEnabled;
        this.root = builder.root;
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    String.format("Setting '%s' must be a double value", name));
            setBoundingBoxCacheFraction((Double) value);
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
            return clazz.cast(boundingBoxCacheFraction);
        } else {
            throw new IllegalArgumentException("Unsupported configuration setting: " + name);
        }
    }

    // dynamically change the fraction of the new nodes which caches their bounding
    // boxes
    // 0 would mean less space usage, but slower throughput
    // 1 would imply larger space but better throughput
    public void setBoundingBoxCacheFraction(double fraction) {
        checkArgument(0 <= fraction && fraction <= 1, "incorrect parameter");
        boundingBoxCacheFraction = fraction;
        nodeStore.resizeCache(fraction);
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

        if (root == Null) {
            root = nodeStore.addLeaf(pointIndex, sequenceIndex);
            return pointIndex;
        } else {

            float[] point = pointStoreView.get(pointIndex);
            Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
            int[] first = pathToRoot.pop();
            int leafNode = first[0];
            int savedParent = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
            if (!nodeStore.isLeaf(leafNode)) {
                // this corresponds to rebuilding a partial tree
                if (savedParent == Null) {
                    root = pointIndex + numberOfLeaves; // note this capacity is nodestore.capacity + 1
                } else {
                    nodeStore.addToPartialTree(pathToRoot, point, pointIndex);
                    nodeStore.manageAncestorsAdd(pathToRoot, point, pointStoreView);
                    nodeStore.addLeaf(pointIndex, sequenceIndex);
                }
                return pointIndex;
            }
            int leafSavedSibling = first[1];
            int sibling = leafSavedSibling;
            int leafPointIndex = nodeStore.getPointIndex(leafNode);
            float[] oldPoint = pointStoreView.get(leafPointIndex);
            Stack<int[]> parentPath = new Stack<>();

            if (Arrays.equals(point, oldPoint)) {
                nodeStore.increaseLeafMass(leafNode);
                checkArgument(!nodeStore.freeNodeManager.isEmpty(), "incorrect/impossible state");
                nodeStore.manageAncestorsAdd(pathToRoot, point, pointStoreView);
                nodeStore.addLeaf(leafPointIndex, sequenceIndex);
                return leafPointIndex;
            } else {
                int node = leafNode;
                int savedNode = node;
                int parent = savedParent;
                float savedCutValue = (float) 0.0;
                BoundingBoxFloat currentBox = new BoundingBoxFloat(oldPoint, oldPoint);
                BoundingBoxFloat savedBox = currentBox.newCopy();
                int savedDim = Integer.MAX_VALUE;
                Random rng;
                if (testRandom == null) {
                    rng = new Random(randomSeed);
                    randomSeed = rng.nextLong();
                } else {
                    rng = testRandom;
                }
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
                    if (currentBox.contains(point) || parent == Null) {
                        break;
                    } else {
                        nodeStore.growNodeBox(currentBox, pointStoreView, parent, sibling);
                        int[] next = pathToRoot.pop();
                        node = next[0];
                        sibling = next[1];
                        if (pathToRoot.size() != 0) {
                            parent = pathToRoot.lastElement()[0];
                        } else {
                            parent = Null;
                        }
                    }
                }
                if (savedParent != Null) {
                    while (!parentPath.isEmpty()) {
                        pathToRoot.push(parentPath.pop());
                    }
                    assert (pathToRoot.lastElement()[0] == savedParent);
                }

                int mergedNode = nodeStore.addNode(pathToRoot, point, sequenceIndex, pointIndex, savedNode, savedDim,
                        savedCutValue, savedBox);
                if (savedParent == Null) {
                    root = mergedNode;
                }
            }
            return pointIndex;
        }
    }

    public Integer deletePoint(Integer pointIndex, long sequenceIndex) {

        if (root == Null) {
            throw new IllegalStateException(" deleting from an empty tree");
        }
        float[] point = pointStoreView.get(pointIndex);
        Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
        int[] first = pathToRoot.pop();
        int leafSavedSibling = first[1];
        int leafNode = first[0];
        int leafPointIndex = nodeStore.getPointIndex(leafNode);

        if (leafPointIndex != pointIndex && !pointStoreView.pointEquals(leafPointIndex, point)) {
            throw new IllegalStateException(" deleting wrong node " + leafPointIndex + " instead of " + pointIndex);
        } else if (storeSequenceIndexesEnabled) {
            nodeStore.removeLeaf(leafPointIndex, sequenceIndex);
        }

        if (nodeStore.decreaseLeafMass(leafNode) == 0) {
            if (pathToRoot.size() == 0) {
                root = Null;
            } else {
                int parent = pathToRoot.pop()[0];
                if (pathToRoot.size() == 0) {
                    root = leafSavedSibling;
                    nodeStore.setRoot(root);
                } else {
                    int grandParent = pathToRoot.lastElement()[0];
                    nodeStore.replaceParentBySibling(grandParent, parent, leafNode);
                    nodeStore.manageAncestorsDelete(pathToRoot, point, pointStoreView);
                }
                nodeStore.deleteInternalNode(parent);
            }
        } else {
            nodeStore.manageAncestorsDelete(pathToRoot, point, pointStoreView);
        }
        return leafPointIndex;
    }

    public double scalarScore(double[] point, int ignoreMass, BiFunction<Double, Double, Double> scoreSeen,
            BiFunction<Double, Double, Double> scoreUnseen, BiFunction<Double, Double, Double> damp,
            BiFunction<Double, Double, Double> normalizer) {
        Function<Double, Double> treeDamp = x -> damp.apply(x, getMass() * 1.0);

        return normalizer.apply(
                nodeStore.dynamicScore(root, ignoreMass, point, pointStoreView, scoreSeen, scoreUnseen, treeDamp),
                getMass() * 1.0);
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
        checkState(root != Null, "this tree doesn't contain any nodes");
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
        checkState(root != Null, "this tree doesn't contain any nodes");
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
        return root == Null ? 0 : nodeStore.getMass(root);
    }

    public int getNumberOfLeaves() {
        return numberOfLeaves;
    }

    public boolean isCenterOfMassEnabled() {
        return centerOfMassEnabled;
    }

    public boolean isStoreSequenceIndexesEnabled() {
        return storeSequenceIndexesEnabled;
    }

    public double getBoundingBoxCacheFraction() {
        return boundingBoxCacheFraction;
    }

    public int getDimension() {
        return dimension;
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

    public AbstractNodeStore getNodeStore() {
        return nodeStore;
    }

    public static class Builder<T extends Builder<T>> {
        protected boolean storeSequenceIndexesEnabled = RandomCutForest.DEFAULT_STORE_SEQUENCE_INDEXES_ENABLED;
        protected boolean centerOfMassEnabled = RandomCutForest.DEFAULT_CENTER_OF_MASS_ENABLED;
        protected double boundingBoxCacheFraction = RandomCutForest.DEFAULT_BOUNDING_BOX_CACHE_FRACTION;
        protected long randomSeed = new Random().nextLong();
        protected Random random = null;
        protected int capacity = RandomCutForest.DEFAULT_SAMPLE_SIZE;
        protected Optional<Integer> outputAfter = Optional.empty();
        protected int dimension;
        protected IPointStoreView<float[]> pointStoreView;
        protected AbstractNodeStore nodeStore;
        protected int root = Null;

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T boundingBoxCacheFraction(double boundingBoxCacheFraction) {
            this.boundingBoxCacheFraction = boundingBoxCacheFraction;
            return (T) this;
        }

        public T pointStoreView(IPointStoreView<float[]> pointStoreView) {
            this.pointStoreView = pointStoreView;
            return (T) this;
        }

        public T nodeStore(AbstractNodeStore nodeStore) {
            this.nodeStore = nodeStore;
            return (T) this;
        }

        public T randomSeed(long randomSeed) {
            this.randomSeed = randomSeed;
            return (T) this;
        }

        public T random(Random random) {
            this.random = random;
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

        public T setRoot(int root) {
            this.root = root;
            return (T) this;
        }

        public T storeSequenceIndexesEnabled(boolean storeSequenceIndexesEnabled) {
            this.storeSequenceIndexesEnabled = storeSequenceIndexesEnabled;
            return (T) this;
        }

        public T centerOfMassEnabled(boolean centerOfMassEnabled) {
            this.centerOfMassEnabled = centerOfMassEnabled;
            return (T) this;
        }

        public RandomCutTree build() {
            return new RandomCutTree(this);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}
