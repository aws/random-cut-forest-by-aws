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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Stack;

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
    protected final HashMap<Integer, Integer> leafMass;
    protected double[] rangeSumData;
    protected float[] boundingBoxData;
    protected float[] pointSum;
    protected HashMap<Integer, List<Long>> sequenceMap;

    protected RandomCutTree(Builder<?> builder) {
        pointStoreView = builder.pointStoreView;
        numberOfLeaves = builder.capacity;
        randomSeed = builder.randomSeed;
        testRandom = builder.random;
        outputAfter = builder.outputAfter.orElse(numberOfLeaves / 4);
        dimension = (builder.dimension != 0) ? builder.dimension : pointStoreView.getDimensions();
        nodeStore = (builder.nodeStore != null) ? builder.nodeStore
                : AbstractNodeStore.builder().capacity(numberOfLeaves - 1).dimension(dimension).build();
        this.boundingBoxCacheFraction = builder.boundingBoxCacheFraction;
        this.storeSequenceIndexesEnabled = builder.storeSequenceIndexesEnabled;
        this.centerOfMassEnabled = builder.centerOfMassEnabled;
        this.root = builder.root;
        leafMass = new HashMap<>();
        int cache_limit = (int) Math.floor(boundingBoxCacheFraction * (numberOfLeaves - 1));
        rangeSumData = new double[cache_limit];
        boundingBoxData = new float[2 * dimension * cache_limit];
        if (this.centerOfMassEnabled) {
            pointSum = new float[(numberOfLeaves - 1) * dimension];
        }
        if (this.storeSequenceIndexesEnabled) {
            sequenceMap = new HashMap<>();
        }
    }

    @Override
    public <T> void setConfig(String name, T value, Class<T> clazz) {
        if (Config.BOUNDING_BOX_CACHE_FRACTION.equals(name)) {
            checkArgument(Double.class.isAssignableFrom(clazz),
                    () -> String.format("Setting '%s' must be a double value", name));
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
                    () -> String.format("Setting '%s' must be a double value", name));
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
        resizeCache(fraction);
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for a bounding box and its union with a point. The cut must
     * exist unless the union box is a single point. There are floating point issues
     * -- even though the original values are in float anf the calculations are in
     * double, which can show up with large number of dimensions (each trigerring an
     * addition/substraction).
     *
     * @param factor A random cut
     * @param point  the point whose union is taken with the box
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    protected Cut randomCut(double factor, float[] point, BoundingBox box) {
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

        checkArgument(range > 0, () -> " the union is a single point " + Arrays.toString(point)
                + "or the box is inappropriate, box" + box.toString() + "factor =" + factor);

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

        // if we are here then factor is likely almost 1 and we have floating point
        // issues
        // we will randomize between the first and the last non-zero ranges and choose
        // the
        // same cutValue as using nextAfter above -- we will use the factor as a seed
        // and
        // not be optimizing this sequel (either in execution or code) to ensure easier
        // debugging
        // this should be an anomaly - no pun intended.

        Random rng = new Random((long) factor);
        if (rng.nextDouble() < 0.5) {
            for (int i = 0; i < box.getDimensions(); i++) {
                float minValue = (float) box.getMinValue(i);
                float maxValue = (float) box.getMaxValue(i);
                if (point[i] < minValue) {
                    minValue = point[i];
                } else if (point[i] > maxValue) {
                    maxValue = point[i];
                }
                if (maxValue > minValue) {
                    double cutValue = Math.nextAfter((float) maxValue, minValue);
                    return new Cut(i, cutValue);
                }
            }
        } else {
            for (int i = box.getDimensions() - 1; i >= 0; i--) {
                float minValue = (float) box.getMinValue(i);
                float maxValue = (float) box.getMaxValue(i);
                if (point[i] < minValue) {
                    minValue = point[i];
                } else if (point[i] > maxValue) {
                    maxValue = point[i];
                }
                if (maxValue > minValue) {
                    double cutValue = Math.nextAfter((float) maxValue, minValue);
                    return new Cut(i, cutValue);
                }
            }
        }

        throw new IllegalStateException("The break point did not lie inside the expected range; factor " + factor
                + ", point " + Arrays.toString(point) + " box " + box.toString());

    }

    public Integer addPoint(Integer pointIndex, long sequenceIndex) {

        if (root == Null) {
            root = convertToLeaf(pointIndex);
            addLeaf(pointIndex, sequenceIndex);
            return pointIndex;
        } else {

            float[] point = projectToTree(pointStoreView.get(pointIndex));
            checkArgument(point.length == dimension, () -> " mismatch in dimensions for " + pointIndex);
            Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
            int[] first = pathToRoot.pop();
            int leafNode = first[0];
            int savedParent = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
            int leafSavedSibling = first[1];
            int sibling = leafSavedSibling;
            int leafPointIndex = getPointIndex(leafNode);
            float[] oldPoint = projectToTree(pointStoreView.get(leafPointIndex));
            checkArgument(oldPoint.length == dimension, () -> " mismatch in dimensions for " + pointIndex);

            Stack<int[]> parentPath = new Stack<>();

            if (Arrays.equals(point, oldPoint)) {
                increaseLeafMass(leafNode);
                checkArgument(!nodeStore.freeNodeManager.isEmpty(), "incorrect/impossible state");
                manageAncestorsAdd(pathToRoot, point);
                addLeaf(leafPointIndex, sequenceIndex);
                return leafPointIndex;
            } else {
                int node = leafNode;
                int savedNode = node;
                int parent = savedParent;
                float savedCutValue = (float) 0.0;
                BoundingBox currentBox = new BoundingBox(oldPoint, oldPoint);
                BoundingBox savedBox = currentBox.copy();
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
                        savedBox = currentBox.copy();
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
                        growNodeBox(currentBox, pointStoreView, parent, sibling);
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

                int childMassIfLeaf = isLeaf(savedNode) ? getLeafMass(savedNode) : 0;
                int mergedNode = nodeStore.addNode(pathToRoot, point, sequenceIndex, pointIndex, savedNode,
                        childMassIfLeaf, savedDim, savedCutValue, savedBox);
                addLeaf(pointIndex, sequenceIndex);
                addBox(mergedNode, point, savedBox);
                manageAncestorsAdd(pathToRoot, point);
                if (pointSum != null) {
                    recomputePointSum(mergedNode);
                }
                if (savedParent == Null) {
                    root = mergedNode;
                }
            }
            return pointIndex;
        }
    }

    protected void manageAncestorsAdd(Stack<int[]> path, float[] point) {
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            nodeStore.increaseMassOfInternalNode(index);
            if (pointSum != null) {
                recomputePointSum(index);
            }
            if (boundingBoxCacheFraction > 0.0) {
                checkContainsAndRebuildBox(index, point, pointStoreView);
                checkContainsAndAddPoint(index, point);
            }
        }
    }

    public void addPointToPartialTree(Integer pointIndex, long sequenceIndex) {

        checkArgument(root != Null, " a null root is not a partial tree");
        float[] point = projectToTree(pointStoreView.get(pointIndex));
        checkArgument(point.length == dimension, () -> " incorrect projection at index " + pointIndex);

        Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
        int[] first = pathToRoot.pop();
        int leafNode = first[0];
        int savedParent = (pathToRoot.size() == 0) ? Null : pathToRoot.lastElement()[0];
        if (!nodeStore.isLeaf(leafNode)) {
            if (savedParent == Null) {
                root = convertToLeaf(pointIndex);
            } else {
                nodeStore.assignInPartialTree(savedParent, point, convertToLeaf(pointIndex));
                nodeStore.manageInternalNodesPartial(pathToRoot);
                addLeaf(pointIndex, sequenceIndex);
            }
            return;
        }
        int leafPointIndex = getPointIndex(leafNode);
        float[] oldPoint = projectToTree(pointStoreView.get(leafPointIndex));

        checkArgument(oldPoint.length == dimension && Arrays.equals(point, oldPoint),
                () -> "incorrect state on adding " + pointIndex);
        increaseLeafMass(leafNode);
        checkArgument(!nodeStore.freeNodeManager.isEmpty(), "incorrect/impossible state");
        nodeStore.manageInternalNodesPartial(pathToRoot);
        addLeaf(leafPointIndex, sequenceIndex);
        return;
    }

    public Integer deletePoint(Integer pointIndex, long sequenceIndex) {

        checkArgument(root != Null, " deleting from an empty tree");
        float[] point = projectToTree(pointStoreView.get(pointIndex));
        checkArgument(point.length == dimension, () -> " incorrect projection at index " + pointIndex);
        Stack<int[]> pathToRoot = nodeStore.getPath(root, point, false);
        int[] first = pathToRoot.pop();
        int leafSavedSibling = first[1];
        int leafNode = first[0];
        int leafPointIndex = getPointIndex(leafNode);

        checkArgument(leafPointIndex == pointIndex,
                () -> " deleting wrong node " + leafPointIndex + " instead of " + pointIndex);

        removeLeaf(leafPointIndex, sequenceIndex);

        if (decreaseLeafMass(leafNode) == 0) {
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
                    manageAncestorsDelete(pathToRoot, point);
                }
                nodeStore.deleteInternalNode(parent);
                if (pointSum != null) {
                    invalidatePointSum(parent);
                }
                int idx = translate(parent);
                if (idx != Integer.MAX_VALUE) {
                    rangeSumData[idx] = 0.0;
                }
            }
        } else {
            manageAncestorsDelete(pathToRoot, point);
        }
        return leafPointIndex;
    }

    protected void manageAncestorsDelete(Stack<int[]> path, float[] point) {
        boolean resolved = false;
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            nodeStore.decreaseMassOfInternalNode(index);
            if (pointSum != null) {
                recomputePointSum(index);
            }
            if (boundingBoxCacheFraction > 0.0 && !resolved) {
                resolved = checkContainsAndRebuildBox(index, point, pointStoreView);
            }
        }
    }

    //// leaf, nonleaf representations

    public boolean isLeaf(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index >= numberOfLeaves;
    }

    public boolean isInternal(int index) {
        // note that numberOfLeaves - 1 corresponds to an unspefied leaf in partial tree
        // 0 .. numberOfLeaves - 2 corresponds to internal nodes
        return index < numberOfLeaves - 1;
    }

    public int convertToLeaf(int pointIndex) {
        return pointIndex + numberOfLeaves;
    }

    public int getPointIndex(int index) {
        checkArgument(index >= numberOfLeaves, () -> " does not have a point associated " + index);
        return index - numberOfLeaves;
    }

    public int getLeftChild(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get left Index " + index);
        return nodeStore.getLeftIndex(index);
    }

    public int getRightChild(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get right child " + index);
        return nodeStore.getRightIndex(index);
    }

    public int getCutDimension(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get cut dimension " + index);
        return nodeStore.getCutDimension(index);
    }

    public double getCutValue(int index) {
        checkArgument(isInternal(index), () -> "incorrect call to get cut value " + index);
        return nodeStore.getCutValue(index);
    }

    ///// mass assignments; separating leafs and internal nodes

    protected int getMass(int index) {
        return (isLeaf(index)) ? getLeafMass(index) : nodeStore.getMass(index);
    }

    protected int getLeafMass(int index) {
        int y = (index - numberOfLeaves);
        Integer value = leafMass.get(y);
        return (value != null) ? value + 1 : 1;
    }

    protected void increaseLeafMass(int index) {
        int y = (index - numberOfLeaves);
        leafMass.merge(y, 1, Integer::sum);
    }

    protected int decreaseLeafMass(int index) {
        int y = (index - numberOfLeaves);
        Integer value = leafMass.remove(y);
        if (value != null) {
            if (value > 1) {
                leafMass.put(y, (value - 1));
                return value;
            } else {
                return 1;
            }
        } else {
            return 0;
        }
    }

    @Override
    public int getMass() {
        return root == Null ? 0 : isLeaf(root) ? getLeafMass(root) : nodeStore.getMass(root);
    }

    /////// Bounding box

    public void resizeCache(double fraction) {
        if (fraction == 0) {
            rangeSumData = null;
            boundingBoxData = null;
        } else {
            int limit = (int) Math.floor(fraction * (numberOfLeaves - 1));
            rangeSumData = (rangeSumData == null) ? new double[limit] : Arrays.copyOf(rangeSumData, limit);
            boundingBoxData = (boundingBoxData == null) ? new float[limit * 2 * dimension]
                    : Arrays.copyOf(boundingBoxData, limit * 2 * dimension);
        }
        boundingBoxCacheFraction = fraction;
    }

    protected int translate(int index) {
        if (rangeSumData == null || rangeSumData.length <= index) {
            return Integer.MAX_VALUE;
        } else {
            return index;
        }
    }

    void copyBoxToData(int idx, BoundingBox box) {
        int base = 2 * idx * dimension;
        int mid = base + dimension;
        System.arraycopy(box.getMinValues(), 0, boundingBoxData, base, dimension);
        System.arraycopy(box.getMaxValues(), 0, boundingBoxData, mid, dimension);
        rangeSumData[idx] = box.getRangeSum();
    }

    boolean checkContainsAndAddPoint(int index, float[] point) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE && rangeSumData[idx] != 0) {
            int base = 2 * idx * dimension;
            int mid = base + dimension;
            double rangeSum = 0;
            for (int i = 0; i < dimension; i++) {
                boundingBoxData[base + i] = Math.min(boundingBoxData[base + i], point[i]);
            }
            for (int i = 0; i < dimension; i++) {
                boundingBoxData[mid + i] = Math.max(boundingBoxData[mid + i], point[i]);
            }
            for (int i = 0; i < dimension; i++) {
                rangeSum += boundingBoxData[mid + i] - boundingBoxData[base + i];
            }
            boolean answer = (rangeSumData[idx] == rangeSum);
            rangeSumData[idx] = rangeSum;
            return answer;
        }
        return false;
    }

    public BoundingBox getBox(int index) {
        if (isLeaf(index)) {
            float[] point = projectToTree(pointStoreView.get(getPointIndex(index)));
            checkArgument(point.length == dimension, () -> "failure in projection at index " + index);
            return new BoundingBox(point, point);
        } else {
            checkArgument(isInternal(index), " incomplete state");
            int idx = translate(index);
            if (idx != Integer.MAX_VALUE) {
                if (rangeSumData[idx] != 0) {
                    // return non-trivial boxes
                    return getBoxFromData(idx);
                } else {
                    BoundingBox box = reconstructBox(index, pointStoreView);
                    copyBoxToData(idx, box);
                    return box;
                }
            }
            return reconstructBox(index, pointStoreView);
        }
    }

    BoundingBox reconstructBox(int index, IPointStoreView<float[]> pointStoreView) {
        BoundingBox mutatedBoundingBox = getBox(nodeStore.getLeftIndex(index));
        growNodeBox(mutatedBoundingBox, pointStoreView, index, nodeStore.getRightIndex(index));
        return mutatedBoundingBox;
    }

    boolean checkStrictlyContains(int index, float[] point) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            int base = 2 * idx * dimension;
            int mid = base + dimension;
            boolean isInside = true;
            for (int i = 0; i < dimension && isInside; i++) {
                if (point[i] >= boundingBoxData[mid + i] || boundingBoxData[base + i] >= point[i]) {
                    isInside = false;
                }
            }
            return isInside;
        }
        return false;
    }

    boolean checkContainsAndRebuildBox(int index, float[] point, IPointStoreView<float[]> pointStoreView) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE && rangeSumData[idx] != 0) {
            if (!checkStrictlyContains(index, point)) {
                BoundingBox mutatedBoundingBox = reconstructBox(index, pointStoreView);
                copyBoxToData(idx, mutatedBoundingBox);
                return false;
            }
            return true;
        }
        return false;
    }

    BoundingBox getBoxFromData(int idx) {
        int base = 2 * idx * dimension;
        int mid = base + dimension;

        return new BoundingBox(Arrays.copyOfRange(boundingBoxData, base, base + dimension),
                Arrays.copyOfRange(boundingBoxData, mid, mid + dimension));
    }

    void addBox(int index, float[] point, BoundingBox box) {
        if (isInternal(index)) {
            int idx = translate(index);
            if (idx != Integer.MAX_VALUE) { // always add irrespective of rangesum
                copyBoxToData(idx, box);
                checkContainsAndAddPoint(index, point);
            }
        }
    }

    void growNodeBox(BoundingBox box, IPointStoreView<float[]> pointStoreView, int node, int sibling) {
        if (isLeaf(sibling)) {
            float[] point = projectToTree(pointStoreView.get(getPointIndex(sibling)));
            checkArgument(point.length == dimension, () -> " incorrect projection at index " + sibling);
            box.addPoint(point);
        } else {
            if (!isInternal(sibling)) {
                throw new IllegalStateException(" incomplete state " + sibling);
            }
            int siblingIdx = translate(sibling);
            if (siblingIdx != Integer.MAX_VALUE) {
                if (rangeSumData[siblingIdx] != 0) {
                    box.addBox(getBoxFromData(siblingIdx));
                } else {
                    BoundingBox newBox = getBox(siblingIdx);
                    copyBoxToData(siblingIdx, newBox);
                    box.addBox(newBox);
                }
                return;
            }
            growNodeBox(box, pointStoreView, sibling, nodeStore.getLeftIndex(sibling));
            growNodeBox(box, pointStoreView, sibling, nodeStore.getRightIndex(sibling));
            return;
        }
    }

    public double probabilityOfCut(int node, float[] point, BoundingBox otherBox) {
        int nodeIdx = translate(node);
        if (nodeIdx != Integer.MAX_VALUE && rangeSumData[nodeIdx] != 0) {
            int base = 2 * nodeIdx * dimension;
            int mid = base + dimension;
            double minsum = 0;
            double maxsum = 0;
            for (int i = 0; i < dimension; i++) {
                minsum += Math.max(boundingBoxData[base + i] - point[i], 0);
            }
            for (int i = 0; i < dimension; i++) {
                maxsum += Math.max(point[i] - boundingBoxData[mid + i], 0);
            }
            double sum = maxsum + minsum;

            if (sum == 0.0) {
                return 0.0;
            }
            return sum / (rangeSumData[nodeIdx] + sum);
        } else if (otherBox != null) {
            return otherBox.probabilityOfCut(point);
        } else {
            BoundingBox box = getBox(node);
            return box.probabilityOfCut(point);
        }
    }

    /// additional information at nodes

    public float[] getPointSum(int index) {
        checkArgument(centerOfMassEnabled, " enable center of mass");
        if (isLeaf(index)) {
            float[] point = projectToTree(pointStoreView.get(getPointIndex(index)));
            checkArgument(point.length == dimension, () -> " incorrect projection");
            int mass = getMass(index);
            for (int i = 0; i < point.length; i++) {
                point[i] *= mass;
            }
            return point;
        } else {
            return Arrays.copyOfRange(pointSum, index * dimension, (index + 1) * dimension);
        }
    }

    public void invalidatePointSum(int index) {
        for (int i = 0; i < dimension; i++) {
            pointSum[index * dimension + i] = 0;
        }
    }

    public void recomputePointSum(int index) {
        float[] left = getPointSum(nodeStore.getLeftIndex(index));
        float[] right = getPointSum(nodeStore.getRightIndex(index));
        for (int i = 0; i < dimension; i++) {
            pointSum[index * dimension + i] = left[i] + right[i];
        }
    }

    public HashMap<Long, Integer> getSequenceMap(int index) {
        HashMap<Long, Integer> hashMap = new HashMap<>();
        List<Long> list = getSequenceList(index);
        for (Long e : list) {
            hashMap.merge(e, 1, Integer::sum);
        }
        return hashMap;
    }

    public List<Long> getSequenceList(int index) {
        return sequenceMap.get(index);
    }

    protected void addLeaf(int pointIndex, long sequenceIndex) {
        if (storeSequenceIndexesEnabled) {
            List<Long> leafList = sequenceMap.remove(pointIndex);
            if (leafList == null) {
                leafList = new ArrayList<>(1);
            }
            leafList.add(sequenceIndex);
            sequenceMap.put(pointIndex, leafList);
        }
    }

    public void removeLeaf(int leafPointIndex, long sequenceIndex) {
        if (storeSequenceIndexesEnabled) {
            List<Long> leafList = sequenceMap.remove(leafPointIndex);
            checkArgument(leafList != null, " leaf index not found in tree");
            checkArgument(leafList.remove(sequenceIndex), " sequence index not found in leaf");
            if (!leafList.isEmpty()) {
                sequenceMap.put(leafPointIndex, leafList);
            }
        }
    }

    //// validations

    public void validateAndReconstruct() {
        if (root != Null) {
            validateAndReconstruct(root);
        }
    }

    /**
     * This function is supposed to validate the integrity of the tree and rebuild
     * internal data structures. At this moment the only internal structure is the
     * pointsum.
     * 
     * @param index the node of a tree
     * @return a bounding box of the points
     */
    public BoundingBox validateAndReconstruct(int index) {
        if (isLeaf(index)) {
            return getBox(index);
        } else {
            BoundingBox leftBox = validateAndReconstruct(getLeftChild(index));
            BoundingBox rightBox = validateAndReconstruct(getRightChild(index));
            if (leftBox.maxValues[getCutDimension(index)] > getCutValue(index)
                    || rightBox.minValues[getCutDimension(index)] <= getCutValue(index)) {
                throw new IllegalStateException(" incorrect bounding state at index " + index + " cut value "
                        + getCutValue(index) + "cut dimension " + getCutDimension(index) + " left Box "
                        + leftBox.toString() + " right box " + rightBox.toString());
            }
            if (centerOfMassEnabled) {
                recomputePointSum(index);
            }
            rightBox.addBox(leftBox);
            int idx = translate(index);
            if (idx != Integer.MAX_VALUE) { // always add irrespective of rangesum
                copyBoxToData(idx, rightBox);
            }
            return rightBox;
        }
    }

    //// traversals

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
    public <R> R traverse(float[] point, IVisitorFactory<R> visitorFactory) {
        checkState(root != Null, "this tree doesn't contain any nodes");
        Visitor<R> visitor = visitorFactory.newVisitor(this, point);
        NodeView currentNodeView = new NodeView(this, pointStoreView, root);
        traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    protected <R> void traversePathToLeafAndVisitNodes(float[] point, Visitor<R> visitor, NodeView currentNodeView,
            int node, int depthOfNode) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), true);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            checkArgument(isInternal(node), () -> " incomplete state " + node + " " + depthOfNode);
            if (nodeStore.toLeft(point, node)) {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, nodeStore.getLeftIndex(node),
                        depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getRightIndex(node), !visitor.isConverged());
            } else {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, nodeStore.getRightIndex(node),
                        depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), !visitor.isConverged());
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traversal path (defined
     * in {@link #traverse(float[], IVisitorFactory)}) but at Node in checks to see
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
    public <R> R traverseMulti(float[] point, IMultiVisitorFactory<R> visitorFactory) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitorFactory, "visitor must not be null");
        checkState(root != Null, "this tree doesn't contain any nodes");
        MultiVisitor<R> visitor = visitorFactory.newVisitor(this, point);
        NodeView currentNodeView = new NodeView(this, pointStoreView, root);
        traverseTreeMulti(point, visitor, currentNodeView, root, 0);
        return visitorFactory.liftResult(this, visitor.getResult());
    }

    protected <R> void traverseTreeMulti(float[] point, MultiVisitor<R> visitor, NodeView currentNodeView, int node,
            int depthOfNode) {
        if (nodeStore.isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), false);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            checkArgument(nodeStore.isInternal(node), " incomplete state");
            currentNodeView.setCurrentNodeOnly(node);
            if (visitor.trigger(currentNodeView)) {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getLeftIndex(node), depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                currentNodeView.setCurrentNodeOnly(nodeStore.getRightIndex(node));
                traverseTreeMulti(point, newVisitor, currentNodeView, nodeStore.getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), false);
                visitor.combine(newVisitor);
            } else if (nodeStore.toLeft(point, node)) {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getLeftIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getRightIndex(node), false);
            } else {
                traverseTreeMulti(point, visitor, currentNodeView, nodeStore.getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, nodeStore.getLeftIndex(node), false);
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
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

    public float[] projectToTree(float[] point) {
        return Arrays.copyOf(point, point.length);
    }

    public float[] liftFromTree(float[] result) {
        return Arrays.copyOf(result, result.length);
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
