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

import static com.amazon.randomcutforest.CommonUtils.toFloatArray;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Stack;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.store.IPointStoreView;
import com.amazon.randomcutforest.store.IntervalManager;

/**
 * A fixed-size buffer for storing interior tree nodes. An interior node is
 * defined by its location in the tree (parent and child nodes), its random cut,
 * and its bounding box. The NodeStore class uses arrays to store these field
 * values for a collection of nodes. An index in the store can be used to look
 * up the field values for a particular node.
 *
 * The internal nodes (handled by this store) corresponds to
 * [0..upperRangeLimit]
 *
 * If we think of an array of Node objects as being row-oriented (where each row
 * is a Node), then this class is analogous to a column-oriented database of
 * Nodes.
 *
 * Note that a NodeStore does not store instances of the
 * {@link com.amazon.randomcutforest.tree.Node} class.
 */
public abstract class AbstractNodeStore {

    public static double SWITCH_FRACTION = 0.499;

    protected final int capacity;
    protected final int dimensions;
    protected final float[] cutValue;
    protected double nodeCacheFraction;
    protected IntervalManager freeNodeManager;
    protected double[] rangeSumData;
    protected float[] boundingBoxData;
    protected final HashMap<Integer, Integer> leafMass;

    /**
     * Create a new NodeStore with the given capacity.
     *
     * @param capacity The maximum number of Nodes whose data can be stored.
     */
    public AbstractNodeStore(int capacity, int dimensions, double nodeCacheFraction) {
        this.capacity = capacity;
        this.dimensions = dimensions;
        freeNodeManager = new IntervalManager(capacity - 1);
        this.nodeCacheFraction = nodeCacheFraction;
        cutValue = new float[capacity - 1];
        leafMass = new HashMap<>();
        int cache_limit = (int) Math.floor(nodeCacheFraction * capacity);
        rangeSumData = new double[cache_limit];
        boundingBoxData = new float[2 * dimensions * cache_limit];
    }

    abstract int addNode(int parentIndex, int leftIndex, int rightIndex, int cutDimension, double cutValue, int mass);

    public int addLeaf(int parentIndex, int pointIndex, int mass) {
        return pointIndex + capacity;
    }

    public boolean isLeaf(int index) {
        return index != 0 && index >= capacity;
    }

    public abstract int getLeftIndex(int index);

    public abstract int getRightIndex(int index);

    public abstract void setRoot(int index);

    public void increaseLeafMass(int index) {
        int y = (index - capacity);
        leafMass.merge(y, 1, Integer::sum);
    }

    public int decreaseLeafMass(int index) {
        int y = (index - capacity);
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

    public int translate(int index) {
        if (index != 0 && rangeSumData.length <= index - 1) {
            return Integer.MAX_VALUE;
        } else {
            return index - 1;
        }
    }

    public void copyBoxToData(int index, BoundingBoxFloat box) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            int base = 2 * idx * dimensions;
            int mid = base + dimensions;
            System.arraycopy(box.getMinValues(), 0, boundingBoxData, base, dimensions);
            System.arraycopy(box.getMaxValues(), 0, boundingBoxData, mid, dimensions);
            rangeSumData[idx] = box.getRangeSum();
        }
    }

    public boolean checkContainsAndAddPoint(int index, float[] point) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            int base = 2 * idx * dimensions;
            int mid = base + dimensions;
            double rangeSum = 0;
            for (int i = 0; i < dimensions; i++) {
                boundingBoxData[base + i] = Math.min(boundingBoxData[base + i], point[i]);
            }
            for (int i = 0; i < dimensions; i++) {
                boundingBoxData[mid + i] = Math.max(boundingBoxData[mid + i], point[i]);
            }
            for (int i = 0; i < dimensions; i++) {
                rangeSum += boundingBoxData[mid + i] - boundingBoxData[base + i];
            }
            boolean answer = (rangeSumData[idx] == rangeSum);
            rangeSumData[idx] = rangeSum;
            return answer;
        }
        return false;
    }

    public BoundingBoxFloat getBox(int index, IPointStoreView<float[]> pointStoreView) {
        if (isLeaf(index)) {
            float[] point = pointStoreView.get(getPointIndex(index));
            return new BoundingBoxFloat(point, point);
        } else {
            int idx = translate(index);
            if (idx != Integer.MAX_VALUE) {
                return getBoxFromData(idx);
            }
            return reconstructBox(index, pointStoreView);
        }
    }

    public BoundingBoxFloat reconstructBox(int index, IPointStoreView<float[]> pointStoreView) {
        BoundingBoxFloat mutatedBoundingBox = getBox(getLeftIndex(index), pointStoreView);
        growNodeBox(mutatedBoundingBox, pointStoreView, index, getRightIndex(index));
        return mutatedBoundingBox;
    }

    boolean checkStrictlyContains(int index, float[] point) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            int base = 2 * idx * dimensions;
            int mid = base + dimensions;
            boolean isInside = true;
            for (int i = 0; i < dimensions && isInside; i++) {
                if (point[i] >= boundingBoxData[mid + i] || boundingBoxData[base + i] >= point[i]) {
                    isInside = false;
                }
            }
            return isInside;
        }
        return false;
    }

    public boolean checkContainsAndRebuildBox(int index, float[] point, IPointStoreView<float[]> pointStoreView) {
        int idx = translate(index);
        if (idx != Integer.MAX_VALUE) {
            if (!checkStrictlyContains(index, point)) {
                BoundingBoxFloat mutatedBoundingBox = reconstructBox(index, pointStoreView);
                copyBoxToData(index, mutatedBoundingBox);
                return false;
            }
            return true;
        } else {
            return false;
        }
    }

    public BoundingBoxFloat getBoxFromData(int idx) {
        int base = 2 * idx * dimensions;
        int mid = base + dimensions;

        return new BoundingBoxFloat(Arrays.copyOfRange(boundingBoxData, base, base + dimensions),
                Arrays.copyOfRange(boundingBoxData, mid, mid + dimensions));
    }

    public void addBox(int index, BoundingBoxFloat box) {
        if (!isLeaf(index)) {
            copyBoxToData(index, box);
        }
    }

    public void growNodeBox(BoundingBoxFloat box, IPointStoreView<float[]> pointStoreView, int node, int sibling) {
        if (isLeaf(sibling)) {
            float[] point = pointStoreView.get(getPointIndex(sibling));
            box.addPoint(point);
        } else {
            int sibling_idx = translate(sibling);
            if (sibling_idx != Integer.MAX_VALUE) {
                box.addBox(getBoxFromData(sibling_idx));
                return;
            }
            growNodeBox(box, pointStoreView, sibling, getLeftIndex(sibling));
            growNodeBox(box, pointStoreView, sibling, getRightIndex(sibling));
            return;
        }
    }

    public double probabilityOfCut(int node, float[] point, IPointStoreView<float[]> pointStoreView,
            BoundingBoxFloat otherBox) {
        int node_idx = translate(node);
        if (node_idx != Integer.MAX_VALUE) {
            int base = 2 * node_idx * dimensions;
            int mid = base + dimensions;
            double minsum = 0;
            double maxsum = 0;
            for (int i = 0; i < dimensions; i++) {
                minsum += Math.max(boundingBoxData[base + i] - point[i], 0);
            }
            for (int i = 0; i < dimensions; i++) {
                maxsum += Math.max(point[i] - boundingBoxData[mid + i], 0);
            }
            double sum = maxsum + minsum;

            if (sum == 0.0) {
                return 0.0;
            }
            return sum / (rangeSumData[node_idx] + sum);
        } else if (otherBox != null) {
            return otherBox.probabilityOfCut(point);
        } else {
            BoundingBoxFloat box = getBox(node, pointStoreView);
            return box.probabilityOfCut(point);
        }
    }

    protected abstract void decreaseMassOfInternalNode(int node);

    protected abstract void increaseMassOfInternalNode(int node);

    protected void manageAncestorsAdd(Stack<int[]> path, float[] point, IPointStoreView<float[]> pointStoreview) {
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            increaseMassOfInternalNode(index);
            if (nodeCacheFraction > 0.0) {
                checkContainsAndRebuildBox(index, point, pointStoreview);
                checkContainsAndAddPoint(index, point);
            }
        }
    }

    protected void manageAncestorsDelete(Stack<int[]> path, float[] point, IPointStoreView<float[]> pointStoreview) {
        boolean resolved = false;
        while (!path.isEmpty()) {
            int index = path.pop()[0];
            decreaseMassOfInternalNode(index);
            if (nodeCacheFraction > 0.0 && !resolved) {
                resolved = checkContainsAndRebuildBox(index, point, pointStoreview);
            }
        }
    }

    public Stack<int[]> getPath(int root, float[] point, boolean verbose) {
        int node = root;
        Stack<int[]> answer = new Stack<>();
        answer.push(new int[] { root, 0 });
        while (!isLeaf(node)) {
            if (leftOf(node, point)) {
                answer.push(new int[] { getLeftIndex(node), getRightIndex(node) });
                node = getLeftIndex(node);
            } else {
                answer.push(new int[] { getRightIndex(node), getLeftIndex(node) });
                node = getRightIndex(node);
            }
        }
        return answer;
    }

    public abstract void deleteInternalNode(int index);

    public int getLeafMass(int index) {
        int y = (index - capacity);
        Integer value = leafMass.get(y);
        if (value != null) {
            return value;
        } else {
            return 1;
        }
    }

    public abstract int getMass(int index);

    public int getPointIndex(int index) {
        return index - capacity;
    }

    public boolean leftOf(int node, float[] point) {
        return point[getCutDimension(node)] <= cutValue[node - 1];
    }

    public int getSibling(int node, int parent) {
        int sibling = getLeftIndex(parent);
        if (node == sibling) {
            sibling = getRightIndex(parent);
        }
        return sibling;
    }

    public abstract void spliceEdge(int parent, int node, int newNode);

    public abstract void replaceParentBySibling(int grandParent, int parent, int node);

    public double dynamicScore2(int root, int ignoreMass, double[] point, IPointStoreView<float[]> pointStoreView,
            BiFunction<Double, Double, Double> scoreSeen, BiFunction<Double, Double, Double> scoreUnseen,
            Function<Double, Double> treeDamp) {
        if (root == 0) {
            return 0.0;
        }
        float[] floatPoint = toFloatArray(point);
        Stack<int[]> path = getPath(root, floatPoint, false);
        int[] leafState = path.pop();
        double depth = path.size(); // accounts for level 0 at root
        double mass = getMass(leafState[0]);
        float[] oldPoint = pointStoreView.get(getPointIndex(leafState[0]));

        boolean shadow = (mass <= ignoreMass);
        if (Arrays.equals(floatPoint, oldPoint) && !shadow) {
            return treeDamp.apply(mass) * scoreSeen.apply(depth, mass);
        }

        double scoreValue = scoreUnseen.apply(depth, mass);
        if (!path.isEmpty()) {
            int sibling = leafState[1];
            int parent = path.lastElement()[0];
            BoundingBoxFloat boundingBox = (shadow) ? getBox(sibling, pointStoreView)
                    : (nodeCacheFraction < SWITCH_FRACTION) ? getBox(parent, pointStoreView) : null;
            while (!path.isEmpty()) { // otherwise a single node at root
                depth -= 1;
                mass = getMass(parent);
                double prob = (!shadow) ? probabilityOfCut(parent, floatPoint, pointStoreView, boundingBox)
                        : boundingBox.probabilityOfCut(floatPoint);
                if (prob == 0.0) {
                    break;
                }
                scoreValue = (1 - prob) * scoreValue + prob * scoreUnseen.apply(depth, mass);
                sibling = path.pop()[1];
                if (!path.isEmpty()) {
                    parent = path.lastElement()[0];
                    if (boundingBox != null) {
                        growNodeBox(boundingBox, pointStoreView, parent, sibling);
                    }
                }
            }
        }
        return scoreValue;
    }

    public double dynamicScore(int root, int ignoreMass, double[] point, IPointStoreView<float[]> pointStoreView,
            BiFunction<Double, Double, Double> scoreSeen, BiFunction<Double, Double, Double> scoreUnseen,
            Function<Double, Double> treeDamp) {
        if (root == 0) {
            return 0.0;
        }
        BoundingBoxFloat boundingBox = null;
        if (nodeCacheFraction < SWITCH_FRACTION || ignoreMass > 0) {
            float[] fakePoint = new float[point.length];
            boundingBox = new BoundingBoxFloat(fakePoint, fakePoint);
        }
        return scoreScalar(root, 0, boundingBox, ignoreMass, toFloatArray(point), pointStoreView, scoreSeen,
                scoreUnseen, treeDamp)[1];
    }

    public double[] scoreScalar(int node, int depth, BoundingBoxFloat box, int ignoreMass, float[] point,
            IPointStoreView<float[]> pointStoreView, BiFunction<Double, Double, Double> scoreSeen,
            BiFunction<Double, Double, Double> scoreUnseen, Function<Double, Double> treeDamp) {
        if (isLeaf(node)) {
            double mass = getMass(node);
            float[] oldPoint = pointStoreView.get(getPointIndex(node));
            int ignoreFlag = (mass > ignoreMass) ? 1 : 0;
            if (box != null) {
                box.replaceBox(oldPoint);
            }
            if (Arrays.equals(point, oldPoint) && ignoreFlag == 1) {
                return new double[] { 0.0, treeDamp.apply(mass) * scoreSeen.apply(depth * 1.0, mass), ignoreFlag };
            } else {
                return new double[] { 1.0, scoreUnseen.apply(1.0 * depth, mass), ignoreFlag };
            }
        }
        double[] answer;
        int sibling;
        if (leftOf(node, point)) {
            answer = scoreScalar(getLeftIndex(node), depth + 1, box, ignoreMass, point, pointStoreView, scoreSeen,
                    scoreUnseen, treeDamp);
            if (answer[0] != 0.0 && box != null) {
                if (answer[2] == 1) {
                    growNodeBox(box, pointStoreView, node, getRightIndex(node));
                } else {
                    box.copyFrom(getBox(getRightIndex(node), pointStoreView));
                    answer[2] = 1;
                }
            }
        } else {
            answer = scoreScalar(getRightIndex(node), depth + 1, box, ignoreMass, point, pointStoreView, scoreSeen,
                    scoreUnseen, treeDamp);
            if (answer[0] != 0.0 && box != null) {
                if (answer[2] == 1) {
                    growNodeBox(box, pointStoreView, node, getLeftIndex(node));
                } else {
                    box.copyFrom(getBox(getLeftIndex(node), pointStoreView));
                    answer[2] = 1;
                }
            }
        }

        if (answer[0] == 0.0) {
            return answer;
        }

        double prob = (ignoreMass == 0) ? probabilityOfCut(node, point, pointStoreView, box)
                : box.probabilityOfCut(point);
        answer[0] = prob;
        answer[1] = answer[1] * (1.0 - prob) + prob * scoreUnseen.apply(1.0 * depth, 1.0 * getMass(node));
        return answer;
    }

    public abstract int getCutDimension(int index);

    public double getCutValue(int index) {
        return cutValue[index - 1];
    }

    public double getNodeCacheFraction() {
        return nodeCacheFraction;
    }

    protected <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, int root,
            IPointStoreView<float[]> pointStoreView, Function<double[], double[]> projectToTree) {
        AbstractNodeView currentNodeView = new AbstractNodeView(this, pointStoreView, root);
        traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, root, 0);
    }

    protected boolean toLeft(double[] point, int currentNodeOffset) {
        return point[getCutDimension(currentNodeOffset)] <= cutValue[currentNodeOffset - 1];
    }

    BoundingBoxFloat getLeftBox(int index, IPointStoreView<float[]> pointStoreView) {
        return getBox(getLeftIndex(index), pointStoreView);
    }

    BoundingBoxFloat getRightBox(int index, IPointStoreView<float[]> pointStoreView) {
        return getBox(getRightIndex(index), pointStoreView);
    }

    protected <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor,
            AbstractNodeView currentNodeView, int node, int depthOfNode) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), false);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            if (toLeft(point, node)) {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, getLeftIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, getRightIndex(node), false);
            } else {
                traversePathToLeafAndVisitNodes(point, visitor, currentNodeView, getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, getLeftIndex(node), false);
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
    }

    protected <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, int root,
            IPointStoreView<float[]> pointStoreView, Function<double[], double[]> liftToTree) {
        AbstractNodeView currentNodeView = new AbstractNodeView(this, pointStoreView, root);
        traverseTreeMulti(point, visitor, currentNodeView, root, 0);
    }

    protected <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, AbstractNodeView currentNodeView,
            int node, int depthOfNode) {
        if (isLeaf(node)) {
            currentNodeView.setCurrentNode(node, getPointIndex(node), false);
            visitor.acceptLeaf(currentNodeView, depthOfNode);
        } else {
            currentNodeView.setCurrentNodeOnly(node);
            if (visitor.trigger(currentNodeView)) {
                traverseTreeMulti(point, visitor, currentNodeView, getLeftIndex(node), depthOfNode + 1);
                MultiVisitor<R> newVisitor = visitor.newCopy();
                currentNodeView.setCurrentNodeOnly(getRightIndex(node));
                traverseTreeMulti(point, newVisitor, currentNodeView, getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, getLeftIndex(node), false);
                visitor.combine(newVisitor);
            } else if (toLeft(point, node)) {
                traverseTreeMulti(point, visitor, currentNodeView, getLeftIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, getRightIndex(node), false);
            } else {
                traverseTreeMulti(point, visitor, currentNodeView, getRightIndex(node), depthOfNode + 1);
                currentNodeView.updateToParent(node, getLeftIndex(node), false);
            }
            visitor.accept(currentNodeView, depthOfNode);
        }
    }

}
