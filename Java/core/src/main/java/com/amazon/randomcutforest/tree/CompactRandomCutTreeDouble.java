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

import static com.amazon.randomcutforest.CommonUtils.*;

import java.util.Arrays;
import java.util.Random;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.MultiVisitor;
import com.amazon.randomcutforest.Visitor;
import com.amazon.randomcutforest.executor.Sequential;
import com.amazon.randomcutforest.store.*;

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
public class CompactRandomCutTreeDouble implements ITree<Integer> {

    /**
     * The index value used to represent the absence of a node. For example, when
     * the tree is created the root node index will be NULL. After a point is added
     * and a root node is created, the root node's parent will be NULL, and so on.
     */
    public static final short NULL = -1;

    private final Random random;
    private final int maxSize;
    protected final NodeStore internalNodes;
    protected final LeafStore leafNodes;
    protected final IPointStore<double[]> pointStore;
    protected short rootIndex;

    public CompactRandomCutTreeDouble(int maxSize, long seed, IPointStore<double[]> pointStore) {
        checkArgument(maxSize > 0, "maxSize must be greater than 0");
        checkNotNull(pointStore, "pointStore must not be null");
        this.maxSize = maxSize;
        this.pointStore = pointStore;
        internalNodes = new NodeStore((short) (this.maxSize - 1));
        leafNodes = new LeafStore((short) this.maxSize);
        random = new Random(seed);
        rootIndex = NULL;
    }

    protected boolean isLeaf(short index) {
        return index - maxSize >= 0;
    }

    /**
     * Return a new {@link Cut}, which is chosen uniformly over the space of
     * possible cuts for the given bounding box.
     *
     * @param random A random number generator
     * @param box    A bounding box that we want to find a random cut for.
     * @return A new Cut corresponding to a random cut in the bounding box.
     */
    static Cut randomCut(Random random, BoundingBox box) {
        double rangeSum = box.getRangeSum();
        checkArgument(rangeSum > 0, "box.getRangeSum() must be greater than 0");

        double breakPoint = random.nextDouble() * rangeSum;

        for (int i = 0; i < box.getDimensions(); i++) {
            double range = box.getRange(i);
            if (breakPoint <= range) {
                double cutValue = box.getMinValue(i) + breakPoint;

                // Random cuts have to take a value in the half-open interval [minValue,
                // maxValue) to ensure that a
                // Node has a valid left child and right child.
                if ((cutValue == box.getMaxValue(i)) && (box.getMinValue(i) < box.getMaxValue(i))) {
                    cutValue = Math.nextAfter(box.getMaxValue(i), box.getMinValue(i));
                }

                return new Cut(i, cutValue);
            }
            breakPoint -= range;
        }

        throw new IllegalStateException("The break point did not lie inside the expected range");
    }

    /** spoof a leaf */
    Node getLeafNode(int index) {
        int leafOffset = index - maxSize;
        double[] leafPoint = pointStore.get(leafNodes.pointIndex[leafOffset]);
        Node leaf = newLeafNode(leafPoint, -1);
        leaf.setMass(leafNodes.mass[leafOffset]);
        return leaf;
    }

    /** spoof a node */
    Node getInternalNode(int index) {
        return newNode(null, null, new Cut(internalNodes.cutDimension[index], internalNodes.cutValue[index]),
                internalNodes.boundingBox[index]);
    }

    /**
     * Replace a Node in the tree structure. This method replaces oldNode with
     * newNode as a child of oldNode.getParent().
     *
     * @param oldNodeOffset The node we are replacing.
     * @param newNodeOffset The new node we are inserting into the tree.
     */

    void replaceNode(short oldNodeOffset, short newNodeOffset) {

        short parent;
        if (isLeaf(oldNodeOffset)) {
            parent = leafNodes.parentIndex[oldNodeOffset - maxSize];
        } else {
            parent = internalNodes.parentIndex[oldNodeOffset];
        }

        if (parent != NULL) {
            checkState(parent < maxSize, "incorrect");
            if (internalNodes.leftIndex[parent] == oldNodeOffset) {
                internalNodes.leftIndex[parent] = newNodeOffset;
            } else {
                internalNodes.rightIndex[parent] = newNodeOffset;
            }
        }

        if (isLeaf(newNodeOffset)) {
            leafNodes.parentIndex[newNodeOffset - maxSize] = parent;
        } else {
            internalNodes.parentIndex[newNodeOffset] = parent;
        }
    }

    /**
     * Return the sibling of a non-root node. Note that every non-leaf node in a
     * Random Cut Tree has two children.
     *
     * @param nodeOffset The node whose sibling we are requesting.
     * @return the sibling of node in the tree.
     */

    short getSibling(short nodeOffset) {
        checkArgument(nodeOffset >= 0, " cannotbe negative");
        int parent;
        if (isLeaf(nodeOffset)) {
            parent = leafNodes.parentIndex[nodeOffset - maxSize];
        } else {
            parent = internalNodes.parentIndex[nodeOffset];
        }

        if (parent != NULL) {
            checkArgument(parent < maxSize, "incorrect");
            if (internalNodes.leftIndex[parent] == nodeOffset) {
                return internalNodes.rightIndex[parent];
            } else if (internalNodes.rightIndex[parent] == nodeOffset) {
                return internalNodes.leftIndex[parent];
            } else {
                throw new IllegalArgumentException("node parent does not link back to node");
            }
        } else
            return NULL; // root is leaf
    }

    protected boolean leftOf(float[] point, short nodeOffset) {
        return leftOf(CommonUtils.toDoubleArray(point), nodeOffset);
    }

    protected boolean leftOf(double[] point, short nodeOffset) {
        checkArgument(!isLeaf(nodeOffset), " error ");
        return (point[internalNodes.cutDimension[nodeOffset]] <= internalNodes.cutValue[nodeOffset]);
    }

    /**
     * Delete the given point from this tree.
     *
     * @param sequential An offset of the point in the tree that we wish to delete.
     */
    @Override
    public void deletePoint(Sequential<Integer> sequential) {
        checkState(rootIndex != NULL, "root must not be null");
        deletePoint(rootIndex, pointStore.get(sequential.getValue()), 0);
    }

    /**
     * This function deletes the point from the tree recursively. We traverse the
     * tree based on the cut stored in each interior node until we reach a leaf
     * node. We then delete the leaf node if the mass of the node is 1, otherwise we
     * reduce the mass by 1.
     *
     * @param nodeOffset node that we are visiting in the tree.
     * @param point      the point that is being deleted from the tree.
     * @param level      the level (i.e., the length of the path to the root) of the
     *                   node being evaluated.
     * @return the index of the deleted node.
     */
    private int deletePoint(short nodeOffset, double[] point, int level) {

        if (isLeaf(nodeOffset)) {
            short leafOffset = (short) (nodeOffset - maxSize);
            if (!pointStore.pointEquals(leafNodes.pointIndex[leafOffset], point)) {
                throw new IllegalStateException(
                        Arrays.toString(point) + " " + Arrays.toString(pointStore.get(leafNodes.pointIndex[leafOffset]))
                                + " " + leafOffset + " node " + leafNodes.pointIndex[leafOffset] + " " + false
                                + " Inconsistency in trees in delete step here.");
            }

            //
            // the above assumes that sequence indexes are unique ... which is true for the
            // specific sampler used
            //

            if (leafNodes.mass[leafOffset] > 1) {
                --leafNodes.mass[leafOffset];
                return leafNodes.pointIndex[leafOffset];
            }

            short parent = leafNodes.parentIndex[leafOffset];
            int saved = leafNodes.pointIndex[leafOffset];
            if (parent == NULL) {
                rootIndex = NULL;
                leafNodes.delete(leafOffset);
                return saved;
            }

            int grandParent = internalNodes.parentIndex[parent];
            if (grandParent == NULL) {
                rootIndex = getSibling(nodeOffset);
                if (isLeaf(rootIndex)) {
                    leafNodes.parentIndex[rootIndex - maxSize] = NULL;
                } else {
                    internalNodes.parentIndex[rootIndex] = NULL;
                }
            } else {
                short sibling = getSibling(nodeOffset);
                replaceNode(parent, sibling);
            }
            leafNodes.delete(leafOffset);
            internalNodes.delete(parent);
            return saved;
        }

        // node is not a leaf, and is an internal node
        int ret;
        if (leftOf(point, nodeOffset)) {
            ret = deletePoint(internalNodes.leftIndex[nodeOffset], point, level + 1);
        } else {
            ret = deletePoint(internalNodes.rightIndex[nodeOffset], point, level + 1);
        }

        BoundingBox leftBox;
        if (isLeaf(internalNodes.leftIndex[nodeOffset])) {
            double[] leafPoint = pointStore.get(leafNodes.pointIndex[internalNodes.leftIndex[nodeOffset] - maxSize]);
            leftBox = new BoundingBox(leafPoint);
        } else {
            leftBox = internalNodes.boundingBox[internalNodes.leftIndex[nodeOffset]];
        }

        BoundingBox rightBox;
        if (isLeaf(internalNodes.rightIndex[nodeOffset])) {
            double[] leafPoint = pointStore.get(leafNodes.pointIndex[internalNodes.rightIndex[nodeOffset] - maxSize]);
            rightBox = new BoundingBox(leafPoint);
        } else {
            rightBox = internalNodes.boundingBox[internalNodes.rightIndex[nodeOffset]];
        }

        internalNodes.boundingBox[nodeOffset] = leftBox.getMergedBox(rightBox);
        --internalNodes.mass[nodeOffset];
        return ret;
    }

    public Integer addPoint(Sequential<Integer> seq) {
        int pointIndex = seq.getValue();
        double[] pointValue = pointStore.get(pointIndex);
        if (rootIndex == NULL) {
            rootIndex = (short) (leafNodes.add(NULL, pointIndex, 1) + maxSize);
            return pointIndex;
        } else {
            return addPoint(rootIndex, pointValue, pointIndex);
        }
    }

    /**
     * This function adds a point to the tree recursively. The algorithm for adding
     * a point is as follows:
     * <ol>
     * <li>At the current node we create a new bounding box by merging the point
     * with the existing box.</li>
     * <li>We pick a dimension and a value of cut.</li>
     * <li>If the cut falls outside the existing box we create a new node and
     * replace the current node with the new node. We then add the current node as
     * the child node to the new node. We create another leaf node containing the
     * point that we want to insert and add it as the other child of the node that
     * we have created earlier.</li>
     * <li>If the cut falls inside the existing box. We follow the cut of the
     * existing box and move to right or left child of the current node based on the
     * existing cut.</li>
     * </ol>
     *
     * @param nodeOffset the current node in the tree we are on
     * @param point      the point that we want to add to the tree
     * @param pointIndex is the location of the point in pointstore
     */

    private int addPoint(short nodeOffset, double[] point, int pointIndex) {
        if (isLeaf(nodeOffset) && pointStore.pointEquals(leafNodes.pointIndex[nodeOffset - maxSize], point)) {
            // the inserted point is equal to an existing leaf point
            ++leafNodes.mass[nodeOffset - maxSize];
            return leafNodes.pointIndex[nodeOffset - maxSize];
        }

        // either the node is not a leaf, or else it's a leaf node containing a
        // different point

        double[] dpoint = point;

        BoundingBox existingBox = (isLeaf(nodeOffset))
                ? new BoundingBox(pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize]))
                : internalNodes.boundingBox[nodeOffset];
        BoundingBox mergedBox = existingBox.getMergedBox(dpoint);

        if (!existingBox.contains(dpoint)) {

            // Propose a random cut

            Cut cut = randomCut(random, mergedBox);
            int splitDimension = cut.getDimension();
            double splitValue = cut.getValue();
            double minValue = existingBox.getMinValue(splitDimension);
            double maxValue = existingBox.getMaxValue(splitDimension);

            // if the proposed cut separates the new point from the existing bounding box:
            // * create a new leaf node for the point
            // * make it a sibling of the existing bounding box
            // * make the new leaf node and the existing node children of a new node with
            // the merged bounding box

            if (minValue > splitValue || maxValue <= splitValue) {
                int oldmass = (isLeaf(nodeOffset)) ? leafNodes.mass[nodeOffset - maxSize]
                        : internalNodes.mass[nodeOffset];
                short leafOffset = (short) (leafNodes.add(NULL, pointIndex, 1) + maxSize); // parent pointIndex
                                                                                           // needs to be fixed
                short mergedNode = (minValue > splitValue)
                        ? internalNodes.addNode(NULL, leafOffset, nodeOffset, splitDimension, splitValue, oldmass + 1)
                        : internalNodes.addNode(NULL, nodeOffset, leafOffset, splitDimension, splitValue, oldmass + 1);

                // uncomment on for state verification in addPoint
                // if (minValue> splitValue){
                // checkState(verify(nodeOffset,splitDimension,splitValue,false),"right
                // incorrect");
                // } else {
                // checkState(verify(nodeOffset,splitDimension,splitValue,true)," left
                // incorrect");
                // }

                int parent;
                if (isLeaf(nodeOffset))
                    parent = leafNodes.parentIndex[nodeOffset - maxSize];
                else
                    parent = internalNodes.parentIndex[nodeOffset];

                if (parent == NULL) {
                    rootIndex = mergedNode;
                } else {
                    replaceNode(nodeOffset, mergedNode);
                }
                leafNodes.parentIndex[leafOffset - maxSize] = mergedNode;
                if (isLeaf(nodeOffset))
                    leafNodes.parentIndex[nodeOffset - maxSize] = mergedNode;
                else
                    internalNodes.parentIndex[nodeOffset] = mergedNode;

                internalNodes.boundingBox[mergedNode] = mergedBox;

                return pointIndex;
            }
        }

        // Either the new point is contained in this node's bounding box, or else the
        // proposed cut did not separate
        // it from the existing bounding box. Try again at the next level.

        int ret;
        if (leftOf(point, nodeOffset)) {
            ret = addPoint(internalNodes.leftIndex[nodeOffset], point, pointIndex);
        } else {
            ret = addPoint(internalNodes.rightIndex[nodeOffset], point, pointIndex);
        }

        internalNodes.boundingBox[nodeOffset] = mergedBox;
        ++internalNodes.mass[nodeOffset];
        return ret; // the above can be rearranged for tail recursion
    }

    boolean verify(short nodeOffset, int dimension, double cutValue, boolean allSmall) {
        if (isLeaf(nodeOffset)) {
            if (pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize])[dimension] < cutValue) {
                if (!allSmall) {
                    System.out.println("ERROR Small "
                            + pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize])[dimension] + " " + cutValue);
                }
                return (allSmall);
            } else {
                if (allSmall) {
                    System.out.println("ERROR large "
                            + pointStore.get(leafNodes.pointIndex[nodeOffset - maxSize])[dimension] + " " + cutValue);
                }
                return (!allSmall);
            }
        } else {
            boolean answer = verify(internalNodes.leftIndex[nodeOffset], dimension, cutValue, allSmall);
            if (answer == false)
                return false;
            return verify(internalNodes.rightIndex[nodeOffset], dimension, cutValue, allSmall);
        }
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
     * @param point   A point which determines the traversal path from the root to a
     *                leaf node.
     * @param visitor A visitor that will be invoked for each node on the path.
     * @param <R>     The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    @Override
    public <R> R traverse(double[] point, Visitor<R> visitor) {
        checkState(rootIndex != NULL, "this tree doesn't contain any nodes");
        traversePathToLeafAndVisitNodes(point, visitor, rootIndex, 0);
        return visitor.getResult();
    }

    private <R> void traversePathToLeafAndVisitNodes(double[] point, Visitor<R> visitor, short currentNode,
            int depthOfNode) {
        if (isLeaf(currentNode)) {
            visitor.acceptLeaf(getLeafNode(currentNode), depthOfNode);
        } else {
            short childNode = leftOf(point, currentNode) ? internalNodes.leftIndex[currentNode]
                    : internalNodes.rightIndex[currentNode];

            traversePathToLeafAndVisitNodes(point, visitor, childNode, depthOfNode + 1);

            Node node = getInternalNode(currentNode);

            if (!isLeaf(internalNodes.leftIndex[currentNode])) {
                node.setLeftChild(getInternalNode(internalNodes.leftIndex[currentNode]));
            } else {
                node.setLeftChild(getLeafNode(internalNodes.leftIndex[currentNode]));
            }
            if (!isLeaf(internalNodes.rightIndex[currentNode])) {
                node.setRightChild(getInternalNode(internalNodes.rightIndex[currentNode]));
            } else {
                node.setRightChild(getLeafNode(internalNodes.rightIndex[currentNode]));
            }

            visitor.accept(node, depthOfNode);
        }
    }

    /**
     * This is a traversal method which follows the standard traveral path (defined
     * in {@link #traverse(double[], Visitor)}) but at Node in checks to see whether
     * the visitor should split. If a split is triggered, then independent copies of
     * the visitor are sent down each branch of the tree and then merged before
     * propogating the result.
     *
     * @param point   A point which determines the traversal path from the root to a
     *                leaf node.
     * @param visitor A visitor that will be invoked for each node on the path.
     * @param <R>     The return type of the Visitor.
     * @return the value of {@link Visitor#getResult()}} after the traversal.
     */
    /*
     * public <R> R traverseTreeMulti(double[] point, MultiVisitor<R> visitor) {
     * checkNotNull(point, "point must not be null"); checkNotNull(visitor,
     * "visitor must not be null"); checkState(root != null,
     * "this tree doesn't contain any nodes"); traverseTreeMulti(point, visitor,
     * root, 0); return visitor.getResult(); }
     */

    @Override
    public <R> R traverseMulti(double[] point, MultiVisitor<R> visitor) {
        checkNotNull(point, "point must not be null");
        checkNotNull(visitor, "visitor must not be null");
        checkState(rootIndex != NULL, "this tree doesn't contain any nodes");
        traverseTreeMulti(point, visitor, rootIndex, 0);
        return visitor.getResult();
    }

    private <R> void traverseTreeMulti(double[] point, MultiVisitor<R> visitor, short currentNode, int depthOfNode) {
        if (isLeaf(currentNode)) {
            visitor.acceptLeaf(getLeafNode(currentNode), depthOfNode);
        } else if (visitor.trigger(getInternalNode(currentNode))) {
            traverseTreeMulti(point, visitor, internalNodes.leftIndex[currentNode], depthOfNode + 1);
            MultiVisitor<R> newVisitor = visitor.newCopy();
            traverseTreeMulti(point, newVisitor, internalNodes.rightIndex[currentNode], depthOfNode + 1);
            visitor.combine(newVisitor);
            visitor.accept(getInternalNode(currentNode), depthOfNode);
        } else {
            short childNode = leftOf(point, currentNode) ? internalNodes.leftIndex[currentNode]
                    : internalNodes.rightIndex[currentNode];
            traverseTreeMulti(point, visitor, childNode, depthOfNode + 1);
            visitor.accept(getInternalNode(currentNode), depthOfNode);
        }
    }

    public Node getRoot() {
        if (rootIndex == NULL)
            return null;
        else if (isLeaf(rootIndex))
            return getLeafNode(rootIndex);
        else
            return getInternalNode(rootIndex);
    }

    @Override
    public int getMass() {
        if (rootIndex == NULL)
            return 0;
        else if (isLeaf(rootIndex))
            return leafNodes.mass[rootIndex - maxSize];
        else
            return internalNodes.mass[rootIndex];
    }

    private Node newLeafNode(double[] point, long sequenceIndex) {
        Node node = new Node(point);
        node.setMass(1);

        return node;
    }

    private Node newNode(Node leftChild, Node rightChild, Cut cut, BoundingBox box) {
        Node node = new Node(leftChild, rightChild, cut, box, false);

        return node;
    }

    public void reflateTree() {
        if ((rootIndex == NULL) || (rootIndex >= maxSize))
            return;
        reflateNode(rootIndex);
    }

    private BoundingBox reflateNode(int offset) {
        if (offset >= maxSize) {
            double[] leafPoint = pointStore.get(leafNodes.pointIndex[offset - maxSize]);
            return new BoundingBox(leafPoint);
        }
        internalNodes.boundingBox[offset] = reflateNode(internalNodes.leftIndex[offset])
                .getMergedBox(reflateNode(internalNodes.rightIndex[offset]));
        return internalNodes.boundingBox[offset];
    }

    NodeStoreData getNodes() {
        return new NodeStoreData(internalNodes);
    }

    LeafStoreData getLeaves() {
        return new LeafStoreData(leafNodes);
    }

    public void reInitialize(TreeData treeData) {
        rootIndex = treeData.rootIndex;
        leafNodes.reInitialize(treeData.leafData);
        internalNodes.reInitialize(treeData.nodeData);
        reflateTree();
    }
}
