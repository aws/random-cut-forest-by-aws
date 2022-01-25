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
import static com.amazon.randomcutforest.tree.AbstractNodeStore.Null;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class HyperTree extends RandomCutTree {

    private final Function<IBoundingBoxView, double[]> gVecBuild;

    public Function<IBoundingBoxView, double[]> getgVec() {
        return gVecBuild;
    }

    public static Builder builder() {
        return new Builder();
    }

    protected HyperTree(HyperTree.Builder builder) {
        super(builder);
        this.gVecBuild = builder.gVec;
    }

    public void makeTree(List<Integer> list, int seed) {
        // this function allows a public call, which may be useful someday
        if (list.size() > 0 && list.size() < numberOfLeaves + 1) {
            int[] leftIndex = new int[numberOfLeaves - 1];
            int[] rightIndex = new int[numberOfLeaves - 1];
            Arrays.fill(leftIndex, numberOfLeaves - 1);
            Arrays.fill(rightIndex, numberOfLeaves - 1);
            int[] cutDimension = new int[numberOfLeaves - 1];
            float[] cutValue = new float[numberOfLeaves - 1];
            root = makeTreeInt(list, seed, 0, this.gVecBuild, leftIndex, rightIndex, cutDimension, cutValue);
            nodeStore = AbstractNodeStore.builder().storeSequencesEnabled(false).pointStoreView(pointStoreView)
                    .dimensions(dimension).capacity(numberOfLeaves - 1).leftIndex(leftIndex).rightIndex(rightIndex)
                    .cutDimension(cutDimension).cutValues(cutValue).build();
            // the cuts are specififed; now build tree
            for (int i = 0; i < list.size(); i++) {
                addPoint(list.get(i), 0L);
            }
        } else {
            root = Null;
        }
    }

    private int makeTreeInt(List<Integer> pointList, int seed, int firstFree,
            Function<IBoundingBoxView, double[]> vecBuild, int[] left, int[] right, int[] cutDimension,
            float[] cutValue) {

        if (pointList.size() == 0)
            return Null;

        BoundingBox thisBox = new BoundingBox(pointStoreView.get(pointList.get(0)));
        for (int i = 1; i < pointList.size(); i++) {
            thisBox = (BoundingBox) thisBox.getMergedBox(pointStoreView.get(pointList.get(i)));
        }
        if (thisBox.getRangeSum() <= 0) {
            return pointList.get(0) + nodeStore.getCapacity() + 1;
        }

        Random ring = new Random(seed);
        int leftSeed = ring.nextInt();
        int rightSeed = ring.nextInt();
        Cut cut = getCut(thisBox, ring, vecBuild);

        List<Integer> leftList = new ArrayList<>();
        List<Integer> rightList = new ArrayList<>();

        for (int j = 0; j < pointList.size(); j++) {
            if (nodeStore.leftOf((float) cut.getValue(), cut.getDimension(), pointStoreView.get(pointList.get(j)))) {
                leftList.add(pointList.get(j));
            } else
                rightList.add(pointList.get(j));

        }
        int leftIndex = makeTreeInt(leftList, leftSeed, firstFree + 1, vecBuild, left, right, cutDimension, cutValue);
        int rightIndex = makeTreeInt(rightList, rightSeed, firstFree + leftList.size(), vecBuild, left, right,
                cutDimension, cutValue);
        left[firstFree] = Math.min(leftIndex, numberOfLeaves - 1);
        right[firstFree] = Math.min(rightIndex, numberOfLeaves - 1);
        cutDimension[firstFree] = cut.getDimension();
        cutValue[firstFree] = (float) cut.getValue();
        return firstFree;
    }

    private Cut getCut(IBoundingBoxView bb, Random ring, Function<IBoundingBoxView, double[]> vecSeparation) {
        Random rng = new Random(ring.nextInt());
        double cutf = rng.nextDouble();
        double dimf = rng.nextDouble();
        int td = -1;
        double rangeSum = 0;
        double[] vector = vecSeparation.apply(bb);
        for (int i = 0; i < bb.getDimensions(); i++) {
            vector[i] = (float) vector[i];
            rangeSum += vector[i];
        }

        double breakPoint = dimf * rangeSum;
        float cutValue = 0;
        for (int i = 0; i < bb.getDimensions(); i++) {
            double range = vector[i];
            if (range > 0) {
                if ((breakPoint > 0) && (breakPoint <= range)) {
                    td = i;
                    cutValue = (float) (bb.getMinValue(td) + bb.getRange(td) * cutf);
                    if (cutValue == bb.getMaxValue(td)) {
                        cutValue = (float) bb.getMinValue(td);
                    }
                }
                breakPoint -= range;
            }
        }

        checkArgument(td != -1, "Pivot selection failed.");
        return new Cut(td, cutValue);
    }

    public static class Builder extends RandomCutTree.Builder<Builder> {
        private Function<IBoundingBoxView, double[]> gVec;

        public Builder buildGVec(Function<IBoundingBoxView, double[]> gVec) {
            this.gVec = gVec;
            return this;
        }

        public HyperTree build() {
            return new HyperTree(this);
        }
    }
}
