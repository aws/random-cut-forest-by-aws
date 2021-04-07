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

package com.amazon.randomcutforest.store;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

/**
 * PointStore is a fixed size repository of points, where each point is a float
 * array of a specified length. A PointStore counts references to points that
 * are added, and frees space internally when a given point is no longer in use.
 * The primary use of this store is to enable compression since the points in
 * two different trees do not have to be stored separately.
 *
 * Stored points are referenced by index values which can be used to look up the
 * point values and increment and decrement reference counts. Valid index values
 * are between 0 (inclusive) and capacity (exclusive).
 */
public abstract class PointStore<Store, Point> extends IndexManager implements IPointStore<Point> {

    public static int INFEASIBLE_LOCATION = -1;

    public static int INFEASIBLE_INDEX = -1;
    /**
     * generic store class
     */
    protected Store store;
    /**
     * pointers to store locations
     */
    protected int[] locationList;

    /**
     * refCount[i] counts of the number of trees that are currently using the point
     * determined by locationList[i] or (for directLocationMapping) the point at
     * store[i * dimensions]
     */
    protected short[] refCount;
    /**
     * first location where new data can be safely copied;
     */
    int startOfFreeSegment;
    /**
     * overall dimension of the point (after shingling)
     */
    int dimensions;
    /**
     * shingle size, if known. Setting shingle size = 1 rules out overlapping
     */
    int shingleSize;
    /**
     * number of original dimensions which are shingled to produce and overall point
     * dimensions = shingleSize * baseDimensions. However there is a possibility
     * that even though the data is shingled, we may not choose to use the
     * overlapping (say for out of order updates).
     */
    int baseDimension;
    /**
     * even for shingleSize>1 we may seek to disable certain behavior, for example
     * if we know that the set of points in the store already have low overlap or
     * the order of updates is not in a sequential increasing order
     */
    boolean shingleAwareOverlapping;
    /**
     * are the addresses mapped directly (saves space and runtime for shingleSize =
     * 1)
     */
    boolean directLocationMap;

    /**
     * Create a new PointStore with the given dimensions and capacity.
     *
     * @param dimensions The number of dimensions in stored points.
     * @param capacity   The maximum number of points that can be stored.
     */
    public PointStore(int dimensions, int shingleSize, int capacity, boolean shingleAwareOverlapping,
            boolean directLocationMap) {
        super(capacity);
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        checkArgument(shingleSize == 1 || dimensions % shingleSize == 0, "incorrect use");
        checkArgument(!directLocationMap || !shingleAwareOverlapping,
                " cannot have overlapped shingles and direct map simultaneously");
        this.shingleSize = shingleSize;
        this.dimensions = dimensions;
        refCount = new short[capacity];
        startOfFreeSegment = 0;
        this.directLocationMap = directLocationMap;
        this.shingleAwareOverlapping = shingleAwareOverlapping;
        if (!directLocationMap) {
            if (shingleAwareOverlapping && shingleSize > 1) {
                // even if shingle size is 1
                baseDimension = dimensions / shingleSize;
            } else {
                baseDimension = dimensions;
            }
            locationList = new int[capacity];
            if (!shingleAwareOverlapping) { // initialize a 1-1 map
                for (int i = 0; i < capacity; i++) {
                    locationList[i] = i * dimensions;
                }
            }
        } else {
            baseDimension = dimensions;
        }
    }

    public PointStore(int dimensions, int capacity) {
        this(dimensions, 1, capacity, false, true);
    }

    public PointStore(boolean shingleAwareOverlapping, int dimensions, int shingleSize, short[] refCount,
            int[] referenceList, int[] freeIndexes, int freeIndexPointer) {
        super(freeIndexes, freeIndexPointer);
        checkArgument(dimensions > 0, "dimensions must be greater than 0");
        checkArgument(shingleSize == 1 || dimensions % shingleSize == 0, "incorrect use");
        checkArgument(refCount.length == capacity, "incorrect");
        this.shingleSize = shingleSize;
        this.dimensions = dimensions;
        this.refCount = refCount;
        this.locationList = referenceList;
        this.directLocationMap = (referenceList == null);
        this.shingleAwareOverlapping = shingleAwareOverlapping;
        if (shingleAwareOverlapping && shingleSize > 1) {
            this.baseDimension = this.dimensions / this.shingleSize;
        } else {
            this.baseDimension = dimensions;
        }
        // firstFreeLocation would be set by the concrete classes, along with Store
    }

    /**
     * Decrement the reference count for the given index.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    @Override
    public int decrementRefCount(int index) {
        checkValidIndex(index);

        if (refCount[index] == 1) {
            releaseIndex(index);
        }

        return --refCount[index];
    }

    /**
     * the function checks if the provided shingled point aligns with the location
     * 
     * @param location location in the store where the point is copied
     * @param point
     * @return
     */
    abstract boolean checkShingleAlignment(int location, double[] point);

    /**
     * copy the point sratering from its location src to the location in the store
     * for desired length
     * 
     * @param point    input point
     * @param src      location of the point that is not in a previous shingle
     * @param location location in the store
     * @param length   length to be copied
     */
    abstract void copyPoint(double[] point, int src, int location, int length);

    /**
     * Add a point to the point store and return the index of the stored point.
     *
     * @param point       The point being added to the store.
     * @param sequenceNum sequence number of the point
     * @return the index value of the stored point.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     * @throws IllegalStateException    if the point store is full.
     */
    public int add(double[] point, long sequenceNum) {
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");

        if (shingleSize > 1 && shingleAwareOverlapping) {
            /**
             * corresponds to the shingled/overlapped case. Currently we are considering a
             * perfect sequence where the shift corresponds to one time unit and that
             * represents baseDimension new values. Note that if the alignment test fails,
             * then the points are still copied -- in future it may be reasonable to check
             * for different alignments -- which would be beneficial for larger shingles. In
             * extreme a segment tree type data structure could also be used.
             */
            if (startOfFreeSegment > capacity * dimensions - baseDimension) {
                // the above ensures that the most recent values in a shingle can be writtem
                compact();
            }

            boolean test = (startOfFreeSegment - dimensions + baseDimension >= 0);
            if (test) {
                test = checkShingleAlignment(startOfFreeSegment, point);
            }
            if (test && startOfFreeSegment + baseDimension <= capacity * dimensions) {
                int nextIndex = takeIndex(); // no more compactions
                refCount[nextIndex] = 1; // has to be after compactions
                locationList[nextIndex] = startOfFreeSegment - dimensions + baseDimension;
                copyPoint(point, dimensions - baseDimension, startOfFreeSegment, baseDimension);
                startOfFreeSegment += baseDimension;
                return nextIndex;
            }

            // we must write the full contents of the point; we need to check for space
            if (startOfFreeSegment >= capacity * dimensions - dimensions) {
                compact();
            }
            int nextIndex = takeIndex(); // no more compactions
            locationList[nextIndex] = startOfFreeSegment;
            copyPoint(point, 0, startOfFreeSegment, dimensions);
            startOfFreeSegment += dimensions;
            refCount[nextIndex] = 1; // has to be after compactions
            return nextIndex;
        }
        /**
         * the following corresponds to shingleSize=1 or the no-overlap case. The
         * startOfFreeSegment corresponds to the maximum address written by the store.
         * This is useful for serialization where the suffix of the store can be
         * ignored. For short streams, this will yield significnat benefits. In the long
         * run however, the savings will depend on the data being input.
         */
        int nextIndex = takeIndex(); // no more compactions
        int address = (directLocationMap) ? nextIndex * dimensions : locationList[nextIndex];
        startOfFreeSegment = (startOfFreeSegment < address + dimensions) ? address + dimensions : startOfFreeSegment;
        copyPoint(point, 0, address, dimensions);
        refCount[nextIndex] = 1; // has to be after compactions
        return nextIndex;
    }

    /**
     * Increment the reference count for the given index. This operation assumes
     * that there is currently a point stored at the given index and will throw an
     * exception if that's not the case.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    public int incrementRefCount(int index) {
        checkValidIndex(index);
        return ++refCount[index];
    }

    @Override
    public int getDimensions() {
        return dimensions;
    }

    /**
     * Test whether the given point is equal to the point stored at the given index.
     * This operation uses point-wise <code>==</code> to test for equality.
     *
     * @param index The index value of the point we are comparing to.
     * @param point The point we are comparing for equality.
     * @return true if the point stored at the index is equal to the given point,
     *         false otherwise.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     * @throws IllegalArgumentException if the length of the point does not match
     *                                  the point store's dimensions.
     */

    abstract public boolean pointEquals(int index, Point point);

    /**
     * Get a copy of the point at the given index.
     *
     * @param index An index value corresponding to a storage location in this point
     *              store.
     * @return a copy of the point stored at the given index.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is nonpositive.
     */
    @Override
    abstract public Point get(int index);

    @Override
    abstract public String toString(int index);

    public int getShingleSize() {
        return shingleSize;
    }

    public Store getStore() {
        return store;
    }

    public short[] getRefCount() {
        return refCount;
    }

    public int getStartOfFreeSegment() {
        return startOfFreeSegment;
    }

    public int[] getLocationList() {
        return locationList;
    }

    public boolean isShingleAwareOverlapping() {
        return shingleAwareOverlapping;
    }

    public boolean isDirectLocationMap() {
        return directLocationMap;
    }

    /**
     * a simple optimization that identifies the prefix of the arrays (refCount,
     * referenceList) that are being used
     * 
     * @return size of initial prefix in use
     */
    public int getValidPrefix() {
        int prefix = capacity;
        while (prefix > 0 && !occupied.get(prefix - 1)) {
            prefix--;
        }
        return prefix;
    }

    abstract void copyTo(int dest, int source, int length);

    /**
     * The following function eliminates redundant information that builds up in the
     * pointstore and shrinks the pointstore
     */

    public void compact() {
        checkArgument(!directLocationMap, "incorrect call; should not be used for direct location maps");
        int DEFAULT_EMPTY = 0; // indicates location not in use

        int runningLocation = 0;
        startOfFreeSegment = 0;
        int stepDimension = (shingleAwareOverlapping) ? baseDimension : dimensions;

        // we first determine which locations are the startpoints of the shingles
        int[] reverseReferenceList = new int[capacity * dimensions / stepDimension];
        for (int i = 0; i < capacity; i++) {
            if (occupied.get(i)) {
                reverseReferenceList[locationList[i] / stepDimension] = i + 1;
                // offset by 1, to distinguish DEFAULT_EMPTY
            }
        }

        // we make a pass over the data
        while (runningLocation < capacity * dimensions) {
            // find the first eligible shoingle to be copied
            while (runningLocation < capacity * dimensions
                    && reverseReferenceList[runningLocation / stepDimension] == DEFAULT_EMPTY) {
                runningLocation += stepDimension;
            }

            /**
             * recursively keep copying; if a new relevant shingle is found during the
             * copying, the remainsToBeCopied is updated to dimensions
             */
            if (runningLocation < capacity * dimensions) {
                int remainsToBeCopied = dimensions;
                int saveLocation = runningLocation;
                while (runningLocation < capacity * dimensions && remainsToBeCopied > 0) {
                    if (stepDimension == 1 || runningLocation % stepDimension == 0) {
                        checkArgument(stepDimension == 1 || startOfFreeSegment % stepDimension == 0, "error");
                        reverseReferenceList[startOfFreeSegment / stepDimension] = reverseReferenceList[runningLocation
                                / stepDimension];
                        if (reverseReferenceList[runningLocation / stepDimension] != DEFAULT_EMPTY) {
                            checkArgument(locationList[reverseReferenceList[runningLocation / stepDimension]
                                    - 1] == runningLocation, "error");
                            locationList[reverseReferenceList[startOfFreeSegment / stepDimension]
                                    - 1] = startOfFreeSegment;
                            remainsToBeCopied = dimensions;
                            if (runningLocation > startOfFreeSegment) {
                                reverseReferenceList[runningLocation / stepDimension] = DEFAULT_EMPTY;
                            }
                        }
                    }
                    runningLocation++;
                    remainsToBeCopied--;
                }
                copyTo(startOfFreeSegment, saveLocation, runningLocation - saveLocation);
                startOfFreeSegment += runningLocation - saveLocation;
            }
        }
        if (!shingleAwareOverlapping) {
            /**
             * need to restore a 1-1 map; note that the above block would work for
             * shingleSize=1
             */
            int tempLocation = startOfFreeSegment;
            for (int i = 0; i < freeIndexPointer; i++) {
                if (!occupied.get(freeIndexes[i])) {
                    locationList[freeIndexes[i]] = tempLocation;
                    tempLocation += dimensions;
                }
            }
        }

    }

    public short getRefCount(int i) {
        return refCount[i];
    }
}
