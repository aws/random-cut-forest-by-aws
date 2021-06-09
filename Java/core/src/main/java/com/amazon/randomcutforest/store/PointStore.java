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
import static com.amazon.randomcutforest.CommonUtils.checkState;
import static com.amazon.randomcutforest.CommonUtils.validateInternalState;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;

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
public abstract class PointStore<Store, Point> implements IPointStore<Point> {

    public static int INFEASIBLE_POINTSTORE_LOCATION = -2;

    public static int INFEASIBLE_POINTSTORE_INDEX = -1;
    /**
     * an index manager to manage free locations
     */
    protected IndexManager indexManager;
    /**
     * generic store class
     */
    protected Store store;
    /**
     * generic internal shingle, note that input is doubles
     */
    protected double[] internalShingle;
    /**
     * last seen timestamp for internal shingling
     */
    protected long nextSequenceIndex;
    /**
     * pointers to store locations, this decouples direct addressing and points can
     * be moved internally
     */
    protected int[] locationList;

    /**
     * refCount[i] counts of the number of trees that are currently using the point
     * determined by locationList[i] or (for directLocationMapping) the point at
     * store[i * dimensions]
     */
    protected int[] refCount;
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
     * are the addresses mapped directly (saves space and runtime for shingleSize =
     * 1)
     */
    boolean directLocationMap;
    /**
     * maximum capacity
     */
    int capacity;
    /**
     * current capacity of store (number of shingled points)
     */
    int currentStoreCapacity;
    /**
     * ability to resize the store dynamically
     */
    boolean dynamicResizingEnabled;
    /**
     * enabling internal shingling
     */
    boolean internalShinglingEnabled;
    /**
     * enable rotation of shingles; use a cyclic buffer instead of sliding window
     */
    boolean rotationEnabled;

    /**
     * Decrement the reference count for the given index.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non positive.
     */
    @Override
    public int decrementRefCount(int index) {
        indexManager.checkValidIndex(index);
        checkState(refCount[index] >= 0, " incorrect state ");
        if (refCount[index] == 1) {
            indexManager.releaseIndex(index);
        }

        return --refCount[index];
    }

    /**
     * the function checks if the provided shingled point aligns with the location
     * 
     * @param location location in the store where the point is copied
     * @param point    the point to be added
     * @return true/false for an alignment
     */
    abstract boolean checkShingleAlignment(int location, double[] point);

    /**
     * copy the point starting from its location src to the location in the store
     * for desired length
     * 
     * @param point    input point
     * @param src      location of the point that is not in a previous shingle
     * @param location location in the store
     * @param length   length to be copied
     */
    abstract void copyPoint(double[] point, int src, int location, int length);

    /**
     * takes an index from the index manager and rezises if necessary also adjusts
     * refCount size to have increment/decrement be seamless
     *
     * @return an index from the index manager
     */
    int takeIndex() {
        if (indexManager.isFull()) {
            if (indexManager.getCapacity() < capacity) {
                int oldCapacity = indexManager.getCapacity();
                int newCapacity = Math.min(capacity, 2 * oldCapacity);
                indexManager = new IndexManager(indexManager, newCapacity);
                validateInternalState(refCount.length == oldCapacity, " incorrect state ");
                refCount = Arrays.copyOf(refCount, newCapacity);
                validateInternalState(locationList.length == oldCapacity, " incorrect state ");
                locationList = Arrays.copyOf(locationList, newCapacity);
                for (int i = oldCapacity; i < newCapacity; i++) {
                    locationList[i] = INFEASIBLE_POINTSTORE_LOCATION;
                }
            } else {
                throw new IllegalStateException(" index manager in point store is full ");
            }
        }
        int location = indexManager.takeIndex();
        validateInternalState(refCount[location] == 0, " error");
        return location;
    }

    void verifyAndMakeSpace(int amount) {
        validateInternalState(amount <= dimensions, "incorrect parameter");
        if (startOfFreeSegment > currentStoreCapacity * dimensions - amount) {
            // try compaction and then resizing
            compact();
            if (startOfFreeSegment > currentStoreCapacity * dimensions - amount) {
                checkState(dynamicResizingEnabled, " out of store, enable dynamic resizing ");
                resizeStore();
                validateInternalState(startOfFreeSegment + amount <= currentStoreCapacity * dimensions, "out of space");
            }
        }
    }

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
        checkArgument(internalShinglingEnabled || point.length == dimensions,
                "point.length must be equal to dimensions");
        checkArgument(!internalShinglingEnabled || point.length == baseDimension,
                "point.length must be equal to dimensions");

        double[] tempPoint = point;
        if (internalShinglingEnabled) {
            tempPoint = changeShingleInPlace(internalShingle, point);
            nextSequenceIndex++;
            if (nextSequenceIndex < shingleSize) {
                return INFEASIBLE_POINTSTORE_INDEX;
            }
        }
        int nextIndex;
        if (!directLocationMap) {
            // suppose there was shingling then only the contents of the most recent
            // point has to be written, otherwise we need to write the full shingle
            int amountToWrite = checkShingleAlignment(startOfFreeSegment, tempPoint) ? baseDimension : dimensions;

            verifyAndMakeSpace(amountToWrite);
            nextIndex = takeIndex();
            locationList[nextIndex] = startOfFreeSegment - dimensions + amountToWrite;
            copyPoint(tempPoint, dimensions - amountToWrite, startOfFreeSegment, amountToWrite);
            startOfFreeSegment += amountToWrite;
        } else {
            nextIndex = takeIndex();
            int address = locationList[nextIndex];
            validateInternalState(refCount[nextIndex] == 0, "incorrect state");
            if (address == INFEASIBLE_POINTSTORE_LOCATION) {
                if (startOfFreeSegment + dimensions > currentStoreCapacity * dimensions) {
                    checkState(dynamicResizingEnabled, " out of store, enable dynamic resizing ");
                    resizeStore();
                    if (startOfFreeSegment + dimensions > currentStoreCapacity * dimensions) {
                        indexManager.releaseIndex(nextIndex); // put back the last index
                        compact();
                        nextIndex = takeIndex();
                    }
                }
                locationList[nextIndex] = address = startOfFreeSegment;
                startOfFreeSegment = Math.max(startOfFreeSegment, address + dimensions);
            }
            copyPoint(tempPoint, 0, address, dimensions);
        }
        refCount[nextIndex] = 1; // has to be after compactions
        return nextIndex;
    }

    abstract void resizeStore();

    /**
     * Increment the reference count for the given index. This operation assumes
     * that there is currently a point stored at the given index and will throw an
     * exception if that's not the case.
     *
     * @param index The index value.
     * @throws IllegalArgumentException if the index value is not valid.
     * @throws IllegalArgumentException if the current reference count for this
     *                                  index is non positive.
     */
    public int incrementRefCount(int index) {
        indexManager.checkValidIndex(index);
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
     *                                  index is non positive.
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
     *                                  index is non positive.
     */
    @Override
    abstract public Point get(int index);

    /**
     * to print error messages
     * 
     * @param index index of the point in the store
     * @return string corresponding to the point
     */
    @Override
    abstract public String toString(int index);

    /**
     * maximum capacity, in number of points of size dimensions
     */
    public int getCapacity() {
        return capacity;
    }

    /**
     * capacity of the indices
     */
    public int getIndexCapacity() {
        return indexManager.getCapacity();
    }

    /**
     * used in mapper
     * 
     * @return gets the shingle size (if known, otherwise is 1)
     */
    public int getShingleSize() {
        return shingleSize;
    }

    /**
     * gets the current store capacity in the number of points with dimension many
     * values
     * 
     * @return capacity in number of points
     */
    public int getCurrentStoreCapacity() {
        return currentStoreCapacity;
    }

    /**
     * used for mappers
     * 
     * @return the store that stores the values
     */
    public Store getStore() {
        return store;
    }

    /**
     * used for mapper
     * 
     * @return the array of counts referring to different points
     */
    public int[] getRefCount() {
        return refCount;
    }

    /**
     * useful in mapper to not copy
     * 
     * @return the length of the prefix
     */
    public int getStartOfFreeSegment() {
        return startOfFreeSegment;
    }

    /**
     * used in mapper
     * 
     * @return the list of locations where points are stored
     */
    public int[] getLocationList() {
        return locationList;
    }

    /**
     * used in mapper
     * 
     * @return if shingling is performed internally
     */
    public boolean isInternalShinglingEnabled() {
        return internalShinglingEnabled;
    }

    /**
     *
     * @return if the shingles performed internally are rotated as in a cyclic
     *         buffer
     */
    public boolean isInternalRotationEnabled() {
        return rotationEnabled;
    }

    /**
     * used in mapper and in extrapolation
     * 
     * @return the last timestamp seen
     */
    public long getNextSequenceIndex() {
        return nextSequenceIndex;
    }

    /**
     * used in mapper
     * 
     * @return ability to start from a small size an increase the store
     */
    public boolean isDynamicResizingEnabled() {
        return dynamicResizingEnabled;
    }

    /**
     * used in mapper, as well as an optimization for shingle size 1
     * 
     * @return is locationList being used
     */
    public boolean isDirectLocationMap() {
        return directLocationMap;
    }

    /**
     * used to obtain the most recent shingle seen so far in case of internal
     * shingling
     * 
     * @return for internal shingling, returns the last seen shingle
     */
    public double[] getInternalShingle() {
        return (internalShinglingEnabled) ? Arrays.copyOf(internalShingle, dimensions) : null;
    }

    /**
     * used for serialization
     * 
     * @return the free indices as a stack 9can be implicitly defined)
     */
    public int[] getFreeIndexes() {
        return indexManager.getFreeIndexes();
    }

    /**
     * used for serialization
     * 
     * @return the pointer to the stack of free indices
     */
    public int getFreeIndexPointer() {
        return indexManager.getFreeIndexPointer();
    }

    /**
     *
     * @return the number of indices stored
     */
    public int size() {
        return indexManager.capacity - indexManager.freeIndexPointer - 1;
    }

    /**
     * a simple optimization that identifies the prefix of the arrays (refCount,
     * referenceList) that are being used
     * 
     * @return size of initial prefix in use
     */
    public int getValidPrefix() {
        return indexManager.occupied.previousSetBit(capacity) + 1;
    }

    /**
     * copy function for the store
     * 
     * @param dest   location to move to
     * @param source moving from
     * @param length number of values copied
     */
    abstract void copyTo(int dest, int source, int length);

    /**
     * the following function returns the locations which are in use; note that it
     * adjusts for multivariate input when baseDimension is greater than 1.
     * 
     * @return the bitset corresponding to the locations in use
     */

    BitSet inUse() {
        BitSet result = new BitSet(currentStoreCapacity * dimensions / baseDimension);

        for (int i = 0; i < indexManager.capacity; i++) {
            if (indexManager.occupied.get(i)) {
                validateInternalState(refCount[i] > 0, " bits out of sync when count = 0 ");
                validateInternalState(locationList[i] != PointStore.INFEASIBLE_POINTSTORE_INDEX,
                        " incorrect location info");
                result.set(locationList[i] / baseDimension);
            } else {
                validateInternalState(refCount[i] == 0, "bits out of sync with count > 0 ");
                locationList[i] = INFEASIBLE_POINTSTORE_LOCATION;
            }
        }
        return result;
    }

    /**
     * The following function eliminates redundant information that builds up in the
     * point store and shrinks the point store
     */

    public void compact() {

        int runningLocation = 0;
        startOfFreeSegment = 0;

        // we first determine which locations are the start points of the shingles
        // since the shingles extend for a length and can overlap this help define the
        // region that should be copied

        HashMap<Integer, Integer> movedTo = new HashMap<>();

        // the bit set corresponds to the locations that can be in use over the actual
        // store array
        // this is not the same as the number of points that can be stored
        BitSet inUse = inUse();

        // we make a pass over the store data
        while (runningLocation < currentStoreCapacity * dimensions) {
            // find the first eligible shingle to be copied
            // for rotationEnabled, this should be a multiple of dimensions

            runningLocation = inUse.nextSetBit(runningLocation / baseDimension) * baseDimension;
            if (runningLocation < 0) { // no next bit set
                runningLocation = currentStoreCapacity * dimensions;
            }

            // we are now at the start of the data but for rotation enabled internal
            // shingling
            // we need to ensure that locations remain a multiple of dimensions

            if (rotationEnabled && runningLocation % dimensions != 0) {
                // put back the items so that we begin from a multiple of dimensions
                // in case rotation is enabled
                while (runningLocation % dimensions != 0) {
                    runningLocation--;
                }
            }

            /**
             * recursively keep copying; if a new relevant shingle is found during the
             * copying, the remainsToBeCopied is updated to dimensions
             */
            if (runningLocation < currentStoreCapacity * dimensions) {
                int remainsToBeCopied = dimensions;
                int saveLocation = runningLocation;
                int shadowLocation = startOfFreeSegment;
                /**
                 * note that remainsToBeCopied corresponds to the shingle; the test for division
                 * by dimensions ensure that every rotated shingle
                 */
                while (runningLocation < currentStoreCapacity * dimensions
                        && (remainsToBeCopied > 0 || rotationEnabled && runningLocation % dimensions != 0)) {
                    if (baseDimension == 1 || runningLocation % baseDimension == 0) {
                        checkArgument(baseDimension == 1 || shadowLocation % baseDimension == 0, "error");

                        if (inUse.get(runningLocation / baseDimension)) { // need to copy dimension more bits
                            remainsToBeCopied = dimensions;
                            if (shadowLocation < runningLocation) { // actual move is necessary
                                movedTo.put(runningLocation, shadowLocation);
                            }
                        }
                    }
                    runningLocation++;
                    shadowLocation++;
                    remainsToBeCopied--;
                }
                copyTo(startOfFreeSegment, saveLocation, runningLocation - saveLocation);
                startOfFreeSegment += runningLocation - saveLocation;
            }
        }
        // now fix the addressing, assuming something has moved
        if (!movedTo.isEmpty()) {
            for (int i = 0; i < indexManager.capacity; i++) {
                if (movedTo.containsKey(locationList[i])) { // need not have moved
                    locationList[i] = movedTo.get(locationList[i]);
                }
            }
        }

    }

    /**
     * returns the number of copies of a point
     * 
     * @param i index of a point
     * @return number of copies of the point managed by the store
     */
    public int getRefCount(int i) {
        return refCount[i];
    }

    /**
     * transforms a point to a shingled point if internal shingling is turned on
     * 
     * @param point new input values
     * @return shingled point
     */
    @Override
    public double[] transformToShingledPoint(double[] point) {
        checkArgument(internalShinglingEnabled, " only allowed for internal shingling");
        checkArgument(point.length == baseDimension, " incorrect length");
        return changeShingleInPlace(Arrays.copyOf(internalShingle, dimensions), point);
    }

    /**
     * the following function is used to update the shingle in place; it can be used
     * to produce new copies as well
     * 
     * @param target the array containing the shingled point
     * @param point  the new values
     * @return the array which now contains the updated shingle
     */
    private double[] changeShingleInPlace(double[] target, double[] point) {
        if (!rotationEnabled) {
            for (int i = 0; i < dimensions - baseDimension; i++) {
                target[i] = target[i + baseDimension];
            }
            for (int i = 0; i < baseDimension; i++) {
                target[dimensions - baseDimension + i] = (point[i] == 0.0) ? 0.0 : point[i];
            }
        } else {
            int offset = ((int) nextSequenceIndex % dimensions);
            checkArgument(baseDimension == 1 || offset % baseDimension == 0, "incorrect state");
            for (int i = 0; i < baseDimension; i++) {
                target[offset + i] = (point[i] == 0.0) ? 0.0 : point[i];
            }
        }
        return target;
    }

    /**
     * for extrapolation and imputation, in presence of internal shingling we need
     * to update the list of missing values from the space of the input dimensions
     * to the shingled dimensions
     * 
     * @param indexList list of missing values in the input point
     * @return list of missing values in the shingled point
     */
    @Override
    public int[] transformIndices(int[] indexList) {
        checkArgument(internalShinglingEnabled, " only allowed for internal shingling");
        checkArgument(indexList.length <= baseDimension, " incorrect length");
        int[] results = Arrays.copyOf(indexList, indexList.length);
        if (!rotationEnabled) {
            for (int i = 0; i < indexList.length; i++) {
                results[i] += dimensions - baseDimension;
            }
        } else {
            int offset = ((int) nextSequenceIndex % dimensions);
            checkArgument(baseDimension == 1 || offset % baseDimension == 0, "incorrect state");
            for (int i = 0; i < indexList.length; i++) {
                results[i] = (results[i] + offset) % dimensions;
            }
        }
        return results;
    }

    /**
     * @return a new builder.
     */
    public static PointStore.Builder<?> builder() {
        return new PointStore.Builder<>();
    }

    /**
     * a builder
     */

    public static class Builder<T extends PointStore.Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        private int dimensions;
        private int shingleSize = 1;
        private boolean internalShinglingEnabled = false;
        private boolean dynamicResizingEnabled = true;
        private boolean internalRotationEnabled = false;
        private boolean directLocationEnabled = false;
        private int capacity;
        private int currentStoreCapacity;
        private int indexCapacity;
        private double[] knownShingle = null;
        private int[] freeIndexes = null;
        private int freeIndexPointer = INFEASIBLE_POINTSTORE_INDEX;
        private int[] locationList = null;
        private int[] refCount = null;
        private long nextTimeStamp = 0;
        private int startOfFreeSegment = 0;

        public T dimensions(int dimensions) {
            this.dimensions = dimensions;
            return (T) this;
        }

        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        public T currentStoreCapacity(int currentStoreCapacity) {
            this.currentStoreCapacity = currentStoreCapacity;
            return (T) this;
        }

        public T indexCapacity(int indexCapacity) {
            this.indexCapacity = indexCapacity;
            return (T) this;
        }

        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        public T internalShinglingEnabled(boolean internalShinglingEnabled) {
            this.internalShinglingEnabled = internalShinglingEnabled;
            return (T) this;
        }

        public T directLocationEnabled(boolean directLocationEnabled) {
            this.directLocationEnabled = directLocationEnabled;
            return (T) this;
        }

        public T internalRotationEnabled(boolean internalRotationEnabled) {
            this.internalRotationEnabled = internalRotationEnabled;
            return (T) this;
        }

        public T dynamicResizingEnabled(boolean dynamicResizingEnabled) {
            this.dynamicResizingEnabled = dynamicResizingEnabled;
            return (T) this;
        }

        public T knownShingle(double[] knownShingle) {
            this.knownShingle = knownShingle;
            return (T) this;
        }

        public T refCount(int[] refCount) {
            this.refCount = refCount;
            return (T) this;
        }

        public T locationList(int[] locationList) {
            this.locationList = locationList;
            return (T) this;
        }

        /*
         * public T freeIndexes(int[] freeIndexes) { this.freeIndexes = freeIndexes;
         * return (T) this; }
         * 
         * public T freeIndexPointer(int freeIndexPointer) { this.freeIndexPointer =
         * freeIndexPointer; return (T) this; }
         */
        public T startOfFreeSegment(int startOfFreeSegment) {
            this.startOfFreeSegment = startOfFreeSegment;
            return (T) this;
        }

        public T nextTimeStamp(long nextTimeStamp) {
            this.nextTimeStamp = nextTimeStamp;
            return (T) this;
        }

    }

    public PointStore(Builder builder) {
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        checkArgument(builder.shingleSize == 1 || builder.dimensions == builder.shingleSize
                || builder.dimensions % builder.shingleSize == 0, "incorrect use of shingle size");
        checkArgument(builder.dynamicResizingEnabled || builder.currentStoreCapacity > 0,
                "capacity must be positive if dynamic resizing is disabled");
        checkArgument(!builder.internalRotationEnabled || builder.internalShinglingEnabled,
                "rotation can be enabled for internal shingling only");
        if (builder.refCount != null || builder.freeIndexes != null || builder.locationList != null
                || builder.knownShingle != null) {
            checkArgument(builder.refCount == null || builder.refCount.length == builder.indexCapacity,
                    "incorrect state");
            // following may change if IndexManager is dynamically resized as well
            checkArgument(builder.locationList == null || builder.locationList.length == builder.indexCapacity,
                    " incorrect state");
            checkArgument(builder.knownShingle == null || builder.internalShinglingEnabled, "incorrect state");
        }

        this.shingleSize = builder.shingleSize;
        this.dimensions = builder.dimensions;
        this.directLocationMap = builder.directLocationEnabled;
        this.internalShinglingEnabled = builder.internalShinglingEnabled;
        this.rotationEnabled = builder.internalRotationEnabled;
        this.currentStoreCapacity = builder.currentStoreCapacity;
        this.dynamicResizingEnabled = builder.dynamicResizingEnabled;
        this.baseDimension = this.dimensions / this.shingleSize;
        this.capacity = builder.capacity;

        if (builder.refCount != null || builder.freeIndexes != null || builder.locationList != null
                || builder.knownShingle != null) {

            this.refCount = builder.refCount;
            this.locationList = builder.locationList;
            this.startOfFreeSegment = builder.startOfFreeSegment;
            this.nextSequenceIndex = builder.nextTimeStamp;

            if (internalShinglingEnabled) {
                this.internalShingle = new double[dimensions];
                if (builder.knownShingle != null) { // can be for empty forest
                    System.arraycopy(builder.knownShingle, 0, this.internalShingle, 0, dimensions);
                }
            }
            BitSet bits = new BitSet(builder.indexCapacity);
            for (int i = 0; i < builder.indexCapacity; i++) {
                if (refCount[i] > 0) {
                    bits.set(i);
                }
            }
            indexManager = new IndexManager(builder.indexCapacity, bits);
        } else {
            indexManager = new IndexManager(builder.indexCapacity);
            startOfFreeSegment = 0;
            refCount = new int[builder.indexCapacity];
            if (internalShinglingEnabled) {
                nextSequenceIndex = 0;
                internalShingle = new double[dimensions];
            }
            locationList = new int[builder.indexCapacity];
            Arrays.fill(locationList, INFEASIBLE_POINTSTORE_LOCATION);
        }
    }

}
