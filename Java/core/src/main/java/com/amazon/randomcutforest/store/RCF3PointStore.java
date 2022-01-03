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
import static java.lang.Math.max;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Optional;
import java.util.Vector;

public abstract class RCF3PointStore implements IPointStore<float[]> {

    public static int INFEASIBLE_POINTSTORE_INDEX = -1;
    /**
     * an index manager to manage free locations
     */
    protected IntervalManager indexManager;
    /**
     * generic store class
     */
    protected float[] store;
    /**
     * generic internal shingle, note that input is doubles
     */
    protected double[] internalShingle;
    /**
     * enable rotation of shingles; use a cyclic buffer instead of sliding window
     */
    boolean rotationEnabled;
    /**
     * last seen timestamp for internal shingling
     */
    protected long nextSequenceIndex;
    /**
     * pointers to store locations, this decouples direct addressing and points can
     * be moved internally
     */
    // protected char[] locationList;

    /**
     * refCount[i] counts of the number of trees that are currently using the point
     * determined by locationList[i] or (for directLocationMapping) the point at
     * store[i * dimensions]
     */
    protected byte[] refCount;

    protected HashMap<Integer, Integer> refCountMap;
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
     * maximum capacity
     */
    int capacity;
    /**
     * current capacity of store (number of shingled points)
     */
    int currentStoreCapacity;

    /**
     * enabling internal shingling
     */
    boolean internalShinglingEnabled;

    abstract void setInfeasiblePointstoreLocationIndex(int index);

    abstract void extendLocationList(int newCapacity);

    abstract void setLocation(int index, int location);

    abstract int getLocation(int index);

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
        checkArgument((refCount[index] & 0xff) > 0, " cannot decrement index");
        Integer value = refCountMap.remove(index);
        if (value == null) {
            if ((refCount[index] & 0xff) == 1) {
                indexManager.releaseIndex(index);
                refCount[index] = (byte) 0;
                setInfeasiblePointstoreLocationIndex(index);
                return 0;
            } else {
                int newVal = (byte) ((refCount[index] & 0xff) - 1);
                refCount[index] = (byte) newVal;
                return newVal;
            }
        } else {
            if (value > 1) {
                refCountMap.put(index, value - 1);
            }
            return value - 1 + (refCount[index] & 0xff);
        }
    }

    /**
     * takes an index from the index manager and rezises if necessary also adjusts
     * refCount size to have increment/decrement be seamless
     *
     * @return an index from the index manager
     */
    int takeIndex() {
        if (indexManager.isEmpty()) {
            if (indexManager.getCapacity() < capacity) {
                int oldCapacity = indexManager.getCapacity();
                int newCapacity = Math.min(capacity, 1 + (int) Math.floor(1.1 * oldCapacity));
                indexManager.extendCapacity(newCapacity);
                refCount = Arrays.copyOf(refCount, newCapacity);
                extendLocationList(newCapacity);
            } else {
                throw new IllegalStateException(" index manager in point store is full ");
            }
        }
        return indexManager.takeIndex();
    }

    protected int getAmountToWrite(double[] tempPoint) {
        if (checkShingleAlignment(startOfFreeSegment, tempPoint)) {
            if (!rotationEnabled
                    || startOfFreeSegment % dimensions == (nextSequenceIndex - 1) * baseDimension % dimensions) {
                return baseDimension;
            }
        } else if (!rotationEnabled) {
            return dimensions;

        }
        // the following adds the padding for what exists;
        // then the padding for the new part; all mod (dimensions)
        // note that the expression is baseDimension when the condition
        // startOfFreeSegment % dimensions == (nextSequenceIndex-1)*baseDimension %
        // dimension
        // is met
        return dimensions + (dimensions - startOfFreeSegment % dimensions
                + (int) ((nextSequenceIndex) * baseDimension) % dimensions) % dimensions;
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
        nextSequenceIndex++;
        if (internalShinglingEnabled) {
            // rotation is supported via the output and input is unchanged
            tempPoint = constructShingleInPlace(internalShingle, point, false);
            if (nextSequenceIndex < shingleSize) {
                return INFEASIBLE_POINTSTORE_INDEX;
            }
        }
        int nextIndex;

        int amountToWrite = getAmountToWrite(tempPoint);

        if (startOfFreeSegment > currentStoreCapacity * dimensions - amountToWrite) {
            // try compaction and then resizing
            compact();
            // the compaction can change the array contents
            amountToWrite = getAmountToWrite(tempPoint);
            if (startOfFreeSegment > currentStoreCapacity * dimensions - amountToWrite) {
                resizeStore();
                checkState(startOfFreeSegment + amountToWrite <= currentStoreCapacity * dimensions, "out of space");
            }
        }

        nextIndex = takeIndex();

        setLocation(nextIndex, startOfFreeSegment - dimensions + amountToWrite);
        if (amountToWrite <= dimensions) {
            copyPoint(tempPoint, dimensions - amountToWrite, startOfFreeSegment, amountToWrite);
        } else {
            copyPoint(tempPoint, 0, startOfFreeSegment + amountToWrite - dimensions, dimensions);
        }
        startOfFreeSegment += amountToWrite;

        refCount[nextIndex] = 1;
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
     *                                  index is non positive.
     */
    public int incrementRefCount(int index) {
        // indexManager.checkValidIndex(index);
        Integer value = refCountMap.remove(index);
        if (value == null) {
            if ((refCount[index] & 0xff) == 255) {
                refCountMap.put(index, 1);
                return 256;
            } else {
                int newVal = (byte) ((refCount[index] & 0xff) + 1);
                refCount[index] = (byte) newVal;
                return newVal;
            }
        } else {
            refCountMap.put(index, value + 1);
            return value + 1;
        }
    }

    @Override
    public int getDimensions() {
        return dimensions;
    }

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
    public float[] getStore() {
        return store;
    }

    /**
     * used for mapper
     *
     * @return the array of counts referring to different points
     */
    public int[] getRefCount() {
        int[] newarray = new int[refCount.length];
        for (int i = 0; i < refCount.length; i++) {
            newarray[i] = refCount[i] & 0xff;
            Integer value = refCountMap.get(i);
            if (value != null) {
                newarray[i] += value;
            }
        }
        return newarray;
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
     * @return if shingling is performed internally
     */
    public boolean isInternalShinglingEnabled() {
        return internalShinglingEnabled;
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
     * used to obtain the most recent shingle seen so far in case of internal
     * shingling
     *
     * @return for internal shingling, returns the last seen shingle
     */
    public double[] getInternalShingle() {
        checkState(internalShinglingEnabled, "internal shingling is not enabled");
        return copyShingle();
    }

    /**
     * The following function eliminates redundant information that builds up in the
     * point store and shrinks the point store
     */

    abstract int locationListLength();

    public void compact() {

        Vector<Integer[]> reverseReference = new Vector<>();
        for (int i = 0; i < locationListLength(); i++) {
            int locn = getLocation(i);
            if (locn < currentStoreCapacity * dimensions && locn >= 0) {
                reverseReference.add(new Integer[] { locn, i });
            }
        }
        reverseReference.sort((o1, o2) -> o1[0].compareTo(o2[0]));
        int freshStart = 0;
        int jStatic = 0;
        int jDynamic = 0;
        int jEnd = reverseReference.size();
        while (jStatic < jEnd) {
            int blockStart = reverseReference.get(jStatic)[0];
            int blockEnd = blockStart + dimensions;
            int initial = 0;
            if (rotationEnabled) {
                initial = (dimensions - freshStart + blockStart) % dimensions;
            }
            int k = jStatic + 1;
            jDynamic = jStatic + 1;
            while (k < jEnd) {
                int newElem = reverseReference.get(k)[0];
                if (blockEnd >= newElem) {
                    k += 1;
                    jDynamic += 1;
                    blockEnd = max(blockEnd, newElem + dimensions);
                } else {
                    k = jEnd;
                }
            }

            // aligning the boundaries
            for (int i = 0; i < initial; i++) {
                store[freshStart] = 0;
                ++freshStart;
            }

            for (int i = blockStart; i < blockEnd; i++) {
                store[freshStart] = store[i];
                assert (!rotationEnabled || freshStart % dimensions == i % dimensions);

                if (jStatic < jEnd) {
                    int locn = reverseReference.get(jStatic)[0];
                    if (i == locn) {
                        int newIdx = reverseReference.get(jStatic)[1];
                        setLocation(newIdx, freshStart);
                        jStatic += 1;
                    }
                }
                freshStart += 1;
            }

            if (jStatic != jDynamic) {
                throw new IllegalStateException("There is discepancy in indices");
            }
        }
        startOfFreeSegment = freshStart;
    }

    /**
     * returns the number of copies of a point
     *
     * @param i index of a point
     * @return number of copies of the point managed by the store
     */
    public int getRefCount(int i) {
        int val = refCount[i] & 0xff;
        Integer value = refCountMap.get(i);
        if (value != null) {
            val += value;
        }
        return val;
    }

    @Override
    public boolean isInternalRotationEnabled() {
        return rotationEnabled;
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
        return constructShingleInPlace(copyShingle(), point, rotationEnabled);
    }

    private double[] copyShingle() {
        if (!rotationEnabled) {
            return Arrays.copyOf(internalShingle, dimensions);
        } else {
            double[] answer = new double[dimensions];
            int offset = (int) (nextSequenceIndex * baseDimension);
            for (int i = 0; i < dimensions; i++) {
                answer[(offset + i) % dimensions] = internalShingle[i];
            }
            return answer;
        }
    }

    /**
     * the following function is used to update the shingle in place; it can be used
     * to produce new copies as well
     *
     * @param target the array containing the shingled point
     * @param point  the new values
     * @return the array which now contains the updated shingle
     */
    protected double[] constructShingleInPlace(double[] target, double[] point, boolean rotationEnabled) {
        if (!rotationEnabled) {
            for (int i = 0; i < dimensions - baseDimension; i++) {
                target[i] = target[i + baseDimension];
            }
            for (int i = 0; i < baseDimension; i++) {
                target[dimensions - baseDimension + i] = (point[i] == 0.0) ? 0.0 : point[i];
            }
        } else {
            int offset = ((int) (nextSequenceIndex * baseDimension) % dimensions);
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
                checkArgument(results[i] < baseDimension, "incorrect index");
                results[i] += dimensions - baseDimension;
            }
        } else {
            int offset = ((int) (nextSequenceIndex * baseDimension) % dimensions);
            for (int i = 0; i < indexList.length; i++) {
                checkArgument(results[i] < baseDimension, "incorrect index");
                results[i] = (results[i] + offset) % dimensions;
            }
        }
        return results;
    }

    /**
     * a builder
     */

    public static class Builder<T extends Builder<T>> {

        // We use Optional types for optional primitive fields when it doesn't make
        // sense to use a constant default.

        protected int dimensions;
        protected int shingleSize = 1;
        protected int baseDimension;
        protected boolean internalRotationEnabled = false;
        protected boolean internalShinglingEnabled = false;
        protected int capacity;
        protected Optional<Integer> initialPointStoreSize = Optional.empty();
        protected int currentStoreCapacity = 0;
        protected int indexCapacity = 0;
        protected float[] store = null;
        protected double[] knownShingle = null;
        protected int[] locationList = null;
        protected int[] refCount = null;
        protected long nextTimeStamp = 0;
        protected int startOfFreeSegment = 0;

        // dimension of the points being stored
        public T dimensions(int dimensions) {
            this.dimensions = dimensions;
            return (T) this;
        }

        // maximum number of points in the store
        public T capacity(int capacity) {
            this.capacity = capacity;
            return (T) this;
        }

        // initial size of the pointstore, dynamicResizing must be on
        // and value cannot exceed capacity
        public T initialSize(int initialPointStoreSize) {
            this.initialPointStoreSize = Optional.of(initialPointStoreSize);
            return (T) this;
        }

        // shingleSize for opportunistic compression
        public T shingleSize(int shingleSize) {
            this.shingleSize = shingleSize;
            return (T) this;
        }

        // is internal shingling enabled
        public T internalShinglingEnabled(boolean internalShinglingEnabled) {
            this.internalShinglingEnabled = internalShinglingEnabled;
            return (T) this;
        }

        // are shingles rotated
        public T internalRotationEnabled(boolean internalRotationEnabled) {
            this.internalRotationEnabled = internalRotationEnabled;
            return (T) this;
        }

        // the size of the array storing the specific points
        // this is used for serialization
        public T currentStoreCapacity(int currentStoreCapacity) {
            this.currentStoreCapacity = currentStoreCapacity;
            return (T) this;
        }

        // the size of the pointset being tracked
        // this is used for serialization
        public T indexCapacity(int indexCapacity) {
            this.indexCapacity = indexCapacity;
            return (T) this;
        }

        // last known shingle, if internalshingle is on
        // this shingle is not rotated
        // this is used for serialization
        public T knownShingle(double[] knownShingle) {
            this.knownShingle = knownShingle;
            return (T) this;
        }

        // count of the points being tracked
        // used for serialization
        public T refCount(int[] refCount) {
            this.refCount = refCount;
            return (T) this;
        }

        // location of the points being tracked, if not directmapped
        // used for serialization
        public T locationList(int[] locationList) {
            this.locationList = locationList;
            return (T) this;
        }

        public T store(float[] store) {
            this.store = store;
            return (T) this;
        }

        // location of where points can be written
        // used for serialization
        public T startOfFreeSegment(int startOfFreeSegment) {
            this.startOfFreeSegment = startOfFreeSegment;
            return (T) this;
        }

        // the next timeStamp to accept
        // used for serialization
        public T nextTimeStamp(long nextTimeStamp) {
            this.nextTimeStamp = nextTimeStamp;
            return (T) this;
        }

        public IPointStore build() {
            if (shingleSize * capacity < Character.MAX_VALUE) {
                return new RCF3PointStoreSmall(this);
            } else {
                return new RCF3PointStoreLarge(this);
            }
        }
    }

    public RCF3PointStore(RCF3PointStore.Builder builder) {
        checkArgument(builder.dimensions > 0, "dimensions must be greater than 0");
        checkArgument(builder.capacity > 0, "capacity must be greater than 0");
        checkArgument(builder.shingleSize == 1 || builder.dimensions == builder.shingleSize
                || builder.dimensions % builder.shingleSize == 0, "incorrect use of shingle size");
        /**
         * the following checks are due to mappers (kept for future)
         */
        if (builder.refCount != null || builder.locationList != null || builder.knownShingle != null) {
            checkArgument(builder.refCount != null, "reference count must be present");
            checkArgument(builder.locationList != null, "location list must be present");
            checkArgument(builder.refCount.length == builder.indexCapacity, "incorrect reference count length");
            // following may change if IndexManager is dynamically resized as well
            checkArgument(builder.locationList.length == builder.indexCapacity, " incorrect length of locations");
            checkArgument(
                    builder.knownShingle == null
                            || builder.internalShinglingEnabled && builder.knownShingle.length == builder.dimensions,
                    "incorrect shingling information");
        }

        this.shingleSize = builder.shingleSize;
        this.dimensions = builder.dimensions;
        this.internalShinglingEnabled = builder.internalShinglingEnabled;
        this.rotationEnabled = builder.internalRotationEnabled;
        this.baseDimension = this.dimensions / this.shingleSize;
        this.capacity = builder.capacity;
        this.refCountMap = new HashMap<>();

        if (builder.refCount == null) {
            int size = (int) builder.initialPointStoreSize.orElse(builder.capacity);
            currentStoreCapacity = size;
            this.indexManager = new IntervalManager(size);
            startOfFreeSegment = 0;
            refCount = new byte[size];
            if (internalShinglingEnabled) {
                nextSequenceIndex = 0;
                internalShingle = new double[dimensions];
            }
            store = new float[currentStoreCapacity * dimensions];
        } else {
            throw new IllegalStateException("not yet supported");
        }
    }

    void resizeStore() {
        int maxCapacity = (rotationEnabled) ? 2 * capacity : capacity;
        int newCapacity = (int) Math.floor(Math.min(1.1 * currentStoreCapacity, maxCapacity));
        if (newCapacity > currentStoreCapacity) {
            float[] newStore = new float[newCapacity * dimensions];
            System.arraycopy(store, 0, newStore, 0, currentStoreCapacity * dimensions);
            currentStoreCapacity = newCapacity;
            store = newStore;
        }
    }

    boolean checkShingleAlignment(int location, double[] point) {
        boolean test = (location - dimensions + baseDimension >= 0);
        for (int i = 0; i < dimensions - baseDimension && test; i++) {
            test = (((float) point[i]) == store[location - dimensions + baseDimension + i]);
        }
        return test;
    }

    void copyPoint(double[] point, int src, int location, int length) {
        for (int i = 0; i < length; i++) {
            store[location + i] = (float) point[src + i];
        }
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

    @Override
    public boolean pointEquals(int index, float[] point) {
        // indexManager.checkValidIndex(index);
        checkArgument(point.length == dimensions, "point.length must be equal to dimensions");
        int address = getLocation(index);
        if (!rotationEnabled) {
            for (int j = 0; j < dimensions; j++) {
                if (point[j] != store[j + address]) {
                    return false;
                }
            }
        } else {
            for (int j = 0; j < dimensions; j++) {
                if (point[(j + address) % dimensions] != store[j + address]) {
                    return false;
                }
            }
        }

        return true;
    }

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
    public float[] get(int index) {
        // indexManager.checkValidIndex(index);
        int address = getLocation(index);
        if (!rotationEnabled) {
            return Arrays.copyOfRange(store, address, address + dimensions);
        } else {
            float[] answer = new float[dimensions];
            for (int i = 0; i < dimensions; i++) {
                answer[(address + i) % dimensions] = store[address + i];
            }
            return answer;
        }
    }

    public float[] getScaledPoint(int index, double factor) {
        float[] answer = get(index);
        for (int i = 0; i < dimensions; i++) {
            answer[i] *= factor;
        }
        return answer;
    }

    public String toString(int index) {
        return Arrays.toString(get(index));
    }

    void copyTo(int dest, int source, int length) {
        // validateInternalState(dest <= source, "error");
        for (int i = 0; i < length; i++) {
            store[dest + i] = store[source + i];
        }
    }

    public static Builder builder() {
        return new Builder();
    }

}
