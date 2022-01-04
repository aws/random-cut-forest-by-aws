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

import java.util.Arrays;

public class PointStoreSmall extends PointStore {

    public static char INFEASIBLE_SMALL_POINTSTORE_LOCN = (char) -1;
    protected char[] locationList;

    void setInfeasiblePointstoreLocationIndex(int index) {
        locationList[index] = INFEASIBLE_SMALL_POINTSTORE_LOCN;
    };

    void extendLocationList(int newCapacity) {
        int oldCapacity = locationList.length;
        assert (oldCapacity < newCapacity);
        locationList = Arrays.copyOf(locationList, newCapacity);
        for (int i = oldCapacity; i < newCapacity; i++) {
            locationList[i] = INFEASIBLE_SMALL_POINTSTORE_LOCN;
        }
    };

    void setLocation(int index, int location) {
        locationList[index] = (char) (location / baseDimension);
        assert (baseDimension * (int) locationList[index] == location);
    }

    int getLocation(int index) {
        return baseDimension * (int) locationList[index];
    }

    int locationListLength() {
        return locationList.length;
    }

    public PointStoreSmall(PointStore.Builder builder) {
        super(builder);
        checkArgument(shingleSize * capacity < Character.MAX_VALUE, " incorrect parameters");
        if (builder.locationList != null) {
            locationList = new char[builder.locationList.length];
            for (int i = 0; i < locationList.length; i++) {
                locationList[i] = (char) builder.locationList[i];
            }
        } else {
            locationList = new char[currentStoreCapacity];
            Arrays.fill(locationList, INFEASIBLE_SMALL_POINTSTORE_LOCN);
        }
    }

    public PointStoreSmall(int dimensions, int capacity) {
        this(PointStore.builder().capacity(capacity).dimensions(dimensions).shingleSize(1).initialSize(capacity));
    }

    @Override
    protected void checkFeasible(int index) {
        checkArgument(locationList[index] != INFEASIBLE_SMALL_POINTSTORE_LOCN, " invalid point");
    }

    @Override
    public int size() {
        int count = 0;
        for (int i = 0; i < locationList.length; i++) {
            if (locationList[i] != INFEASIBLE_SMALL_POINTSTORE_LOCN) {
                ++count;
            }
        }
        return count;
    }

    @Override
    public int[] getLocationList() {
        int[] answer = new int[locationList.length];
        for (int i = 0; i < locationList.length; i++) {
            answer[i] = locationList[i];
        }
        return answer;
    }
}
