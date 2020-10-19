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

package com.amazon.randomcutforest.sampler;

/**
 * A container class representing a weighted sequential value. This generic type
 * is used by {@link SimpleStreamSampler} to store weighted points of arbitrary
 * type.
 * 
 * @param <P> The representation of the point value.
 */
public class Weighted<P> {

    public P value;
    private float weight;

    /**
     * Create a new weighted value from a point value of type P.
     * 
     * @param point  A value.
     * @param weight The weight value.
     */
    public Weighted(P point, float weight) {
        this.value = point;
        this.weight = weight;
    }

    /**
     * copy constructor
     * 
     * @param other weighted point being copied
     */
    public Weighted(Weighted<P> other) {
        this.value = other.getValue();
        this.weight = other.getWeight();
    }

    /**
     * @return the weight value.
     */
    public float getWeight() {
        return weight;
    }

    /**
     *
     * @return the value
     */
    public P getValue() {
        return value;
    }

    /**
     * copy operation
     * 
     * @param other is another point
     */
    public void setValue(P other) {
        value = other;
    }

    /**
     * sets the weight
     * 
     * @param otherWeight new weight
     */
    public void setWeight(float otherWeight) {
        this.weight = otherWeight;
    }

}
