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

package com.amazon.randomcutforest.parkservices.statistics;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

/**
 * This class maintains a simple discounted statistics. Setters are avoided
 * except for discount rate which is useful as initialization from raw scores
 */
public class Deviation {

    protected double discount;

    protected double weight = 0;

    protected double sumSquared = 0;

    protected double sum = 0;

    protected int count = 0;

    public Deviation() {
        discount = 0;
    }

    public Deviation(double discount) {
        checkArgument(0 <= discount && discount < 1, "incorrect discount parameter");
        this.discount = discount;
    }

    public Deviation(double discount, double weight, double sumSquared, double sum, int count) {
        this.discount = discount;
        this.weight = weight;
        this.sumSquared = sumSquared;
        this.sum = sum;
        this.count = count;
    }

    public Deviation copy() {
        return new Deviation(this.discount, this.weight, this.sumSquared, this.sum, this.count);
    }

    public double getMean() {
        return (weight <= 0) ? 0 : sum / weight;
    }

    public void update(double score) {
        double factor = (discount == 0) ? 1 : Math.min(1 - discount, 1 - 1.0 / (count + 2));
        sum = sum * factor + score;
        sumSquared = sumSquared * factor + score * score;
        weight = weight * factor + 1.0;
        ++count;
    }

    public double getDeviation() {
        if (weight <= 0) {
            return 0;
        }
        double temp = sum / weight;
        double answer = sumSquared / weight - temp * temp;
        return (answer > 0) ? Math.sqrt(answer) : 0;
    }

    public boolean isEmpty() {
        return weight == 0;
    }

    public double getDiscount() {
        return discount;
    }

    public void setDiscount(double discount) {
        this.discount = discount;
    }

    public double getSum() {
        return sum;
    }

    public double getSumSquared() {
        return sumSquared;
    }

    public double getWeight() {
        return weight;
    }

    public int getCount() {
        return count;
    }

    public void setCount(int count) {
        this.count = count;
    }
}
