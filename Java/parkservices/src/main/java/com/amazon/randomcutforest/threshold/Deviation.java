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

package com.amazon.randomcutforest.threshold;


import static com.amazon.randomcutforest.CommonUtils.checkArgument;

public class Deviation {

    protected final double discount;

    protected double weight = 0;

    protected double sumSquared = 0;

    protected double sum = 0;

    public Deviation(){
        discount = 0;
    }

    public Deviation(double discount){
        this.discount = discount;
    }

    public double getMean(){
        checkArgument(weight>0, "incorrect invocation for mean");
        return sum/weight;
    }

    public void update(double score){
        sum = sum * (1 - discount) + score;
        sumSquared = sumSquared * (1-discount) + score * score;
        weight = weight * (1-discount) + 1.0;
    }

    public double getDeviation(){
        checkArgument(weight>0, "incorrect invocation for standard deviation");
        double temp = sum/weight;
        double answer = sumSquared/weight - temp * temp;
        return (answer>0)?Math.sqrt(answer):0;
    }

    public boolean isEmpty(){
        return weight == 0;
    }

}
