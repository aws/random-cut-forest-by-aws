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

package com.amazon.randomcutforest.util;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.CommonUtils.checkNotNull;
import static java.lang.Math.min;

import java.util.Arrays;

public class ArrayPacking {

    static int logMax(long base) {
        int pack = 0;
        long num = base;
        while (num < Integer.MAX_VALUE) {
            num = num * base;
            ++pack;
        }
        return Math.max(pack, 1); // pack can be 0 for max - min being more than Integer.MaxValue
    }

    public static int[] pack(int[] inputArray, boolean compress) {
        return pack(inputArray, inputArray.length, compress);
    }

    public static int[] pack(int[] inputArray, int length, boolean compress) {
        checkNotNull(inputArray, " array packing invoked on null arrays");
        checkArgument(length <= inputArray.length && length >= 0, "incorrect length parameter");

        if (!compress || length < 3) {
            return Arrays.copyOf(inputArray, length);
        }

        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < length; i++) {
            min = min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        long base = (long) max - min + 1;
        if (base == 1) {
            return new int[] { min, max, length };
        } else {
            int packNum = logMax(base);

            int[] output = new int[3 + (int) Math.ceil(1.0 * length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = length;
            int len = 0;
            int used = 0;
            while (len < length) {
                long code = 0;
                int reach = min(len + packNum - 1, length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = (int) code;
                len += packNum;
            }
            checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }

    }

    public static int[] unPackInts(int[] inputArray, boolean decompress) {
        checkNotNull(inputArray, " array unpacking invoked on null arrays");

        return (inputArray.length < 3) ? unPackInts(inputArray, inputArray.length, decompress)
                : unPackInts(inputArray, inputArray[2], decompress);
    }

    public static int[] unPackInts(int[] inputArray, int length, boolean decompress) {
        checkNotNull(inputArray, " array unpacking invoked on null arrays");
        checkArgument(length >= 0, "incorrect length parameter");

        if (inputArray.length < 3 || !decompress) {
            return Arrays.copyOf(inputArray, length);
        }
        int min = inputArray[0];
        int max = inputArray[1];
        int[] output = new int[length];
        if (min == max) {
            if (inputArray[2] >= length) {
                Arrays.fill(output, min);
            } else {
                for (int i = 0; i < inputArray[2]; i++) {
                    output[i] = min;
                }
            }
        } else {
            long base = ((long) max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < inputArray.length; i++) {
                long code = inputArray[i];
                for (int j = 0; j < packNum && count < min(output.length, length); j++) {
                    output[count++] = (int) (min + code % base);
                    code = (int) (code / base);
                }
            }
        }
        return output;
    }

}
