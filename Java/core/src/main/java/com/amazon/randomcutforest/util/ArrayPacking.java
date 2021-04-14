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

import java.util.Arrays;

public class ArrayPacking {

    static int logMax(int base) {
        int pack = 0;
        long num = base;
        while (num < Integer.MAX_VALUE) {
            num = num * base;
            ++pack;
        }
        return pack;
    }

    public static int[] pack(short[] inputArray, boolean compress) {
        checkNotNull(inputArray, " array packing invoked on null arrays");

        if (!compress || inputArray.length < 3) {
            int[] output = new int[inputArray.length];
            for (int i = 0; i < inputArray.length; i++) {
                output[i] = inputArray[i];
            }
            return output;
        }
        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < inputArray.length; i++) {
            min = Math.min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        int base = max - min + 1;
        if (base == 1) {
            return new int[] { min, max, inputArray.length };
        } else {
            int packNum = logMax(base);
            int[] output = new int[3 + (int) Math.ceil(1.0 * inputArray.length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = inputArray.length;
            int len = 0;
            int used = 0;
            while (len < inputArray.length) {
                int code = 0;
                int reach = Math.min(len + packNum - 1, inputArray.length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = code;
                len += packNum;
            }
            checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }

    }

    public static short[] unPackShorts(int[] inputArray, boolean decomress) {
        checkNotNull(inputArray, " array unpacking invoked on null arrays");
        if (inputArray.length < 3 || !decomress) {
            short[] output = new short[inputArray.length];
            for (int i = 0; i < inputArray.length; i++) {
                output[i] = (short) inputArray[i];
            }
            return output;
        }
        int min = inputArray[0];
        int max = inputArray[1];
        short[] output = new short[inputArray[2]];
        if (min == max) {
            Arrays.fill(output, (short) min);
        } else {
            int base = (max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < inputArray.length; i++) {
                int code = inputArray[i];
                for (int j = 0; j < packNum && count < output.length; j++) {
                    output[count++] = (short) (min + code % base);
                    code = code / base;
                }
            }
            ;
        }
        return output;
    }

    public static int[] pack(int[] inputArray, boolean compress) {
        checkNotNull(inputArray, " array packing invoked on null arrays");

        if (!compress || inputArray.length < 3) {
            return Arrays.copyOf(inputArray, inputArray.length);
        }

        int min = inputArray[0];
        int max = inputArray[0];
        for (int i = 1; i < inputArray.length; i++) {
            min = Math.min(min, inputArray[i]);
            max = Math.max(max, inputArray[i]);
        }
        int base = max - min + 1;
        if (base == 1) {
            return new int[] { min, max, inputArray.length };
        } else {
            int packNum = logMax(base);
            int[] output = new int[3 + (int) Math.ceil(1.0 * inputArray.length / packNum)];
            output[0] = min;
            output[1] = max;
            output[2] = inputArray.length;
            int len = 0;
            int used = 0;
            while (len < inputArray.length) {
                int code = 0;
                int reach = Math.min(len + packNum - 1, inputArray.length - 1);
                for (int i = reach; i >= len; i--) {
                    code = base * code + (inputArray[i] - min);
                }
                output[3 + used++] = code;
                len += packNum;
            }
            checkArgument(used + 3 == output.length, "incorrect state");
            return output;
        }

    }

    public static int[] unPackInts(int[] inputArray, boolean decomress) {
        checkNotNull(inputArray, " array unpacking invoked on null arrays");
        if (inputArray.length < 3 || !decomress) {
            return Arrays.copyOf(inputArray, inputArray.length);
        }
        int min = inputArray[0];
        int max = inputArray[1];
        int[] output = new int[inputArray[2]];
        if (min == max) {
            Arrays.fill(output, min);
        } else {
            int base = (max - min + 1);
            int packNum = logMax(base);
            int count = 0;
            for (int i = 3; i < inputArray.length; i++) {
                int code = inputArray[i];
                for (int j = 0; j < packNum && count < output.length; j++) {
                    output[count++] = (min + code % base);
                    code = code / base;
                }
            }
        }
        return output;
    }

}
