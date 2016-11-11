/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Copyright 1999 CERN - European Organization for Nuclear Research.
Permission to use, copy, modify, distribute and sell this software and its documentation for any purpose
is hereby granted without fee, provided that the above copyright notice appear in all copies and
that both that copyright notice and this permission notice appear in supporting documentation.
CERN makes no representations about the suitability of this software for any purpose.
It is provided "as is" without expressed or implied warranty.
*/

package org.apache.mahout.math.function;

/**
 * Interface that represents a function object: a function that takes two arguments and returns a single value.
 * 拿着两个参数返回一个值
 **/
public abstract class DoubleDoubleFunction {

  /**
   * Apply the function to the arguments and return the result
   *
   * @param arg1 a double for the first argument
   * @param arg2 a double for the second argument
   * @return the result of applying the function
   */
  public abstract double apply(double arg1, double arg2);

  /**
   * @return true iff f(x, 0) = x for any x
   * 对于任何元素 y=0,则无论x是什么,结果都是x,则返回true
   */
  public boolean isLikeRightPlus() {
    return false;
  }

  /**
   * @return true iff f(0, y) = 0 for any y
   * 对于任何元素 x=0,则无论y是什么,结果都是0,则返回true
   */
  public boolean isLikeLeftMult() {
    return false;
  }

  /**
   * @return true iff f(x, 0) = 0 for any x
   * 对于任何元素 y=0,则无论x是什么,结果都是0,则返回true
   */
  public boolean isLikeRightMult() {
    return false;
  }

  /**
   * @return true iff f(x, 0) = f(0, y) = 0 for any x, y
   * 只要x或者y是0,则结果一定是0,则返回true
   */
  public boolean isLikeMult() {
    return isLikeLeftMult() && isLikeRightMult();
  }

  /**
   * @return true iff f(x, y) = f(y, x) for any x, y 任何x和y都能做交换律
   */
  public boolean isCommutative() {
    return false;
  }

  /**
   * @return true iff f(x, f(y, z)) = f(f(x, y), z) for any x, y, z 说明x y z符合结合律
   */
  public boolean isAssociative() {
    return false;
  }

  /**
   * @return true iff f(x, y) = f(y, x) for any x, y AND f(x, f(y, z)) = f(f(x, y), z) for any x, y, z
   * 同时符合交换率和结合律
   */
  public boolean isAssociativeAndCommutative() {
    return isAssociative() && isCommutative();
  }

  /**
   * @return true iff f(0, 0) != 0
   * 两个参数都为0的时候,结果不为0,则返回true
   */
  public boolean isDensifying() {
    return apply(0.0, 0.0) != 0.0;
  }
}
