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
 * Only for performance tuning of compute intensive linear algebraic computations.
 * Constructs functions that return one of
 * <ul>
 * <li><tt>a * constant</tt>
 * <li><tt>a / constant</tt>
 * </ul> 
 * <tt>a</tt> is variable, <tt>constant</tt> is fixed, but for performance reasons publicly accessible.
 * Intended to be passed to <tt>matrix.assign(function)</tt> methods.
 * 
 * a是一个变量,constant是一个常量
 */

public final class Mult extends DoubleFunction {

  private double multiplicator;//常量

  Mult(double multiplicator) {
    this.multiplicator = multiplicator;
  }

  /** Returns the result of the function evaluation. 
   * 所有的参数a都*一个常量 返回值 
   **/
  @Override
  public double apply(double a) {
    return a * multiplicator;
  }

  /** <tt>a / constant</tt>. 
   * a * (1/constant),即变成了 a/常量
   **/
  public static Mult div(double constant) {
    return mult(1 / constant);
  }

  /** <tt>a * constant</tt>. 
   * 用于除法,因此常量变成了 1/constant,这样apply方法可以保持不变,达到了a/constant的目的
   **/
  public static Mult mult(double constant) {
    return new Mult(constant);
  }

  public double getMultiplicator() {
    return multiplicator;
  }

  //设置常量
  public void setMultiplicator(double multiplicator) {
    this.multiplicator = multiplicator;
  }
}
