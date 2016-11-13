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

package org.apache.mahout.common.distance;

import org.apache.mahout.math.Vector;

/**
 * This class implements a Euclidean distance metric by summing the square root of the squared differences
 * between each coordinate.
 * <p/>
 * If you don't care about the true distance and only need the values for comparison, then the base class,
 * {@link SquaredEuclideanDistanceMeasure}, will be faster since it doesn't do the actual square root of the
 * squared differences.
 * 欧氏距离
 * 就是  (x-y)的平方和   然后开方
 */
public class EuclideanDistanceMeasure extends SquaredEuclideanDistanceMeasure {

  /**
   * 获取两个向量的距离的平方和  然后开方
   * 例如
   * 1.默认的x-y的平方和 或者 (|x|)平方 + |y|平方 - 2xy 即x的向量模平方+y的向量模平方-2xy的点积
   * 2.然后开方
   */
  @Override
  public double distance(Vector v1, Vector v2) {
    return Math.sqrt(super.distance(v1, v2));
  }//计算(x-y)的平方和   然后开方

  /**
   *
   * @param centroidLengthSquare
   *          Square of the length of centroid 表示centroid向量的getLengthSquared值
   * @param centroid 中心向量
   * @param v 另外一个向量
   * 使用(|x|)平方 + |y|平方 - 2xy方式运算  然后开方
   */
  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return Math.sqrt(super.distance(centroidLengthSquare, centroid, v));
  }
}
