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

import java.util.Collection;
import java.util.Collections;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.Vector;

/**
 * Like {@link EuclideanDistanceMeasure} but it does not take the square root.
 * <p/>
 * Thus, it is not actually the Euclidean Distance, but it is saves on computation when you only need the
 * distance for comparison and don't care about the actual value as a distance.
 * 欧氏距离的平方和
 * 就是  (x-y)的平方和
 */
public class SquaredEuclideanDistanceMeasure implements DistanceMeasure {
  
  @Override
  public void configure(Configuration job) {
  // nothing to do
  }
  
  @Override
  public Collection<Parameter<?>> getParameters() {
    return Collections.emptyList();
  }
  
  @Override
  public void createParameters(String prefix, Configuration jobConf) {
  // nothing to do
  }

  //获取两个向量的距离的平方和
  //例如默认的x-y的平方和 或者 (|x|)平方 + |y|平方 - 2xy 即x的向量模平方+y的向量模平方-2xy的点积
  @Override
  public double distance(Vector v1, Vector v2) {
    return v2.getDistanceSquared(v1);
  }

  /**
   * 使用(|x|)平方 + |y|平方 - 2xy方式运算
   * @param centroidLengthSquare
   *          Square of the length of centroid 表示centroid向量的getLengthSquared值
   * @param centroid 中心向量
   * @param v 另外一个向量
   */
  @Override
  public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
    return centroidLengthSquare - 2 * v.dot(centroid) + v.getLengthSquared();
  }
}
