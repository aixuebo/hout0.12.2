/* Licensed to the Apache Software Foundation (ASF) under one or more
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
package org.apache.mahout.clustering;

import org.apache.mahout.common.parameters.Parametered;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.util.Map;

/**
 * Implementations of this interface have a printable representation and certain
 * attributes that are common across all clustering implementations
 * 
 */
public interface Cluster extends Model<VectorWritable>, Parametered {

  // default directory for initial clusters to prime iterative clustering
  // algorithms
  String INITIAL_CLUSTERS_DIR = "clusters-0";
  
  // default directory for output of clusters per iteration
  String CLUSTERS_DIR = "clusters-";
  
  // default suffix for output of clusters for final iteration
  String FINAL_ITERATION_SUFFIX = "-final";
  
  /**
   * Get the id of the Cluster
   * 
   * @return a unique integer
   * 聚类的唯一ID,即第几个聚类,每一个ID表示一个聚类
   */
  int getId();
  
  /**
   * Get the "center" of the Cluster as a Vector
   * 
   * @return a Vector
   * 该聚类的一个中心点,该中心点实现上是会根据聚类所有点的集合进行不断变化的
   */
  Vector getCenter();
  
  /**
   * Get the "radius" of the Cluster as a Vector. Usually the radius is the
   * standard deviation expressed as a Vector of size equal to the center. Some
   * clusters may return zero values if not appropriate.
   * 
   * @return aVector
   * 表示以该中心点为圆心,多大范围半径内都是该聚类的范围
   */
  Vector getRadius();
    
  /**
   * Produce a custom, human-friendly, printable representation of the Cluster.
   * 
   * @param bindings
   *          an optional String[] containing labels used to format the primary
   *          Vector/s of this implementation.
   * @return a String
   * 格式化输出
   */
  String asFormatString(String[] bindings);

  /**
   * Produce a JSON representation of the Cluster.
   *
   * @param bindings
   *          an optional String[] containing labels used to format the primary
   *          Vector/s of this implementation.
   * @return a Map
   * 格式化输出
   */
  Map<String,Object> asJson(String[] bindings);

  /**
   * @return if the receiver has converged, or false if that has no meaning for
   *         the implementation
   */
  boolean isConverged();
  
}
