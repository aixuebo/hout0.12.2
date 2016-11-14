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
package org.apache.mahout.clustering.iterator;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.math.Vector;

/**
 * A ClusteringPolicy captures the semantics of assignment of points to clusters
 * 
 */
public interface ClusteringPolicy extends Writable {
  
  /**
   * Classify the data vector given the classifier's models
   * 
   * @param data
   *          a data Vector
   * @param prior
   *          a prior ClusterClassifier
   * @return a Vector of probabilities that the data is described by each of the
   *         models
   *  属于属于哪个分类的概率,返回的向量是概率向量,即如果有5个分类,则返回的向量就是5个值,表示属于每一个分类的概率        
   */
  Vector classify(Vector data, ClusterClassifier prior);
  
  /**
   * Return a vector of weights for each of the models given those probabilities
   * 参数是classify方法返回值,即一个元素在各个分类上的概率
   * @param probabilities
   *          a Vector of pdfs
   * @return a Vector of weights
   * 返回值是该向量真的属于哪些分类,非0的元素都是要属于的分类,具体值就是属于该分类的权重
   */
  Vector select(Vector probabilities);
  
  /**
   * Update the policy with the given classifier
   * 
   * @param posterior
   *          a ClusterClassifier、
   * 更新一个代理对象         
   */
  void update(ClusterClassifier posterior);
  
  /**
   * Close the policy using the classifier's models
   * 
   * @param posterior
   *          a posterior ClusterClassifier
   * 重新计算每一个分类的中心值
   */
  void close(ClusterClassifier posterior);
  
}
