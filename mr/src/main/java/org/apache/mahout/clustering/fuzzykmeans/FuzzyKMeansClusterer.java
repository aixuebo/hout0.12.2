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

package org.apache.mahout.clustering.fuzzykmeans;

import java.util.Collection;
import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * 该类用于FuzzyKMeansClusteringPolicy类计算使用
 */
public class FuzzyKMeansClusterer {

  private static final double MINIMAL_VALUE = 0.0000000001;//最小概率,最后会获取倒数,因此就变成最大概率了
  
  private double m = 2.0; // default value 设置模糊因子,即-m对应的参数
  
  /**
   * 真正的计算
   * @param clusters 中心点集合
   * @param clusterDistanceList 该点与每一个中心点对应的距离集合
   * @return
   */
  public Vector computePi(Collection<SoftCluster> clusters, List<Double> clusterDistanceList) {
    Vector pi = new DenseVector(clusters.size());//设置一个密集向量,size的长度就是中心点的大小
    for (int i = 0; i < clusters.size(); i++) {
      double probWeight = computeProbWeight(clusterDistanceList.get(i), clusterDistanceList);//计算权重
      pi.set(i, probWeight);//设置该点到每一个中心点的权重
    }
    return pi;
  }
  
  /**
   * Computes the probability of a point belonging to a cluster
   * 计算该点属于哪个分类的概率 
   * @param clusterDistance 该点到某一个中心点的距离
   * @param clusterDistanceList 该点到所有中心点的距离
   * @return
   */
  public double computeProbWeight(double clusterDistance, Iterable<Double> clusterDistanceList) {
    if (clusterDistance == 0) {//如果该点到这个中心点的距离为0,说明非常接近了,因此概率就是设置为最小概率
      clusterDistance = MINIMAL_VALUE;
    }
    
    /**
     * 分析公式
     * Math.pow(clusterDistance / eachCDist, 2.0 / (m - 1));
     * 1.clusterDistance / eachCDist 表示当前点与该中心点的距离 与每一个其他中心点的距离做比较,该值越小,说明越接近,
     * 比如该点到3个中心点的距离分别是 5 12 18
     * 因此5/5 5/12 5/18 就是最终得知
     * 2. 用1的结果*2.0 / (m - 1)次方,其中m就是因子
     * 3.最后要1/denom,意思是取倒数,因为上述1和2的计算是距离越近,值越小,因此获取倒数后,就是概率越大的意思
     */
    double denom = 0.0;
    for (double eachCDist : clusterDistanceList) {
      if (eachCDist == 0.0) {
        eachCDist = MINIMAL_VALUE;
      }
      denom += Math.pow(clusterDistance / eachCDist, 2.0 / (m - 1));
    }
    return 1.0 / denom;
  }

  public void setM(double m) {
    this.m = m;
  }
}
