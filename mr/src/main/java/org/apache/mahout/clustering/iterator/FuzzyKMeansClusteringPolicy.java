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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.fuzzykmeans.FuzzyKMeansClusterer;
import org.apache.mahout.clustering.fuzzykmeans.SoftCluster;
import org.apache.mahout.math.Vector;

/**
 * This is a probability-weighted clustering policy, suitable for fuzzy k-means
 * clustering
 * 进行模糊查询时候的计算规则代理类
 */
public class FuzzyKMeansClusteringPolicy extends AbstractClusteringPolicy {

  private double m = 2;//-m模糊因子
  private double convergenceDelta = 0.05;//伐值,小于该伐值的距离就停止聚类

  public FuzzyKMeansClusteringPolicy() {
  }

  public FuzzyKMeansClusteringPolicy(double m, double convergenceDelta) {
    this.m = m;
    this.convergenceDelta = convergenceDelta;
  }

  //返回该节点对应所有分类的概率向量,因为模糊kmeans就是指代一个点可以属于多个分类,因此没必要在从向量中选择N个了,直接使用概率向量即可
  @Override
  public Vector select(Vector probabilities) {
    return probabilities;
  }
  
  //模糊kmeans重新定义了该方法,即不再使用pdf方法了
  @Override
  public Vector classify(Vector data, ClusterClassifier prior) {
	//使用两个集合,分别代表该点与每一个中心点对应的距离  
    Collection<SoftCluster> clusters = new ArrayList<>();//存储中心点
    List<Double> distances = new ArrayList<>();//存储该点与中心点对应的距离,至于与哪个中心点的距离,取决于clusters参数对应的位置
    
    //设置中心点集合和距离集合
    for (Cluster model : prior.getModels()) {
      SoftCluster sc = (SoftCluster) model;
      clusters.add(sc);
      distances.add(sc.getMeasure().distance(data, sc.getCenter()));
    }
    
    FuzzyKMeansClusterer fuzzyKMeansClusterer = new FuzzyKMeansClusterer();
    fuzzyKMeansClusterer.setM(m);
    return fuzzyKMeansClusterer.computePi(clusters, distances);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeDouble(m);
    out.writeDouble(convergenceDelta);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    this.m = in.readDouble();
    this.convergenceDelta = in.readDouble();
  }

  @Override
  public void close(ClusterClassifier posterior) {
    for (Cluster cluster : posterior.getModels()) {//关闭前设置是否达到伐值,以及重新计算一下中心点位置
      ((org.apache.mahout.clustering.kmeans.Kluster) cluster).calculateConvergence(convergenceDelta);
      cluster.computeParameters();
    }
    
  }
  
}
