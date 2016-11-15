/*
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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.kmeans.Kluster;

public class CIReducer extends Reducer<IntWritable,ClusterWritable,IntWritable,ClusterWritable> {
  
  private ClusterClassifier classifier;
  private ClusteringPolicy policy;
  
  /**
   * key就是分类
   * value就是该分类的所有map出来的中心点,value有多少个,则取决于多少个map,因为每一个map都会为每一个分类输出一个中心点
   */
  @Override
  protected void reduce(IntWritable key, Iterable<ClusterWritable> values, Context context) throws IOException,
      InterruptedException {
    Iterator<ClusterWritable> iter = values.iterator();
    //reduce就是中心点,因此以第一个value作为中心点,其他map的中心点进行运算,运算出新的中心点,
    //因此如果reduce中只有一个reduce,则不会有重新计算中心点的逻辑了,因此结果中心点偏移量就是0,等于程序不会再次迭代了,因此map如果只有一个的话,聚类的效果只有一次,是不够准确的,一定要让map有很多,让小文件拆分成若干个更小的文件
    //同时也要注意,小文件拆分成多个文件后,虽然可以聚类了,但是有一个前提,就是文件内容不能有顺序,一定是乱序拆分的,否则有顺序的话,一个map输出的分类也就只有0和1两种值,到reduce中也是无法做到重新汇集同一个分类多个中心点的
    //TODO 可能我理解有问题,一个reduce也没问题,因为该reduce里面包含大量的原始信息,比如s0 s1 s2,也可以进行统计,但是还不清楚为什么我第一次结果不理想的原因
    Cluster first = iter.next().getValue(); // there must always be at least one
    while (iter.hasNext()) {
      Cluster cluster = iter.next().getValue();
      first.observe(cluster);
    }
    List<Cluster> models = new ArrayList<>();
    models.add(first);
    classifier = new ClusterClassifier(models, policy);
    classifier.close();
    context.write(key, new ClusterWritable(first));
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String priorClustersPath = conf.get(ClusterIterator.PRIOR_PATH_KEY);
    classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, new Path(priorClustersPath));
    policy = classifier.getPolicy();
    policy.update(classifier);
    super.setup(context);
  }
  
}
