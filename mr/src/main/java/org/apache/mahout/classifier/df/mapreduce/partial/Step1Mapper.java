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

package org.apache.mahout.classifier.df.mapreduce.partial;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.df.Bagging;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredMapper;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * First step of the Partial Data Builder. Builds the trees using the data available in the InputSplit.
 * Predict the oob classes for each tree in its growing partition (input split).
 */
@Deprecated
public class Step1Mapper extends MapredMapper<LongWritable,Text,TreeID,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(Step1Mapper.class);
  
  /** used to convert input values to data instances */
  private DataConverter converter;//如何将String类型的内容转换成instances对象
  
  private Random rng;
  
  /** number of trees to be built by this mapper 这个map要产生多少颗决策树*/
  private int nbTrees;
  
  /** id of the first tree 第一课决策树的id*/
  private int firstTreeId;
  
  /** mapper's partition 当前是第几个partition对应的map*/
  private int partition;
  
  /** will contain all instances if this mapper's split 该map要处理的所有数据集合,即在该集合内创建了若干个决策树*/
  private final List<Instance> instances = new ArrayList<>();
  
  public int getFirstTreeId() {
    return firstTreeId;
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    
    configure(Builder.getRandomSeed(conf), conf.getInt("mapred.task.partition", -1),
      Builder.getNumMaps(conf), Builder.getNbTrees(conf));
  }
  
  /**
   * Useful when testing
   * 
   * @param partition
   *          current mapper inputSplit partition
   * @param numMapTasks
   *          number of running map tasks
   * @param numTrees
   *          total number of trees in the forest
   */
  protected void configure(Long seed, int partition, int numMapTasks, int numTrees) {
    converter = new DataConverter(getDataset());
    
    // prepare random-numders generator
    log.debug("seed : {}", seed);
    if (seed == null) {
      rng = RandomUtils.getRandom();
    } else {
      rng = RandomUtils.getRandom(seed);
    }
    
    // mapper's partition
    Preconditions.checkArgument(partition >= 0, "Wrong partition ID: " + partition + ". Partition must be >= 0!");
    this.partition = partition;
    
    // compute number of trees to build
    nbTrees = nbTrees(numMapTasks, numTrees, partition);
    
    // compute first tree id
    firstTreeId = 0;
    for (int p = 0; p < partition; p++) {
      firstTreeId += nbTrees(numMapTasks, numTrees, p);//计算当前partition前应该已经有多少个决策树了,因此知道了本节点的map应该从第几颗决策树开始
    }
    
    log.debug("partition : {}", partition);
    log.debug("nbTrees : {}", nbTrees);
    log.debug("firstTreeId : {}", firstTreeId);
  }
  
  /**
   * Compute the number of trees for a given partition. The first partitions may be longer
   * than the rest because of the remainder.
   * 计算该map上要产生多少个决策树
   * @param numMaps
   *          total number of maps (partitions) 总共有多少个map任务
   * @param numTrees
   *          total number of trees to build 总共要产生多少颗决策树
   * @param partition
   *          partition to compute the number of trees for 当前是第几个partition
   */
  public static int nbTrees(int numMaps, int numTrees, int partition) {
    int treesPerMapper = numTrees / numMaps;//计算每个map上应该有多少颗决策树
    int remainder = numTrees - numMaps * treesPerMapper;//剩余多少颗决策树不能被分配到map中
    return treesPerMapper + (partition < remainder ? 1 : 0);//将前几个partition中多分配一个多余的决策树
  }
  
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
    instances.add(converter.convert(value.toString()));//将value的字符串转换成Instance对象
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    // prepare the data
    log.debug("partition: {} numInstances: {}", partition, instances.size());
    
    Data data = new Data(getDataset(), instances);//组装成统计的数据源集合
    Bagging bagging = new Bagging(getTreeBuilder(), data);
    
    TreeID key = new TreeID();
    
    log.debug("Building {} trees", nbTrees);
    for (int treeId = 0; treeId < nbTrees; treeId++) {
      log.debug("Building tree number : {}", treeId);
      
      Node tree = bagging.build(rng);//产生一颗决策树
      
      key.set(partition, firstTreeId + treeId);//为该决策树分配一个ID,
      
      if (isOutput()) {
        MapredOutput emOut = new MapredOutput(tree);
        context.write(key, emOut);//key就是决策树的ID,value是决策树内容
      }

      context.progress();
    }
  }
  
}
