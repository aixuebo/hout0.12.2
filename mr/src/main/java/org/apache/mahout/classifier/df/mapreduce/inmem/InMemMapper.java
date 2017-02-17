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

package org.apache.mahout.classifier.df.mapreduce.inmem;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.mahout.classifier.df.Bagging;
import org.apache.mahout.classifier.df.data.Data;
import org.apache.mahout.classifier.df.data.DataLoader;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredMapper;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat.InMemInputSplit;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.RandomUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

/**
 * In-memory mapper that grows the trees using a full copy of the data loaded in-memory. The number of trees
 * to grow is determined by the current InMemInputSplit.
 */
@Deprecated
public class InMemMapper extends MapredMapper<IntWritable,NullWritable,IntWritable,MapredOutput> {
  
  private static final Logger log = LoggerFactory.getLogger(InMemMapper.class);
  
  private Bagging bagging;
  
  private Random rng;

  /**
   * Load the training data
   * 加载数据path,返回数据内容
   */
  private static Data loadData(Configuration conf, Dataset dataset) throws IOException {
    Path dataPath = Builder.getDistributedCacheFile(conf, 1);//加载数据内容
    FileSystem fs = FileSystem.get(dataPath.toUri(), conf);
    return DataLoader.loadData(dataset, fs, dataPath);
  }
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    
    log.info("Loading the data...");
    Data data = loadData(conf, getDataset());//加载数据path,返回数据内容
    log.info("Data loaded : {} instances", data.size());
    
    bagging = new Bagging(getTreeBuilder(), data);
  }
  
  //不考虑数据文件,因为所有数据文件都已经在内存加载好了
  @Override
  protected void map(IntWritable key,//决策树编号--该编号是在全局中的决策树序号
                     NullWritable value,
                     Context context) throws IOException, InterruptedException {
    map(key, context);
  }
  
  void map(IntWritable key, Context context) throws IOException, InterruptedException {
    
	//每一个map任务都随机产生一个随机数
    initRandom((InMemInputSplit) context.getInputSplit());
    
    log.debug("Building...");
    Node tree = bagging.build(rng);//随机构建一颗决策树
    
    if (isOutput()) {
      log.debug("Outputing...");
      MapredOutput mrOut = new MapredOutput(tree);
      
      context.write(key, mrOut);//key不重要,重要的是value
    }
  }
  
  //初始化随机对象
  void initRandom(InMemInputSplit split) {
    if (rng == null) { // first execution of this mapper
      Long seed = split.getSeed();
      log.debug("Initialising rng with seed : {}", seed);
      rng = seed == null ? RandomUtils.getRandom() : RandomUtils.getRandom(seed);
    }
  }
  
}
