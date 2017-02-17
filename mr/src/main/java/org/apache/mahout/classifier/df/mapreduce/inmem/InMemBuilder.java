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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.builder.TreeBuilder;
import org.apache.mahout.classifier.df.mapreduce.Builder;
import org.apache.mahout.classifier.df.mapreduce.MapredOutput;
import org.apache.mahout.classifier.df.node.Node;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;

/**
 * MapReduce implementation where each mapper loads a full copy of the data in-memory. The forest trees are
 * splitted across all the mappers
 * 每一个map加载所有的数据到内存中,产生一颗决策树,决策树森林是所有的map的结果组成的
 */
@Deprecated
public class InMemBuilder extends Builder {
  
  //给定数据内容路径 以及 数据title路径
  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath, Long seed, Configuration conf) {
    super(treeBuilder, dataPath, datasetPath, seed, conf);
  }
  
  public InMemBuilder(TreeBuilder treeBuilder, Path dataPath, Path datasetPath) {
    this(treeBuilder, dataPath, datasetPath, null, new Configuration());
  }
  
  //多少个map任务,就有多少颗决策树产生
  @Override
  protected void configureJob(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    job.setJarByClass(InMemBuilder.class);
    
    FileOutputFormat.setOutputPath(job, getOutputPath(conf));
    
    // put the data in the DistributedCache
    DistributedCache.addCacheFile(getDataPath().toUri(), conf);//数据内容路径
    
    job.setOutputKeyClass(IntWritable.class);//key 决策树编号--该编号是在全局中的决策树序号
    job.setOutputValueClass(MapredOutput.class);//value就是一颗决策树的node流程
    
    job.setMapperClass(InMemMapper.class);
    job.setNumReduceTasks(0); // no reducers 不用reduce
    
    job.setInputFormatClass(InMemInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    
  }
  
  //用于最终job的日志输出,转换成DecisionForest对象
  @Override
  protected DecisionForest parseOutput(Job job) throws IOException {
    Configuration conf = job.getConfiguration();
    
    Map<Integer,MapredOutput> output = new HashMap<>();
    
    Path outputPath = getOutputPath(conf);
    FileSystem fs = outputPath.getFileSystem(conf);
    
    Path[] outfiles = DFUtils.listOutputFiles(fs, outputPath);
    
    // import the InMemOutputs
    for (Path path : outfiles) {
      for (Pair<IntWritable,MapredOutput> record : new SequenceFileIterable<IntWritable,MapredOutput>(path, conf)) {
        output.put(record.getFirst().get(), record.getSecond());
      }
    }
    
    return processOutput(output);
  }
  
  /**
   * Process the output, extracting the trees
   * 会根据所有的决策树结果生成决策森林
   */
  private static DecisionForest processOutput(Map<Integer,MapredOutput> output) {
    List<Node> trees = new ArrayList<>();
    
    for (Map.Entry<Integer,MapredOutput> entry : output.entrySet()) {
      MapredOutput value = entry.getValue();
      trees.add(value.getTree());
    }
    
    return new DecisionForest(trees);
  }
}
