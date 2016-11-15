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

package org.apache.mahout.clustering.kmeans;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.google.common.base.Preconditions;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Given an Input Path containing a {@link org.apache.hadoop.io.SequenceFile}, randomly select k vectors and
 * write them to the output file as a {@link org.apache.mahout.clustering.kmeans.Kluster} representing the
 * initial centroid to use.
 *
 * This implementation uses reservoir sampling as described in http://en.wikipedia.org/wiki/Reservoir_sampling
 */
public final class RandomSeedGenerator {
  
  private static final Logger log = LoggerFactory.getLogger(RandomSeedGenerator.class);
  
  public static final String K = "k";
  
  private RandomSeedGenerator() {}

  public static Path buildRandom(Configuration conf, Path input, Path output, int k, DistanceMeasure measure)
    throws IOException {
    return buildRandom(conf, input, output, k, measure, null);
  }

  //创建k个聚类点,随机方式选择聚类中心点,生成在output/part-randomSeed目录下
  //最终选择的中心点依然还是input中的k个点
  public static Path buildRandom(Configuration conf,
                                 Path input,//所有输入点
                                 Path output,//最终随机k个中心点输出在哪里
                                 int k,//要随机生成几个中心点,即要聚类多少个
                                 DistanceMeasure measure,
                                 Long seed)//随机种子
          throws IOException {

    Preconditions.checkArgument(k > 0, "Must be: k > 0, but k = " + k);//k必须大于0
    // delete the output directory 删除原有的输出目录内容
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    HadoopUtil.delete(conf, output);
    Path outFile = new Path(output, "part-randomSeed");//创建随机生成目录
    boolean newFile = fs.createNewFile(outFile);//创建输出文件夹
    if (newFile) {
      Path inputPathPattern;

      if (fs.getFileStatus(input).isDir()) {
        inputPathPattern = new Path(input, "*");
      } else {
        inputPathPattern = input;
      }

      //获得输入文件集合
      FileStatus[] inputFiles = fs.globStatus(inputPathPattern, PathFilters.logsCRCFilter());

      //产生随机数
      Random random = (seed != null) ? RandomUtils.getRandom(seed) : RandomUtils.getRandom();

      List<Text> chosenTexts = new ArrayList<>(k);//存放文件中的key
      List<ClusterWritable> chosenClusters = new ArrayList<>(k);//k个中心点对象,存储文件中的value,而value就是VectorWritable类型的
      int nextClusterId = 0;

      int index = 0;//处理每一条都累加1,即处理了多少条记录
      for (FileStatus fileStatus : inputFiles) {
        if (!fileStatus.isDir()) {
          for (Pair<Writable, VectorWritable> record
              : new SequenceFileIterable<Writable, VectorWritable>(fileStatus.getPath(), true, conf)) {//读取输入源,输入源必须是序列化的文件,并且格式还是Writable, VectorWritable形式的,即value一定是向量形式的
            //读取key和向量形式的value
            Writable key = record.getFirst();
            VectorWritable value = record.getSecond();

            //用value向量做成一个中心点
            Kluster newCluster = new Kluster(value.get(), nextClusterId++, measure);
            newCluster.observe(value.get(), 1);//将该节点进行观察,即加入到中心点中,权重是1

            //最终选择的中心点依然还是input中的k个点
            Text newText = new Text(key.toString());
            int currentSize = chosenTexts.size();//选择的文本大小
            if (currentSize < k) {//如果选择的还没有满足k
              chosenTexts.add(newText);//添加新的key
              ClusterWritable clusterWritable = new ClusterWritable();
              clusterWritable.setValue(newCluster);
              chosenClusters.add(clusterWritable);
            } else {//说明已经选择k个了
              int j = random.nextInt(index);//从处理多少条记录中随机产生一个数字
              if (j < k) {//如果数字<k,说明被随机选择对了
                chosenTexts.set(j, newText);//则设置该值
                ClusterWritable clusterWritable = new ClusterWritable();
                clusterWritable.setValue(newCluster);
                chosenClusters.set(j, clusterWritable);//重新设置中心点
              }
            }
            index++;
          }
        }
      }

      //将chosenTexts的内容写入到输出中,即最终随机中心点已经确认
      try (SequenceFile.Writer writer =
               SequenceFile.createWriter(fs, conf, outFile, Text.class, ClusterWritable.class)){
        for (int i = 0; i < chosenTexts.size(); i++) {
          writer.append(chosenTexts.get(i), chosenClusters.get(i));
        }
        log.info("Wrote {} Klusters to {}", k, outFile);
      }
    }
    
    return outFile;
  }
  
  public static void main(String[] args) {
	System.out.println("aaa");
	
	Random random = RandomUtils.getRandom();
	
	System.out.println("===>"+random);
	for(int i=2;i<100;i++){
		System.out.println(random.nextInt(i));
	}
  }

}
