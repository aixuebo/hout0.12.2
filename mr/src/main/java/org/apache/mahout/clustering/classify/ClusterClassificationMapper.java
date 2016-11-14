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

package org.apache.mahout.clustering.classify;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.DistanceMeasureCluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterator;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.VectorWritable;

/**
 * Mapper for classifying vectors into clusters.
 * 输入源:输入目录、分类中心点集合、输出目录等等
 * 输出:每一个点所属分类clusterId,以及WeightedPropertyVectorWritable对象,
 * 该WeightedPropertyVectorWritable对象包含了该点具体向量、该点属于该分类的概率(权重),该点与该分类的真实距离。
 * 注意:此时是没有reduce操作的,因此可以点可以属于多个分类
 */
public class ClusterClassificationMapper extends
    Mapper<WritableComparable<?>,VectorWritable,IntWritable,WeightedVectorWritable> {
  
  //设置该伐值,一个点和多个分类都很相近,因此该点在每一个分类的打分都不会太高,因此就会小于该伐值,就不会要这个点了
  private double threshold;//判断概率的最大值是否大于一个伐值,如果大于该伐值,说明可以是存在一个分类中
  private List<Cluster> clusterModels;//中心向量集合
  private ClusterClassifier clusterClassifier;
  private IntWritable clusterId;//所属分类,临时变量
  private boolean emitMostLikely;//true表示一个点必须只能属于一个分类,false表示一个点可以属于多个分类
  
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    
    Configuration conf = context.getConfiguration();
    String clustersIn = conf.get(ClusterClassificationConfigKeys.CLUSTERS_IN);
    threshold = conf.getFloat(ClusterClassificationConfigKeys.OUTLIER_REMOVAL_THRESHOLD, 0.0f);
    emitMostLikely = conf.getBoolean(ClusterClassificationConfigKeys.EMIT_MOST_LIKELY, false);
    
    clusterModels = new ArrayList<>();
    
    if (clustersIn != null && !clustersIn.isEmpty()) {
      Path clustersInPath = new Path(clustersIn);
      //读取clusterOutputPath下clusters-xxx-final文件,理论上只有一个该文件,因此就读取该文件记录的中心点
      clusterModels = populateClusterModels(clustersInPath, conf);
      ClusteringPolicy policy = ClusterClassifier.readPolicy(finalClustersPath(clustersInPath));//读取clusterOutputPath下clusters-xxx-final文件,理论上只有一个该文件,因此就读取一个即可
      clusterClassifier = new ClusterClassifier(clusterModels, policy);
    }
    clusterId = new IntWritable();//仅仅初始化临时变量
  }
  
  /**
   * Mapper which classifies the vectors to respective clusters.
   */
  @Override
  protected void map(WritableComparable<?> key, VectorWritable vw, Context context)
    throws IOException, InterruptedException {
    if (!clusterModels.isEmpty()) {
      // Converting to NamedVectors to preserve the vectorId else its not obvious as to which point
      // belongs to which cluster - fix for MAHOUT-1410
      Class<? extends Vector> vectorClass = vw.get().getClass();//向量对应的class
      Vector vector = vw.get();//向量实例
      
      //如果向量不是NamedVector类型的,即向量没有描述符,则将key作为向量的描述符,生成NamedVector对象
      if (!vectorClass.equals(NamedVector.class)) {
        if (key.getClass().equals(Text.class)) {
          vector = new NamedVector(vector, key.toString());
        } else if (key.getClass().equals(IntWritable.class)) {
          vector = new NamedVector(vector, Integer.toString(((IntWritable) key).get()));
        }
      }
      
      //现在vector向量就一定是NamedVector类型的向量了
      Vector pdfPerCluster = clusterClassifier.classify(vector);//对该向量与给定中心集合做比较,返回概率打分集合
      
      //判断概率的最大值是否大于一个伐值,如果大于该伐值,说明可以是存在一个分类中
      if (shouldClassify(pdfPerCluster)) {//说明该分类打分符合一个分类
        if (emitMostLikely) {//仅保存最应该属于的分类,即一个点仅对应一个分类
          int maxValueIndex = pdfPerCluster.maxValueIndex();//最大分所在序号
          write(new VectorWritable(vector), context, maxValueIndex, 1.0);
        } else {//将打分在伐值以上的分类,让这个点都属于这些分类,即一个点对应多个分类
          writeAllAboveThreshold(new VectorWritable(vector), context, pdfPerCluster);
        }
      }
    }
  }
  
  /**
   * 保存所有大于一定伐值的分类 
   * @param vw 要保存的向量点
   * @param context
   * @param pdfPerCluster vw该点对每一个中心向量的打分
   * @throws IOException
   * @throws InterruptedException
   */
  private void writeAllAboveThreshold(VectorWritable vw, Context context,
      Vector pdfPerCluster) throws IOException, InterruptedException {
	  
    for (Element pdf : pdfPerCluster.nonZeroes()) {//返回打分非0的
      if (pdf.get() >= threshold) {//说明该分类的打分符合伐值,要保存该点到该分类中
        int clusterIndex = pdf.index();//获取分类对应的序号
        write(vw, context, clusterIndex, pdf.get());
      }
    }
  }
  
  /**
   * 真实的向集群写入数据
   * @param vw 要保存的向量点
   * @param context
   * @param clusterIndex 该vw点划分到属于哪个中心向量的分类中
   * @param weight 该vw点属于该中心向量的权重,即打分
   * @throws IOException
   * @throws InterruptedException
   */
  private void write(VectorWritable vw, Context context, int clusterIndex, double weight)
    throws IOException, InterruptedException {
    Cluster cluster = clusterModels.get(clusterIndex);//中心分类对象
    clusterId.set(cluster.getId());//属于第几个分类

    DistanceMeasureCluster distanceMeasureCluster = (DistanceMeasureCluster) cluster;
    DistanceMeasure distanceMeasure = distanceMeasureCluster.getMeasure();
    double distance = distanceMeasure.distance(cluster.getCenter(), vw.get());//计算要保存的向量点与中心向量之间的距离

    Map<Text, Text> props = new HashMap<>();
    props.put(new Text("distance"), new Text(Double.toString(distance)));//真实的距离
    context.write(clusterId, new WeightedPropertyVectorWritable(weight, vw.get(), props));//保存权重向量,即包含权重值,向量点,以及属性
  }
  
  //读取clusterOutputPath下clusters-xxx-final文件,理论上只有一个该文件,因此就读取该文件记录的中心点
  public static List<Cluster> populateClusterModels(Path clusterOutputPath, Configuration conf) throws IOException {
    List<Cluster> clusters = new ArrayList<>();
    FileSystem fileSystem = clusterOutputPath.getFileSystem(conf);
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());//读取clusterOutputPath/clusters-xxx-final文件
    
    //理论上只有一个该文件,因此就读取一个即可
    Iterator<?> it = new SequenceFileDirValueIterator<Writable>(
        clusterFiles[0].getPath(), PathType.LIST, PathFilters.partFilter(),
        null, false, conf);
    while (it.hasNext()) {
      ClusterWritable next = (ClusterWritable) it.next();
      Cluster cluster = next.getValue();
      cluster.configure(conf);
      clusters.add(cluster);
    }
    return clusters;
  }
  
  //判断概率的最大值是否大于一个伐值,如果大于该伐值,说明可以是存在一个分类中
  private boolean shouldClassify(Vector pdfPerCluster) {
    return pdfPerCluster.maxValue() >= threshold;//设置该伐值,一个点和多个分类都很相近,因此该点在每一个分类的打分都不会太高,因此就会小于该伐值,就不会要这个点了
  }
  
  //读取clusterOutputPath下clusters-xxx-final文件,理论上只有一个该文件,因此就读取一个即可
  private static Path finalClustersPath(Path clusterOutputPath) throws IOException {
    System.out.println("finalClustersPathfinalClustersPath:"+clusterOutputPath.toUri());
    FileSystem fileSystem = clusterOutputPath.getFileSystem(new Configuration());
      System.out.println("====fileSystem:"+fileSystem.toString());
    FileStatus[] clusterFiles = fileSystem.listStatus(clusterOutputPath, PathFilters.finalPartFilter());//读取clusterOutputPath/clusters-xxx-final文件
      System.out.println("====clusterFiles:"+ clusterFiles[0].getPath());
    return clusterFiles[0].getPath();
  }
}
