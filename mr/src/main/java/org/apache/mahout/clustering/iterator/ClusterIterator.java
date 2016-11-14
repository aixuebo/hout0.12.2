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

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.io.Closeables;

/**
 * This is a clustering iterator which works with a set of Vector data and a prior ClusterClassifier which has been
 * initialized with a set of models. Its implementation is algorithm-neutral and works for any iterative clustering
 * algorithm (currently k-means and fuzzy-k-means) that processes all the input vectors in each iteration.
 * The cluster classifier is configured with a ClusteringPolicy to select the desired clustering algorithm.
 */
public final class ClusterIterator {
  
  public static final String PRIOR_PATH_KEY = "org.apache.mahout.clustering.prior.path";//存放每一轮迭代的时候中心节点存储的路径

  private ClusterIterator() {
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations
   *
   * @param data
   *          a {@code List<Vector>} of input vectors
   * @param classifier
   *          a prior ClusterClassifier
   * @param numIterations
   *          the int number of iterations to perform
   * 
   * @return the posterior ClusterClassifier
   * 用于测试环境
   */
  public static ClusterClassifier iterate(Iterable<Vector> data, ClusterClassifier classifier, int numIterations) {
    ClusteringPolicy policy = classifier.getPolicy();
    for (int iteration = 1; iteration <= numIterations; iteration++) {
      for (Vector vector : data) {
        // update the policy based upon the prior
        policy.update(classifier);
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = policy.select(probabilities);
        // training causes all models to observe data
        for (Vector.Element e : weights.nonZeroes()) {
          int index = e.index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
    }
    return classifier;
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations using a sequential
   * implementation
   * 
   * @param conf
   *          the Configuration
   * @param inPath
   *          a Path to input VectorWritables
   * @param priorPath
   *          a Path to the prior classifier
   * @param outPath
   *          a Path of output directory
   * @param numIterations
   *          the int number of iterations to perform
   */
  public static void iterateSeq(Configuration conf, Path inPath, Path priorPath, Path outPath, int numIterations)
    throws IOException {
    ClusterClassifier classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, priorPath);
    Path clustersOut = null;
    int iteration = 1;
    while (iteration <= numIterations) {
      for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(inPath, PathType.LIST,
          PathFilters.logsCRCFilter(), conf)) {
        Vector vector = vw.get();
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = classifier.getPolicy().select(probabilities);
        // training causes all models to observe data
        for (Vector.Element e : weights.nonZeroes()) {
          int index = e.index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
      // update the policy
      classifier.getPolicy().update(classifier);
      // output the classifier
      clustersOut = new Path(outPath, Cluster.CLUSTERS_DIR + iteration);
      classifier.writeToSeqFiles(clustersOut);
      FileSystem fs = FileSystem.get(outPath.toUri(), conf);
      iteration++;
      if (isConverged(clustersOut, conf, fs)) {
        break;
      }
    }
    Path finalClustersIn = new Path(outPath, Cluster.CLUSTERS_DIR + (iteration - 1) + Cluster.FINAL_ITERATION_SUFFIX);
    FileSystem.get(clustersOut.toUri(), conf).rename(clustersOut, finalClustersIn);
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations using a mapreduce
   * implementation
   * 
   * @param conf
   *          the Configuration
   * @param inPath
   *          a Path to input VectorWritables
   * @param priorPath
   *          a Path to the prior classifier
   * @param outPath
   *          a Path of output directory
   * @param numIterations
   *          the int number of iterations to perform
   */
  public static void iterateMR(Configuration conf, Path inPath, Path priorPath, Path outPath, int numIterations)
    throws IOException, InterruptedException, ClassNotFoundException {
    ClusteringPolicy policy = ClusterClassifier.readPolicy(priorPath);
    Path clustersOut = null;
    int iteration = 1;//当前循环次数
    while (iteration <= numIterations) {//循环N次
      conf.set(PRIOR_PATH_KEY, priorPath.toString());
      
      String jobName = "Cluster Iterator running iteration " + iteration + " over priorPath: " + priorPath;
      Job job = new Job(conf, jobName);
      job.setMapOutputKeyClass(IntWritable.class);
      job.setMapOutputValueClass(ClusterWritable.class);
      job.setOutputKeyClass(IntWritable.class);
      job.setOutputValueClass(ClusterWritable.class);
      
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setMapperClass(CIMapper.class);
      job.setReducerClass(CIReducer.class);
      
      FileInputFormat.addInputPath(job, inPath);
      //outPath/clusters-N
      clustersOut = new Path(outPath, Cluster.CLUSTERS_DIR + iteration);//每次迭代次数不一样.目录就不一样
      priorPath = clustersOut;//切换输出目录
      FileOutputFormat.setOutputPath(job, clustersOut);
      
      job.setJarByClass(ClusterIterator.class);
      if (!job.waitForCompletion(true)) {
        throw new InterruptedException("Cluster Iteration " + iteration + " failed processing " + priorPath);
      }
      ClusterClassifier.writePolicy(policy, clustersOut);//设置中心
      FileSystem fs = FileSystem.get(outPath.toUri(), conf);
      iteration++;
      if (isConverged(clustersOut, conf, fs)) {//达到伐值了,则不再进行迭代了
        break;
      }
    }
    
    //outPath/clusters-N-final
    Path finalClustersIn = new Path(outPath, Cluster.CLUSTERS_DIR + (iteration - 1) + Cluster.FINAL_ITERATION_SUFFIX);
    FileSystem.get(clustersOut.toUri(), conf).rename(clustersOut, finalClustersIn);//将最后一次循环的结果集,改名为outPath/clusters-N-final
  }
  
  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   * 判断是否达到伐值了,true表示不用再迭代了          
   */
  private static boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {
    for (FileStatus part : fs.listStatus(filePath, PathFilters.partFilter())) {
      SequenceFileValueIterator<ClusterWritable> iterator = new SequenceFileValueIterator<>(
          part.getPath(), true, conf);
      while (iterator.hasNext()) {
        ClusterWritable value = iterator.next();
        if (!value.getValue().isConverged()) {//必须全部达到伐值,才最终不会被迭代
          Closeables.close(iterator, true);
          return false;
        }
      }
    }
    return true;
  }
}
