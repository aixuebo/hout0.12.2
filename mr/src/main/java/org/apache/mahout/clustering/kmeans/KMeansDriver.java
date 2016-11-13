/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassificationDriver;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.topdown.PathDirectory;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * keams聚类的入口
 */
public class KMeansDriver extends AbstractJob {
  
  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);
  
  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new KMeansDriver(), args);
  }
  
  @Override
  public int run(String[] args) throws Exception {
    
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());//距离算法

    ////中心点路径
    addOption(DefaultOptionCreator
        .clustersInOption()
        .withDescription(
            "The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
                + "If k is also specified, then a random set of vectors will be selected"
                + " and written out to this path first").create());

    //聚类的数量,即k
    addOption(DefaultOptionCreator
        .numClustersOption()
        .withDescription(
            "The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
                + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.useSetRandomSeedOption().create());//随机数
    addOption(DefaultOptionCreator.convergenceOption().create());//距离相差多少后,就不在聚类了
    addOption(DefaultOptionCreator.maxIterationsOption().create());//最大循环次数
    addOption(DefaultOptionCreator.overwriteOption().create());//是否覆盖原有目录
    addOption(DefaultOptionCreator.clusteringOption().create());
    addOption(DefaultOptionCreator.methodOption().create());//是本地还是集群
    addOption(DefaultOptionCreator.outlierThresholdOption().create());
   
    if (parseArguments(args) == null) {
      return -1;
    }
    
    Path input = getInputPath();
    Path clusters = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));//clusters
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }

    //距离相差多少后,就不在聚类了
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));

    //最大循环次数
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));

    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.delete(getConf(), output);
    }
    DistanceMeasure measure = ClassUtils.instantiateAs(measureClass, DistanceMeasure.class);

    //聚类的数量,即k
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      int numClusters = Integer.parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION));

      Long seed = null;
      if (hasOption(DefaultOptionCreator.RANDOM_SEED)) {
        seed = Long.parseLong(getOption(DefaultOptionCreator.RANDOM_SEED));
      }

      clusters = RandomSeedGenerator.buildRandom(getConf(), input, clusters, numClusters, measure, seed);
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    boolean runSequential = getOption(DefaultOptionCreator.METHOD_OPTION).equalsIgnoreCase(
        DefaultOptionCreator.SEQUENTIAL_METHOD);
    double clusterClassificationThreshold = 0.0;
    if (hasOption(DefaultOptionCreator.OUTLIER_THRESHOLD)) {
      clusterClassificationThreshold = Double.parseDouble(getOption(DefaultOptionCreator.OUTLIER_THRESHOLD));
    }
    run(getConf(), input, clusters, output, convergenceDelta, maxIterations, runClustering,
        clusterClassificationThreshold, runSequential);
    return 0;
  }
  
  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param runClustering
   *          true if points are to be clustered after iterations are completed
   * @param clusterClassificationThreshold
   *          Is a clustering strictness / outlier removal parameter. Its value should be between 0 and 1. Vectors
   *          having pdf below this value will not be clustered.
   * @param runSequential
   *          if true execute sequential algorithm
   */
  public static void run(Configuration conf, Path input, Path clustersIn, Path output,
    double convergenceDelta, int maxIterations, boolean runClustering, double clusterClassificationThreshold,
    boolean runSequential) throws IOException, InterruptedException, ClassNotFoundException {
    
    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);//聚类距离相差多少后,就不在聚类了
    if (log.isInfoEnabled()) {
      log.info("Input: {} Clusters In: {} Out: {}", input, clustersIn, output);
      log.info("convergence: {} max Iterations: {}", convergenceDelta, maxIterations);
    }
    Path clustersOut = buildClusters(conf, input, clustersIn, output, maxIterations, delta, runSequential);
    if (runClustering) {
      log.info("Clustering data");
      clusterData(conf, input, clustersOut, output, clusterClassificationThreshold, runSequential);
    }
  }
  
  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the results of the final iteration to
   * cluster the input vectors.
   *
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param runClustering
   *          true if points are to be clustered after iterations are completed
   * @param clusterClassificationThreshold
   *          Is a clustering strictness / outlier removal parameter. Its value should be between 0 and 1. Vectors
   *          having pdf below this value will not be clustered.
   * @param runSequential
   *          if true execute sequential algorithm
   */
  public static void run(Path input, Path clustersIn, Path output, double convergenceDelta,
    int maxIterations, boolean runClustering, double clusterClassificationThreshold, boolean runSequential)
    throws IOException, InterruptedException, ClassNotFoundException {
    run(new Configuration(), input, clustersIn, output, convergenceDelta, maxIterations, runClustering,
        clusterClassificationThreshold, runSequential);
  }
  
  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * 
   *
   * @param conf
   *          the Configuration to use
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param maxIterations
   *          the maximum number of iterations
   * @param delta
   *          the convergence delta value
   * @param runSequential
   *          if true execute sequential algorithm
   *
   * @return the Path of the final clusters directory
   */
  public static Path buildClusters(Configuration conf, Path input, Path clustersIn, Path output,
    int maxIterations, String delta, boolean runSequential) throws IOException,
    InterruptedException, ClassNotFoundException {
    
    double convergenceDelta = Double.parseDouble(delta);
    List<Cluster> clusters = new ArrayList<>();
    KMeansUtil.configureWithClusterInfo(conf, clustersIn, clusters);
    
    if (clusters.isEmpty()) {
      throw new IllegalStateException("No input clusters found in " + clustersIn + ". Check your -c argument.");
    }
    
    Path priorClustersPath = new Path(output, Cluster.INITIAL_CLUSTERS_DIR);
    ClusteringPolicy policy = new KMeansClusteringPolicy(convergenceDelta);
    ClusterClassifier prior = new ClusterClassifier(clusters, policy);
    prior.writeToSeqFiles(priorClustersPath);
    
    if (runSequential) {
      ClusterIterator.iterateSeq(conf, input, priorClustersPath, output, maxIterations);
    } else {
      ClusterIterator.iterateMR(conf, input, priorClustersPath, output, maxIterations);
    }
    return output;
  }
  
  /**
   * Run the job using supplied arguments
   *
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output points
   * @param clusterClassificationThreshold
   *          Is a clustering strictness / outlier removal parameter. Its value should be between 0 and 1. Vectors
   *          having pdf below this value will not be clustered.
   * @param runSequential
   *          if true execute sequential algorithm
   */
  public static void clusterData(Configuration conf, Path input, Path clustersIn, Path output,
    double clusterClassificationThreshold, boolean runSequential) throws IOException, InterruptedException,
    ClassNotFoundException {
    
    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: {} Clusters In: {} Out: {}", input, clustersIn, output);
    }
    ClusterClassifier.writePolicy(new KMeansClusteringPolicy(), clustersIn);
    ClusterClassificationDriver.run(conf, input, output, new Path(output, PathDirectory.CLUSTERED_POINTS_DIRECTORY),
        clusterClassificationThreshold, true, runSequential);
  }
  
}
