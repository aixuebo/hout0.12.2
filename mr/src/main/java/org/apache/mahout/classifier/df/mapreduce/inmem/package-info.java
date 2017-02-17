/**
 * <h2>In-memory mapreduce implementation of Random Decision Forests</h2>
 * 在内存中实现随机决策森林
 * 
 * <p>Each mapper is responsible for growing a number of trees with a whole copy of the dataset loaded in memory,
 * it uses the reference implementation's code to build each tree and estimate the oob error.</p>
 * 每一个map负责增长一颗决策树,使用全部数据加载到内存中去实现一颗随机决策树,因此可能有内存溢出的问题要考虑
 * 
 * 
 * <p>The dataset is distributed to the slave nodes using the {@link org.apache.hadoop.filecache.DistributedCache}.
 * A custom {@link org.apache.hadoop.mapreduce.InputFormat}
 * ({@link org.apache.mahout.classifier.df.mapreduce.inmem.InMemInputFormat}) is configured with the
 * desired number of trees and generates a number of {@link org.apache.hadoop.mapreduce.InputSplit}s
 * equal to the configured number of maps.</p>
 *
 * <p>There is no need for reducers, each map outputs (the trees it built and, for each tree, the labels the
 * tree predicted for each out-of-bag instance. This step has to be done in the mapper because only there we
 * know which instances are o-o-b.</p>
 * 不需要reduce,每一个map的输出
 *
 * <p>The Forest builder ({@link org.apache.mahout.classifier.df.mapreduce.inmem.InMemBuilder}) is responsible
 * for configuring and launching the job.
 * At the end of the job it parses the output files and builds the corresponding
 * {@link org.apache.mahout.classifier.df.DecisionForest}.</p>
 */
package org.apache.mahout.classifier.df.mapreduce.inmem;
