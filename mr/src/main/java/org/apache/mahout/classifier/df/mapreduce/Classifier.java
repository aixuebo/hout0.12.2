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

package org.apache.mahout.classifier.df.mapreduce;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.classifier.df.DFUtils;
import org.apache.mahout.classifier.df.DecisionForest;
import org.apache.mahout.classifier.df.data.DataConverter;
import org.apache.mahout.classifier.df.data.Dataset;
import org.apache.mahout.classifier.df.data.Instance;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mapreduce implementation that classifies the Input data using a previousely built decision forest
 */
@Deprecated
public class Classifier {

  private static final Logger log = LoggerFactory.getLogger(Classifier.class);

  private final Configuration conf;
  
  private final Path forestPath;//决策树森林路径
  private final Path datasetPath;//数据title路径
  private final Path inputPath;//数据源的输入路径
  private final Path outputPath; // path that will containt the final output of the classifier 最终的输出目录
  private final Path mappersOutputPath; // mappers will output here 每一个mapp的输出目录
  private double[][] results;//最终所有的key-value输出,真实标签和预测标签
  
  public double[][] getResults() {
    return results;
  }

  public Classifier(Path forestPath,
                    Path inputPath,
                    Path datasetPath,
                    Path outputPath,
                    Configuration conf) {
    this.forestPath = forestPath;
    this.inputPath = inputPath;
    this.datasetPath = datasetPath;
    this.outputPath = outputPath;
    this.conf = conf;

    mappersOutputPath = new Path(outputPath, "mappers");
  }

  private void configureJob(Job job) throws IOException {

    job.setJarByClass(Classifier.class);

    FileInputFormat.setInputPaths(job, inputPath);
    FileOutputFormat.setOutputPath(job, mappersOutputPath);

    job.setOutputKeyClass(DoubleWritable.class);
    job.setOutputValueClass(Text.class);

    job.setMapperClass(CMapper.class);
    job.setNumReduceTasks(0); // no reducers

    job.setInputFormatClass(CTextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

  }

  public void run() throws IOException, ClassNotFoundException, InterruptedException {
    FileSystem fs = FileSystem.get(conf);

    // check the output
    if (fs.exists(outputPath)) {
      throw new IOException("Output path already exists : " + outputPath);
    }

    log.info("Adding the dataset to the DistributedCache");
    // put the dataset into the DistributedCache
    DistributedCache.addCacheFile(datasetPath.toUri(), conf);

    log.info("Adding the decision forest to the DistributedCache");
    DistributedCache.addCacheFile(forestPath.toUri(), conf);

    Job job = new Job(conf, "decision forest classifier");

    log.info("Configuring the job...");
    configureJob(job);

    log.info("Running the job...");
    if (!job.waitForCompletion(true)) {
      throw new IllegalStateException("Job failed!");
    }

    parseOutput(job);

    HadoopUtil.delete(conf, mappersOutputPath);
  }

  /**
   * Extract the prediction for each mapper and write them in the corresponding output file. 
   * The name of the output file is based on the name of the corresponding input file.
   * Will compute the ConfusionMatrix if necessary.
   */
  private void parseOutput(JobContext job) throws IOException {
    Configuration conf = job.getConfiguration();
    FileSystem fs = mappersOutputPath.getFileSystem(conf);

    Path[] outfiles = DFUtils.listOutputFiles(fs, mappersOutputPath);//获取所有的map输出结果

    // read all the output
    List<double[]> resList = new ArrayList<>();//存储所有的key-value信息,每一个key-value组成一个double数组
    for (Path path : outfiles) {
      FSDataOutputStream ofile = null;
      try {
        for (Pair<DoubleWritable,Text> record : new SequenceFileIterable<DoubleWritable,Text>(path, true, conf)) {
          double key = record.getFirst().get();
          String value = record.getSecond().toString();
          if (ofile == null) {
            // this is the first value, it contains the name of the input file 输入路径的名字--即每一个输入路径对应一个输出,内容是真实的标签
            ofile = fs.create(new Path(outputPath, value).suffix(".out"));
          } else {
            // The key contains the correct label of the data. The value contains a prediction
            ofile.writeChars(value); // write the prediction
            ofile.writeChar('\n');

            resList.add(new double[]{key, Double.valueOf(value)});//添加真实标签和预测标签
          }
        }
      } finally {
        Closeables.close(ofile, false);
      }
    }
    results = new double[resList.size()][2];
    resList.toArray(results);
  }

  /**
   * TextInputFormat that does not split the input files. This ensures that each input file is processed by one single
   * mapper.
   */
  private static class CTextInputFormat extends TextInputFormat {
    @Override
    protected boolean isSplitable(JobContext jobContext, Path path) {
      return false;
    }
  }
  
  public static class CMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {

    /** used to convert input values to data instances */
    private DataConverter converter;//用于将字符串的输入源转换成Instance对象
    private DecisionForest forest;//表示一个决策森林
    private final Random rng = RandomUtils.getRandom();
    private boolean first = true;
    private final Text lvalue = new Text();
    private Dataset dataset;//数据title
    private final DoubleWritable lkey = new DoubleWritable();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
      super.setup(context);    //To change body of overridden methods use File | Settings | File Templates.

      Configuration conf = context.getConfiguration();

      Path[] files = HadoopUtil.getCachedFiles(conf);

      if (files.length < 2) {
        throw new IOException("not enough paths in the DistributedCache");
      }
      dataset = Dataset.load(conf, files[0]);
      converter = new DataConverter(dataset);

      forest = DecisionForest.load(conf, files[1]);//加载决策森林
      if (forest == null) {
        throw new InterruptedException("DecisionForest not found!");
      }
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
      if (first) {
        FileSplit split = (FileSplit) context.getInputSplit();
        Path path = split.getPath(); // current split path
        lvalue.set(path.getName());//对第一条数据先写入数据所在文件名字
        lkey.set(key.get());
        context.write(lkey, lvalue);

        first = false;
      }

      String line = value.toString();
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);//转换成实例
        double prediction = forest.classify(dataset, rng, instance);//对该数据进行决策树分析--返回一个合理的标签
        lkey.set(dataset.getLabel(instance));//实例的真实标签
        lvalue.set(Double.toString(prediction));//返回该实例的预测分数
        context.write(lkey, lvalue);
      }
    }
  }
}
