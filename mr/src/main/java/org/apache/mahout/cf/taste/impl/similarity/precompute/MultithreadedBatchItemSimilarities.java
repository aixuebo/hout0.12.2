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

package org.apache.mahout.cf.taste.impl.similarity.precompute;

import com.google.common.io.Closeables;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.precompute.BatchItemSimilarities;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItems;
import org.apache.mahout.cf.taste.similarity.precompute.SimilarItemsWriter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Precompute item similarities in parallel on a single machine. The recommender given to this class must use a
 * DataModel that holds the interactions in memory (such as
 * {@link org.apache.mahout.cf.taste.impl.model.GenericDataModel} or
 * {@link org.apache.mahout.cf.taste.impl.model.file.FileDataModel}) as fast random access to the data is required
 */
public class MultithreadedBatchItemSimilarities extends BatchItemSimilarities {

  private int batchSize;

  private static final int DEFAULT_BATCH_SIZE = 100;

  private static final Logger log = LoggerFactory.getLogger(MultithreadedBatchItemSimilarities.class);

  /**
   * @param recommender recommender to use
   * @param similarItemsPerItem number of similar items to compute per item
   */
  public MultithreadedBatchItemSimilarities(ItemBasedRecommender recommender, int similarItemsPerItem) {
    this(recommender, similarItemsPerItem, DEFAULT_BATCH_SIZE);
  }

  /**
   * @param recommender recommender to use
   * @param similarItemsPerItem number of similar items to compute per item
   * @param batchSize size of item batches sent to worker threads
   */
  public MultithreadedBatchItemSimilarities(ItemBasedRecommender recommender, int similarItemsPerItem, int batchSize) {
    super(recommender, similarItemsPerItem);
    this.batchSize = batchSize;
  }

  /**
   * degreeOfParallelism 线程数
   * 
   * writer 表示最终要向哪个输出流中写入数据
   * 返回值是一共多少个item-item-value数据
   */
  @Override
  public int computeItemSimilarities(int degreeOfParallelism, int maxDurationInHours, SimilarItemsWriter writer)
    throws IOException {

	//创建线程池
    ExecutorService executorService = Executors.newFixedThreadPool(degreeOfParallelism + 1);

    Output output = null;
    try {
      writer.open();//打开输出文件

      DataModel dataModel = getRecommender().getDataModel();//获取user-item-value的模型

      //不断的从dataModel中读取user-item-value值,存储到共享队列中,让消费者worker线程从队列取数据工作
      BlockingQueue<long[]> itemsIDsInBatches = queueItemIDsInBatches(dataModel, batchSize, degreeOfParallelism);//一个共享队列,多线程都从该队列获取数据
      
      BlockingQueue<List<SimilarItems>> results = new LinkedBlockingQueue<>();//多线程的结果都写入该队列,该队列也是共享的

      AtomicInteger numActiveWorkers = new AtomicInteger(degreeOfParallelism);//活着的工作线程数量
      for (int n = 0; n < degreeOfParallelism; n++) {//线程池开启多线程
        executorService.execute(new SimilarItemsWorker(n, itemsIDsInBatches, results, numActiveWorkers));
      }

      output = new Output(results, writer, numActiveWorkers);
      executorService.execute(output);

    } catch (Exception e) {
      throw new IOException(e);
    } finally {
      executorService.shutdown();
      try {
        boolean succeeded = executorService.awaitTermination(maxDurationInHours, TimeUnit.HOURS);
        if (!succeeded) {
          throw new RuntimeException("Unable to complete the computation in " + maxDurationInHours + " hours!");
        }
      } catch (InterruptedException e) {
        throw new RuntimeException(e);
      }
      Closeables.close(writer, false);
    }

    return output.getNumSimilaritiesProcessed();
  }

  /**
   * 不断的从dataModel中读取user-item-value值,存储到共享队列中,让消费者worker线程从队列取数据工作
   * @param dataModel 获取user-item-value的模型
   * @param batchSize 一次批处理的数量
   * @param degreeOfParallelism 多线程数量
   * @return
   * @throws TasteException
   */
  private static BlockingQueue<long[]> queueItemIDsInBatches(DataModel dataModel, int batchSize,
                                                             int degreeOfParallelism)
      throws TasteException {

    LongPrimitiveIterator itemIDs = dataModel.getItemIDs();//获取所有的item集合
    int numItems = dataModel.getNumItems();//返回一共多少个itemid

    //队列的元素是long[] batch,即一个批处理的结果,itemid集合
    BlockingQueue<long[]> itemIDBatches = new LinkedBlockingQueue<>((numItems / batchSize) + 1);

    long[] batch = new long[batchSize];//一次批处理的集合,不断的用同一组批处理文件
    int pos = 0;
    while (itemIDs.hasNext()) {//不断迭代item
      batch[pos] = itemIDs.nextLong();//将item添加到批处理数组中
      pos++;
      if (pos == batchSize) {//当达到批处理伐值时候,重新
        itemIDBatches.add(batch.clone());
        pos = 0;
      }
    }

    if (pos > 0) {//将剩余的文件从0到pos位置的内容,写入到队列中
      long[] lastBatch = new long[pos];
      System.arraycopy(batch, 0, lastBatch, 0, pos);
      itemIDBatches.add(lastBatch);
    }

    if (itemIDBatches.size() < degreeOfParallelism) {
      throw new IllegalStateException("Degree of parallelism [" + degreeOfParallelism + "] " +
              " is larger than number of batches [" + itemIDBatches.size() +"].");
    }

    log.info("Queued {} items in {} batches", numItems, itemIDBatches.size());

    return itemIDBatches;
  }


  private static class Output implements Runnable {

    private final BlockingQueue<List<SimilarItems>> results;//多线程的结果都写入该队列,该队列也是共享的
    private final SimilarItemsWriter writer;
    private final AtomicInteger numActiveWorkers;//活着的工作线程数量
    private int numSimilaritiesProcessed = 0;//返回一共处理了多少个item--item--value数据

    Output(BlockingQueue<List<SimilarItems>> results, SimilarItemsWriter writer, AtomicInteger numActiveWorkers) {
      this.results = results;
      this.writer = writer;
      this.numActiveWorkers = numActiveWorkers;
    }

    private int getNumSimilaritiesProcessed() {
      return numSimilaritiesProcessed;
    }

    @Override
    public void run() {
      while (numActiveWorkers.get() != 0 || !results.isEmpty()) {//有工作线程还活着,并且有输出内容,则就不断处理
        try {
          List<SimilarItems> similarItemsOfABatch = results.poll(10, TimeUnit.MILLISECONDS);//从结果集中取出10个元素
          if (similarItemsOfABatch != null) {
            for (SimilarItems similarItems : similarItemsOfABatch) {
              writer.add(similarItems);//向writer中添加结果,即一个itemid对一组itemid和偏好值的输出
              numSimilaritiesProcessed += similarItems.numSimilarItems();//返回该item本次跟多少个item有关联
            }
          }
        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
    }
  }

  /**
   * 表示每一个工作的线程
   */
  private class SimilarItemsWorker implements Runnable {

    private final int number;//表示线程的ID,即第几个线程
    private final BlockingQueue<long[]> itemIDBatches;//一个共享队列,多线程都从该队列获取数据
    private final BlockingQueue<List<SimilarItems>> results;//多线程的结果都写入该队列,该队列也是共享的
    private final AtomicInteger numActiveWorkers;//活着的工作线程数量

    SimilarItemsWorker(int number, BlockingQueue<long[]> itemIDBatches, BlockingQueue<List<SimilarItems>> results,
        AtomicInteger numActiveWorkers) {
      this.number = number;
      this.itemIDBatches = itemIDBatches;
      this.results = results;
      this.numActiveWorkers = numActiveWorkers;
    }

    @Override
    public void run() {

      int numBatchesProcessed = 0;
      while (!itemIDBatches.isEmpty()) {//队列有内容,则不停止
        try {
          //itemid集合
          long[] itemIDBatch = itemIDBatches.take();//从队列中获取一组数据

          List<SimilarItems> similarItemsOfBatch = new ArrayList<>(itemIDBatch.length);
          for (long itemID : itemIDBatch) {//循环所有的item,找到与该item相关联的item集合和相似度集合
            List<RecommendedItem> similarItems = getRecommender().mostSimilarItems(itemID, getSimilarItemsPerItem());//找item相似的数据.最多找多少个相似的item,getSimilarItemsPerItem
            similarItemsOfBatch.add(new SimilarItems(itemID, similarItems));
          }

          results.offer(similarItemsOfBatch);//将计算的结果存储到结果中

          if (++numBatchesProcessed % 5 == 0) {//打印第几个工作线程已经处理了多少次数据了
            log.info("worker {} processed {} batches", number, numBatchesProcessed);
          }

        } catch (Exception e) {
          throw new RuntimeException(e);
        }
      }
      log.info("worker {} processed {} batches. done.", number, numBatchesProcessed);
      numActiveWorkers.decrementAndGet();//活着的工作线程数量减少1个
    }
  }
}
