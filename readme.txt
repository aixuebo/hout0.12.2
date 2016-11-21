推荐流程
/path/to/dataset.csv文件内容
1,10,1.0
1,11,2.0
1,12,5.0

DataModel model = new FileDataModel(new File("/path/to/dataset.csv"));//数据源组成对象
UserSimilarity similarity = new PearsonCorrelationSimilarity(model);//计算该数据源的相似度,即可以预估user-user之间的相似度 也可以预估item-item之间的相似度  user-item之间的相似度
UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);//计算user的邻居,这个步骤很耗费性能
UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
List recommendations = recommender.recommend(2, 3);//为userid为2的推荐商品,最多推荐3个商品
for (RecommendedItem recommendation : recommendations) {
  System.out.println(recommendation);//打印推荐的商品id以及对应的打分value
}


