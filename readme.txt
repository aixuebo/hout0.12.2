�Ƽ�����
/path/to/dataset.csv�ļ�����
1,10,1.0
1,11,2.0
1,12,5.0

DataModel model = new FileDataModel(new File("/path/to/dataset.csv"));//����Դ��ɶ���
UserSimilarity similarity = new PearsonCorrelationSimilarity(model);//���������Դ�����ƶ�,������Ԥ��user-user֮������ƶ� Ҳ����Ԥ��item-item֮������ƶ�  user-item֮������ƶ�
UserNeighborhood neighborhood = new ThresholdUserNeighborhood(0.1, similarity, model);
UserBasedRecommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
List recommendations = recommender.recommend(2, 3);
for (RecommendedItem recommendation : recommendations) {
  System.out.println(recommendation);
}

