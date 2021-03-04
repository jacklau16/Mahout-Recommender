package uk.ac.reading.csmbd16;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedGraph;

public class FilmTrustRecommender {
	
	public static void main(String[] args) throws FileNotFoundException, IOException, TasteException {

		System.out.println("java.runtime.version: " + System.getProperty("java.runtime.version"));
		System.out.println("mahout version:" + org.apache.mahout.Version.version());
		
		// Item-based neighborhood
		DataModel model = new FileDataModel(new File("/home/ubuntu/eclipse-workspace/CSMBD16 Coursework/ua.base.hadoop"));
		RecommenderBuilder recommenderBuilder;
		// Evaluation parameters
		double trainingPercentage = 0.7;
		double evaluationPercentage = 1.0;

		// case 3: LogLikelihoodSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		
		// Recommend
		Recommender delegate = recommenderBuilder.buildRecommender(model);
		List<RecommendedItem> result = delegate.recommend(1, 5);
		System.out.println("Recommendations: ");
		for (RecommendedItem recommendedItem: result) {
			System.out.println(recommendedItem.getItemID()+": "+recommendedItem.getValue());
		}
/*		
		RecommenderEvaluator avgAbsDiffEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		System.out.println("EuclideanDistanceSimilarity," 
				+ avgAbsDiffEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
*/

		// Test for UserTrustSimilarity

		double threshold = 0.9;
		RecommenderEvaluator avgAbsDiffEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				UserSimilarity similarity = new UserTrustSimilarity();
				//UserSimilarity similarity = new EuclideanDistanceSimilarity(model);
				UserNeighborhood neighborhood = new ThresholdUserNeighborhood(threshold, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		System.out.println("UserTrustSimilarity," + threshold + "," 
				+ avgAbsDiffEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		
		SimpleDirectedGraph<Integer, DefaultEdge> graph =
				new SimpleDirectedGraph<Integer, DefaultEdge>(DefaultEdge.class);

		String file = "trust.csv";

		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				String cols[] = line.split(",");
				int v0 = Integer.valueOf(cols[0]);
				int v1 = Integer.valueOf(cols[1]);
				graph.addVertex(v0);
				graph.addVertex(v1);
				graph.addEdge(v0, v1);
			}
		}

		GraphPath<Integer, DefaultEdge> path = DijkstraShortestPath.findPathBetween(graph, 605, 837);
		//graph.vertexSet().size();
		System.out.println(path.getLength() + " - " + path); 
	}
}
