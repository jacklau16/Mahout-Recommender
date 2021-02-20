package uk.ac.reading.csmbd16;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.List;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.SpearmanCorrelationSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class EvaluateRecommenders {

	public static void main(String[] args) throws IOException, TasteException {

		DataModel model = new FileDataModel(new File("/home/ubuntu/eclipse-workspace/mahout-test/ua.base.hadoop"));

		System.out.println("-----------------------------------------------------------");
		System.out.println("User-based Recommenders");
		System.out.println("-----------------------------------------------------------");		
        // Evaluation using Nearest-N User neighborhood
		evaluateUserBasedRecommendersWithNearestNUserNeighborhood(model);
		
		// Evaluation using Threshold User neighborhood
		System.out.println("-----------------------------------------------------------");
		evaluateUserBasedRecommendersWithThresholdUserNeighborhood(model);

		System.out.println("-----------------------------------------------------------");
		System.out.println("Item-based Recommenders");
		System.out.println("-----------------------------------------------------------");
		evaluateItemBasedRecommenders(model);
		
		/*		System.out.println("User-based Recommender");		
		Recommender recommender = recommenderBuilder.buildRecommender(model);
		List<RecommendedItem> recommendations = recommender.recommend(1, 2);
		for (RecommendedItem recommendation : recommendations) {
			System.out.println("Recommendation: " + recommendation);
		}
		RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		double score = scoreEvaluator.evaluate(recommenderBuilder, null, model, 0.7, 0.1);
		System.out.println("Score: " + score);

		RecommenderIRStatsEvaluator recPrecEvaluator = new GenericRecommenderIRStatsEvaluator();
		IRStatistics stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, 2,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, 1.0);
		System.out.println("Precision: " + stats.getPrecision());
		System.out.println("Recall: " + stats.getRecall());
		 */
	}
	
    // Evaluation using Nearest-N User neighborhood
	public static void evaluateUserBasedRecommendersWithNearestNUserNeighborhood(DataModel model) throws TasteException {
		
		RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder recommenderBuilder;
		// Evaluation parameters
		double trainingPercentage = 0.7;
        double evaluationPercentage = 0.1;
        
        // Print column headers
        System.out.println("Similarity Metric,N,Score");
        
		int[] nearest_n = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 }; 
		for (int n: nearest_n) {
			// case 1: EuclideanDistanceSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new EuclideanDistanceSimilarity(model);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(n, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("EuclideanDistanceSimilarity," + n + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
	
			// case 2: PearsonCorrelationSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(n, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("PearsonCorrelationSimilarity," + n + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

			// case 3: LogLikelihoodSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new LogLikelihoodSimilarity(model);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(n, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("LogLikelihoodSimilarity," + n + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

			// case 4: TanimotoCoefficientSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new TanimotoCoefficientSimilarity(model);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(n, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("TanimotoCoefficientSimilarity," + n + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
		}

	}
	
   // Evaluation using threshold-based neighborhood
	public static void evaluateUserBasedRecommendersWithThresholdUserNeighborhood(DataModel model) throws TasteException {
		
		RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder recommenderBuilder;
		// Evaluation parameters
		double trainingPercentage = 0.7;
        double evaluationPercentage = 0.1;
        
        // Print column headers
        System.out.println("Similarity Metric,Threshold,Score");
        
		double[] thresholds = { 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4 }; 
		for (double t: thresholds) {
			// case 1: EuclideanDistanceSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new EuclideanDistanceSimilarity(model);
					UserNeighborhood neighborhood = new ThresholdUserNeighborhood(t, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("EuclideanDistanceSimilarity," + t + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
	
			// case 2: PearsonCorrelationSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
					UserNeighborhood neighborhood = new ThresholdUserNeighborhood(t, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("PearsonCorrelationSimilarity," + t + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

			// case 3: LogLikelihoodSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new LogLikelihoodSimilarity(model);
					UserNeighborhood neighborhood = new ThresholdUserNeighborhood(t, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("LogLikelihoodSimilarity," + t + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

			// case 4: TanimotoCoefficientSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new TanimotoCoefficientSimilarity(model);
					UserNeighborhood neighborhood = new ThresholdUserNeighborhood(t, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("TanimotoCoefficientSimilarity," + t + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		}

	}	
	
	// Evaluation using threshold-based neighborhood
	public static void evaluateItemBasedRecommenders(DataModel model) throws TasteException {

		RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder recommenderBuilder;

		// Evaluation parameters
		double trainingPercentage = 0.7;
		double evaluationPercentage = 0.1;

		// Print column headers
		System.out.println("Similarity Metric,Score");

		// case 1: EuclideanDistanceSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new EuclideanDistanceSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		System.out.println("EuclideanDistanceSimilarity," 
				+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		// case 2: PearsonCorrelationSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		System.out.println("PearsonCorrelationSimilarity," 
				+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		// case 3: LogLikelihoodSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		System.out.println("LogLikelihoodSimilarity," 
				+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		// case 4: TanimotoCoefficientSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new TanimotoCoefficientSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		System.out.println("TanimotoCoefficientSimilarity," 
				+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

	}

}
