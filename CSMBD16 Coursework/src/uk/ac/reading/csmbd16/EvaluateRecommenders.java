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
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericBooleanPrefUserBasedRecommender;
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
import org.apache.mahout.cf.taste.impl.recommender.svd.*;

public class EvaluateRecommenders {

	public static void main(String[] args) throws IOException, TasteException {

		DataModel model = new FileDataModel(new File("/home/ubuntu/eclipse-workspace/CSMBD16 Coursework/ua.base.hadoop"));

		RecommenderEvaluator avgAbsDiffEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderEvaluator rmsEvaluator = new RMSRecommenderEvaluator();
		
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("User-based Recommenders: Nearest-N User Neighbourhood (AvgAbsDiffEvaluator)");
		System.out.println("---------------------------------------------------------------------------");		
        // Evaluation using Nearest-N User neighborhood User-based Recommender (AvgAbsDiffEvaluator)
		evaluateUserBasedRecommendersWithNearestNUserNeighborhood(model, avgAbsDiffEvaluator);
		
		// Evaluation using Threshold User neighborhood User-based Recommender (AvgAbsDiffEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("User-based Recommenders: Threshold User Neighbourhood (AvgAbsDiffEvaluator)");
		System.out.println("---------------------------------------------------------------------------");	
		evaluateUserBasedRecommendersWithThresholdUserNeighborhood(model, avgAbsDiffEvaluator);

		// Evaluation using Item-based Recommender (AvgAbsDiffEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("Item-based Recommenders (AvgAbsDiffEvaluator)");
		System.out.println("---------------------------------------------------------------------------");
		evaluateItemBasedRecommenders(model, avgAbsDiffEvaluator);
		
		// Evaluation using SVD Recommender (AvgAbsDiffEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("SVD Recommender (AvgAbsDiffEvaluator)");
		System.out.println("---------------------------------------------------------------------------");
		evaluateSVDRecommender(model, avgAbsDiffEvaluator);

		System.out.println("---------------------------------------------------------------------------");
		System.out.println("User-based Recommenders: Nearest-N User Neighbourhood (rmsEvaluator)");
		System.out.println("---------------------------------------------------------------------------");		
        // Evaluation using Nearest-N User neighborhood User-based Recommender (rmsEvaluator)
		evaluateUserBasedRecommendersWithNearestNUserNeighborhood(model, rmsEvaluator);
		
		// Evaluation using Threshold User neighborhood User-based Recommender (rmsEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("User-based Recommenders: Threshold User Neighbourhood (rmsEvaluator)");
		System.out.println("---------------------------------------------------------------------------");	
		evaluateUserBasedRecommendersWithThresholdUserNeighborhood(model, rmsEvaluator);

		// Evaluation using Item-based Recommender (rmsEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("Item-based Recommenders (rmsEvaluator)");
		System.out.println("---------------------------------------------------------------------------");
		evaluateItemBasedRecommenders(model, rmsEvaluator);
		
		// Evaluation using SVD Recommender (rmsEvaluator)
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("SVD Recommender (rmsEvaluator)");
		System.out.println("---------------------------------------------------------------------------");
		evaluateSVDRecommender(model, rmsEvaluator);
		
		// Precision-Recall Evaluation
		System.out.println("---------------------------------------------------------------------------");
		System.out.println("Precision-Recall Evaluation");
		System.out.println("---------------------------------------------------------------------------");
		evaluateRecommendersWithIRStats(model);
		
	}
	
	
    // Evaluation using Nearest-N User neighborhood
	public static void evaluateUserBasedRecommendersWithNearestNUserNeighborhood(DataModel model, RecommenderEvaluator scoreEvaluator) throws TasteException {
		
		//RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
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

			// case 5: SpearmanCorrelationSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new SpearmanCorrelationSimilarity(model);
					UserNeighborhood neighborhood = new NearestNUserNeighborhood(n, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("SpearmanCorrelationSimilarity," + n + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
		}
	}
	
   // Evaluation using threshold-based neighborhood
	public static void evaluateUserBasedRecommendersWithThresholdUserNeighborhood(DataModel model, RecommenderEvaluator scoreEvaluator) throws TasteException {
		
		//RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
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

			// case 5: SpearmanCorrelationSimilarity
			recommenderBuilder = new RecommenderBuilder() {
				public Recommender buildRecommender(DataModel model) throws TasteException {
					UserSimilarity similarity = new SpearmanCorrelationSimilarity(model);
					UserNeighborhood neighborhood = new ThresholdUserNeighborhood(t, similarity, model);
					return new GenericUserBasedRecommender(model, neighborhood, similarity);
				}
			};
			System.out.println("SpearmanCorrelationSimilarity," + t + "," 
					+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));

		}

	}	
	
	// Evaluation using item-based neighborhood
	public static void evaluateItemBasedRecommenders(DataModel model, RecommenderEvaluator scoreEvaluator) throws TasteException {

		//RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
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

	
	public static void evaluateSVDRecommender(DataModel model, RecommenderEvaluator scoreEvaluator) throws TasteException {

		//RecommenderEvaluator scoreEvaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
		RecommenderBuilder recommenderBuilder;

		// Evaluation parameters
		double trainingPercentage = 0.7;
		double evaluationPercentage = 0.1;
		
		// SVDRecommender parameters
		int numFeatures = 10;
		double lambda = 0.05;
		int numIterations = 10;

		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				return new SVDRecommender(model, new ALSWRFactorizer(model, numFeatures, lambda, numIterations));
			}
		};
		
		System.out.println("SVDRecommender," 
				+ scoreEvaluator.evaluate(recommenderBuilder, null, model, trainingPercentage, evaluationPercentage));
	}
	
	public static void evaluateRecommendersWithIRStats(DataModel model) throws TasteException {
		RecommenderIRStatsEvaluator recPrecEvaluator = new GenericRecommenderIRStatsEvaluator();
		RecommenderBuilder recommenderBuilder;
		IRStatistics stats;
		
		int numOfRecommendationsToConsider = 10;
		double evaluationPercentage = 1.0;
		
		// case 1: Best recommender (AvgDiffEvaluation) - Item-based LogLikelihoodSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("Best Recommender 1," + stats.getPrecision() + "," + stats.getRecall());
		
		// Case 2: Best recommender (RMSEvaluator) - User-based with EuclideanDistanceSimilarity metric, 
		//         with neighborhood threshold = 0.9
		final double threshold = 0.9;
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				UserSimilarity similarity = new EuclideanDistanceSimilarity(model);
				UserNeighborhood neighborhood = new ThresholdUserNeighborhood(threshold, similarity, model);
				return new GenericUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("Best Recommender 2," + stats.getPrecision() + "," + stats.getRecall());
		
		// case 3: GenericBooleanPrefItemBasedRecommender - LogLikelihoodSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(model);
				return new GenericBooleanPrefItemBasedRecommender(model, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("GenericBooleanPrefItemBasedRecommender - LogLikelihoodSimilarity," + stats.getPrecision() + "," + stats.getRecall());

		// case 4: GenericBooleanPrefItemBasedRecommender - TanimotoCoefficientSimilarity
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new TanimotoCoefficientSimilarity(model);
				return new GenericBooleanPrefItemBasedRecommender(model, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("GenericBooleanPrefItemBasedRecommender - TanimotoCoefficientSimilarity," + stats.getPrecision() + "," + stats.getRecall());

		// case 5: GenericBooleanPrefUserBasedRecommender - LogLikelihoodSimilarity
		//threshold = 0.9;
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				UserSimilarity similarity = new LogLikelihoodSimilarity(model);
				UserNeighborhood neighborhood = new ThresholdUserNeighborhood(threshold, similarity, model);
				return new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("GenericBooleanPrefUserBasedRecommender - LogLikelihoodSimilarity," + stats.getPrecision() + "," + stats.getRecall());
		
		// case 6: GenericBooleanPrefUserBasedRecommender - TanimotoCoefficientSimilarity
		//threshold = 0.9;
		recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				UserSimilarity similarity = new TanimotoCoefficientSimilarity(model);
				UserNeighborhood neighborhood = new ThresholdUserNeighborhood(threshold, similarity, model);
				return new GenericBooleanPrefUserBasedRecommender(model, neighborhood, similarity);
			}
		};
		stats = recPrecEvaluator.evaluate(recommenderBuilder, null, model, null, numOfRecommendationsToConsider,
				GenericRecommenderIRStatsEvaluator.CHOOSE_THRESHOLD, evaluationPercentage);
		System.out.println("GenericBooleanPrefUserBasedRecommender - TanimotoCoefficientSimilarity," + stats.getPrecision() + "," + stats.getRecall());
	}
}
