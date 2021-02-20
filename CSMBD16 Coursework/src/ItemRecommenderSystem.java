import java.io.File;
import java.io.IOException;
import java.util.List;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

public class ItemRecommenderSystem {
	public static void main(String[] args) throws IOException, TasteException {
		DataModel model = new FileDataModel(new File("/home/ubuntu/ua.base.hadoop"));
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new PearsonCorrelationSimilarity(model);
				//UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		System.out.println("Item-based Recommender");
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
	}
}
