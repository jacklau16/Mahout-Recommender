package uk.ac.reading.csmbd16;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Scanner;

import org.apache.mahout.cf.taste.common.NoSuchUserException;
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

		// Print system information
		System.out.println("java.runtime.version: " + System.getProperty("java.runtime.version"));
		System.out.println("mahout version:" + org.apache.mahout.Version.version());
		
		// Create the DataModel
		System.out.print("\nLoading dataset...");
		DataModel model = new FileDataModel(new File("/home/ubuntu/eclipse-workspace/CSMBD16 Coursework/ua.base.hadoop"));
		System.out.println("[Done]");

		// Create the RecommenderBuilder
		RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model) throws TasteException {
				ItemSimilarity similarity = new LogLikelihoodSimilarity(model);
				return new GenericItemBasedRecommender(model, similarity);
			}
		};
		
		// Initialise the Recommender
		System.out.print("Initialise recommender...");
		Recommender delegate = recommenderBuilder.buildRecommender(model);
		System.out.println("[Done]\n");

		// Print instruction message
		System.out.println("=============================");
		System.out.println(" FilmTrust Recommender");
		System.out.println("=============================");
		System.out.println("Instructions:");
		System.out.println("1) Input the following for perform recommendation:");
		System.out.println("    [User ID], [No. of recommendations]");
		System.out.println("2) Type \"bye\" to quit.");
		
		Scanner input = new Scanner(System.in);
		boolean quit = false;
		String command = "";
		String token[];
		
		// Loop to receive user input from console
		while (!quit) {
			System.out.print("\n> ");
			command = input.nextLine();
			if (command.equals("bye"))
				quit = true;
			else {
				try {
					token = command.split(", ");
					int userID = Integer.valueOf(token[0]);
					int numItems = Integer.valueOf(token[1]);

					List<RecommendedItem> result = delegate.recommend(userID, numItems);

					if (result.size()==0)
						System.out.println("No recommendation!");
					else {
						System.out.println("Recommendations: ");
						System.out.println("ITEM   RATING");
						for (RecommendedItem recommendedItem: result)
							System.out.println(String.format("%1$4s", recommendedItem.getItemID())+" : "+recommendedItem.getValue());
					}
				} catch (NoSuchUserException e) {
					System.out.println("No such user!");
				} catch (NumberFormatException e) {
					System.out.println("Invalid input!");
				}
			}
		}
	}
		
}
