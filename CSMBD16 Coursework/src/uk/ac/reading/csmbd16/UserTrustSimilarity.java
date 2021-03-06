/**
 * CSMBD16 Coursework
 * Description: UserTrustSimilarity - Custom implementation of UserSimilarity for FilmTrust dataset
 * @author jack.lau@student.reading.ac.uk (28838142)
 *
 */
package uk.ac.reading.csmbd16;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.shortestpath.DijkstraShortestPath;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.graph.SimpleGraph;

public class UserTrustSimilarity implements UserSimilarity {
	
	SimpleGraph<Integer, DefaultEdge> graph;
	String file = "trust.txt";
	
	public UserTrustSimilarity() {
		
		graph = new SimpleGraph<Integer, DefaultEdge>(DefaultEdge.class);

		// Construct the user trust graph with adding all users as vertices, and all "trust" relationships as edges
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				String cols[] = line.split(" ");
				int v0 = Integer.valueOf(cols[0]);
				int v1 = Integer.valueOf(cols[1]);
				graph.addVertex(v0);
				graph.addVertex(v1);
				graph.addEdge(v0, v1);
			}
		} catch (Exception e) {
			e.printStackTrace(System.err);
		}
	}

	@Override
	public void refresh(Collection<Refreshable> arg0) {
		// Do nothing

	}

	@Override
	public void setPreferenceInferrer(PreferenceInferrer arg0) {
		// Do nothing

	}

	// Core method for calculating the similarity between two users
	@Override
	public double userSimilarity(long user1, long user2) throws TasteException {

		if (!graph.containsVertex((int)user1) || !graph.containsVertex((int)user2))
			return 0;
		
		GraphPath<Integer, DefaultEdge> path = DijkstraShortestPath.findPathBetween(graph, (int)user1, (int)user2);
		
		if (path==null)
			return 0;
		
		int pathLen = path.getLength();

		if (pathLen == 0)
			return 0;
		else
			return 1/pathLen;
	}

}
