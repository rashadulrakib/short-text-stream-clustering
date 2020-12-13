package testlucene;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.ClassicSimilarity;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class Lucene1 {

	
	
	public static void main(String[] args) throws IOException, ParseException {
		// TODO Auto-generated method stub
		//System.out.println("Lucene1");
		//CreateIndex();
		QueryIndex("D:/githubprojects/PyMigrationRecommendation/src/notebooks/test_stackoverflow_r_true_id_title_tags");
	}
	
	private static void QueryIndex(String testFile) throws IOException, ParseException {
		Directory dir = FSDirectory.open(Paths.get("D:/githubprojects/test-lucene/testlucene/src/testlucene/r_index"));

		IndexReader reader = DirectoryReader.open(dir);

		IndexSearcher searcher = new IndexSearcher(reader);

		Similarity similarity=searcher.getSimilarity(true);
		
		
		QueryParser qp = new QueryParser("content",new StandardAnalyzer());

		BufferedReader br = new BufferedReader(new FileReader(testFile));
		String line = "";//br.readLine();
		
		TreeMap<Integer,Integer> hmHitIndex_Count = new TreeMap<Integer, Integer>();
		TreeMap<Integer,ArrayList<String>> hmHitIndex_Items = new TreeMap<Integer,ArrayList<String>>();
		
		while ( (line=br.readLine()) != null) {
				
				// read next line
				
				String [] arr =line.split("\t");
				if (arr.length!=4){
					continue;
				}
				
				//System.out.println(arr);
				String truelabel=arr[0];
				String id=arr[1];
				String title=arr[2];
				//System.out.println(truelabel+"\t"+id+"\t"+title);
				
				int hitIndex=GetHitIndex(title, truelabel, searcher, qp);
				if (hitIndex>0){
					//System.out.println("found\t"+truelabel+"\t"+id+"\t"+title+"\t"+hitIndex);
					if (hmHitIndex_Count.containsKey(hitIndex)){
						Integer count = hmHitIndex_Count.get(hitIndex);
						count+=1;
						hmHitIndex_Count.put(hitIndex, count);
					}else{
						hmHitIndex_Count.put(hitIndex, 1);
					}
					
					
					if (hmHitIndex_Items.containsKey(hitIndex)){
						ArrayList<String> items = hmHitIndex_Items.get(hitIndex);
						items.add(truelabel+"\t"+id+"\t"+title);
						hmHitIndex_Items.put(hitIndex, items);
					}else{
						ArrayList<String> items = new ArrayList<String>();
						items.add(truelabel+"\t"+id+"\t"+title);
						hmHitIndex_Items.put(hitIndex, items);
					}
					
				}else{
					//System.out.println("not found\t"+truelabel+"\t"+id+"\t"+title);
				}
				
		}
		
		br.close();
		
		reader.close();
		
		//for (Integer hitIndex : hmHitIndex_Count.keySet()) {
		//	Integer count = hmHitIndex_Count.get(hitIndex);
		//	System.out.println(hitIndex+"\t"+count);
			
		//}
		
		System.out.println("-------Count with Items----");
		
		for (Integer hitIndex : hmHitIndex_Items.keySet()) {
			ArrayList<String> items = hmHitIndex_Items.get(hitIndex);
			System.out.println("key="+hitIndex+"\t"+items.size());
			
			for(String item:items){
				System.out.println(hitIndex+"\t"+item);
			}
			
		}
	}

	private static int GetHitIndex(String queryInput, String searchTrueLabel,	IndexSearcher searcher, QueryParser qp) throws IOException, ParseException {
		Query query = qp.parse(QueryParser.escape(queryInput));	 

		TopDocs hits = searcher.search(query, 10000,Sort.RELEVANCE);

		//System.out.println("Retrieved documents: "+hits.totalHits +" ScoreDocs:"+hits.scoreDocs.length);

		

		int c=0;

		for(ScoreDoc sd:hits.scoreDocs){

				Document d = searcher.doc(sd.doc);
				
		//if you want to get value for an index put the index name as a parameter
				String true_id= d.getField("id").stringValue();
				String content=d.getField("content").stringValue();
				
				String trueLabel=true_id.split("_")[0];

				c++;
				
				if (trueLabel.equals(searchTrueLabel)){
					//System.out.println(c+"\t"+ true_id+"\t"+content+"\t input:"+queryInput);
					return c;
				}

		}
		
		return -100;
	}

	public static void CreateIndex() throws IOException{
		

		Directory dir = FSDirectory.open(Paths.get("D:/githubprojects/test-lucene/testlucene/src/testlucene/r_index"));

		IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());//set the analyzer that break text to token

		config.setSimilarity(new ClassicSimilarity());

		System.out.println("Similarity Algorithm: "+config.getSimilarity());

		IndexWriter writer = new IndexWriter(dir, config);

		writer.deleteAll(); //delete all content of existing index

		List<Document> docs = new ArrayList<Document>();

		

		


		//you can create a loop to add multiple documents
		
		String fileName="D:/githubprojects/PyMigrationRecommendation/src/notebooks/train_stackoverflow_r_true_id_title_tags";

		//after that add those documents. this should be outside the loop
		BufferedReader br = new BufferedReader(new FileReader(fileName));
	       
	    String line = "";//br.readLine();
		while ( (line=br.readLine()) != null) {
				
				// read next line
				
				String [] arr =line.split("\t");
				if (arr.length!=4){
					continue;
				}
				
				//System.out.println(arr);
				String truelabel=arr[0];
				String id=arr[1];
				String title=arr[2];
				System.out.println(truelabel+"\t"+id+"\t"+title);
				
				Document document = new Document();

				//create field you want to store. each field contain different values

				FieldType contentType = new FieldType();
				contentType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
				contentType.setStored(true);
				contentType.setTokenized(true);

								
				FieldType idType = new FieldType();
				idType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
				idType.setStored(true);		

				document.add(new Field("content",title,contentType));
				document.add(new Field("id",truelabel+"_"+id,idType));
				
				docs.add(document);
		}
					
				
		br.close();		

		writer.addDocuments(docs);

		writer.commit();

		writer.close();



	}

}
