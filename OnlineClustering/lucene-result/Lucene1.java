package testlucene;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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

import java.time.Duration;
import java.time.Instant;


public class Lucene1 {

	  //file2:---test_postId1, train_PostId2, titleSimilarity, bodySimilarity, tagSimilarity, timeDiff, test_postid1_hiteRank_lucene,  test_postid1_hiteRank_proposed, lucene_hit_timeduration,  proposed_hit_timeduration

	
	
	public static void main(String[] args) throws IOException, ParseException {
		// TODO Auto-generated method stub
		//System.out.println("Lucene1");
		String textType="body"; //tags title body
		String language="csharp";
		String indexFile="d:/githubprojects/test-lucene/testlucene/src/testlucene/"+language+"_"+textType+"_index";
		
		Instant start = Instant.now();
		String trainFile="d:/githubprojects/PyMigrationRecommendation/src/notebooks/train_stackoverflow_"+language+"_true_id_title_tags_body_createtime";
		CreateIndex(indexFile, trainFile, textType);
		Instant end = Instant.now();
		Duration timeElapsed = Duration.between(start, end);
		double train_sec_lucene = (double)timeElapsed.toMillis()/1000;
		System.out.println("Training time diff="+train_sec_lucene);

		
		
		
		String testFile="D:/githubprojects/PyMigrationRecommendation/src/notebooks/test_stackoverflow_"+language+"_true_id_title_tags_body_createtime";
		String luceneOutFile="D:/githubprojects/test-lucene/testlucene/src/testlucene/lucene_result_"+language+"_"+textType+".txt";
		QueryIndex(indexFile, testFile , luceneOutFile, textType);
	}
	
	private static void QueryIndex(String indexFile, String testFile, String luceneOutFile, String textType) throws IOException, ParseException {
		Directory dir = FSDirectory.open(Paths.get(indexFile));

		IndexReader reader = DirectoryReader.open(dir);

		IndexSearcher searcher = new IndexSearcher(reader);

		Similarity similarity=searcher.getSimilarity(true);
		
		
		QueryParser qp = new QueryParser("content",new StandardAnalyzer());

		BufferedReader br = new BufferedReader(new FileReader(testFile));
		String line = "";//br.readLine();
		
		FileWriter fwriter = new FileWriter(new File(luceneOutFile)); 
		//file2:---test_postId1, train_PostId2, titleSimilarity, bodySimilarity, tagSimilarity, timeDiff, test_postid1_hiteRank_lucene,  
		//test_postid1_hiteRank_proposed, lucene_hit_timeduration,  proposed_hit_timeduration
		fwriter.write("testPostId"+"\t"+"trainPostId"+"\tTitleSim\tBodySim\tTagSim\t"+"LuceneHitRank"+"\tProposedHitRank\t"+"lucene_hit_duration"+"\tProposed_hit_duration\tLuceneTestTrueLabel\n");
		
		TreeMap<Integer,Integer> hmHitIndex_Count = new TreeMap<Integer, Integer>();
		TreeMap<Integer,ArrayList<String>> hmHitIndex_Items = new TreeMap<Integer,ArrayList<String>>();
		
		int hitCount=0;
		int totalRecords=0;
				
		while ( (line=br.readLine()) != null) {
				
				// read next line
				
				String [] arr =line.split("\t");
				if (arr.length!=6){
					continue;
				}
				totalRecords++;
				//System.out.println(arr);
				String testTruelabel=arr[0];
				String testPostId=arr[1];
				String testTitle=arr[2];
				String testTags=arr[3];
				String testBodyKeywords=arr[4];
				String testCreatetime=arr[5];
				testTags=testTags.replace("<", "");
				testTags=String.join(" ", Arrays.asList(testTags.split(">"))).trim();
				
				String queryInput=testTags;
				if (textType.equals("tags")){
					queryInput=testTags;
				}else if (textType.equals("title")){
					queryInput=testTitle;
				}else if (textType.equals("body")){
					queryInput=testBodyKeywords;
				}
				
				
				
				Instant start = Instant.now();
				HitObject objTrainHitObject  =GetHitIndex(queryInput, testTruelabel, searcher, qp);
				Instant end = Instant.now();
				Duration timeElapsed = Duration.between(start, end);
				double tes_sec_lucene = (double)timeElapsed.toMillis()/1000;
				
				
				int luceneHitIndex= objTrainHitObject.hitIndex;
				if (luceneHitIndex>0){
					//System.out.println("found found\t"+testTruelabel+"\t"+testPostId+"\t"+testTags+"\t"+luceneHitIndex);
					fwriter.write(testPostId+"\t"+objTrainHitObject.postId+"\t0\t0\t0\t"+luceneHitIndex+"\t0\t"+tes_sec_lucene+"\t0\t"+objTrainHitObject.testTrueLabel+"\n"); 
					//file2:---test_postId1, train_PostId2, titleSimilarity, bodySimilarity, tagSimilarity, timeDiff, test_postid1_hiteRank_lucene,  
					//test_postid1_hiteRank_proposed, lucene_hit_timeduration,  proposed_hit_timeduration
					hitCount++;
					if (hmHitIndex_Count.containsKey(luceneHitIndex)){
						Integer count = hmHitIndex_Count.get(luceneHitIndex);
						count+=1;
						hmHitIndex_Count.put(luceneHitIndex, count);
					}else{
						hmHitIndex_Count.put(luceneHitIndex, 1);
					}
					
					
					if (hmHitIndex_Items.containsKey(luceneHitIndex)){
						ArrayList<String> items = hmHitIndex_Items.get(luceneHitIndex);
						items.add(testTruelabel+"\t"+testPostId+"\t"+testTitle);
						hmHitIndex_Items.put(luceneHitIndex, items);
					}else{
						ArrayList<String> items = new ArrayList<String>();
						items.add(testTruelabel+"\t"+testPostId+"\t"+testTitle);
						hmHitIndex_Items.put(luceneHitIndex, items);
					}
					
				}else{
					//System.out.println("not found\t"+testTruelabel+"\t"+testPostId+"\t"+testTags);
					fwriter.write(testPostId+"\t"+objTrainHitObject.postId+"\t0\t0\t0\t"+luceneHitIndex+"\t0\t"+tes_sec_lucene+"\t0\t"+objTrainHitObject.testTrueLabel+"\n"); 
				}
				
				//test_postId1, train_PostId2, titleSimilarity, bodySimilarity, tagSimilarity, timeDiff, test_postId1_rank
				String test_postId=testPostId;
				
				
		}
		
		br.close();
		
		reader.close();
		
		fwriter.flush();
		fwriter.close();
		
		
		System.out.println("hitCount="+hitCount+"\ttotal="+totalRecords);
	}

	private static HitObject GetHitIndex(String queryInput, String searchTrueLabel,	IndexSearcher searcher, QueryParser qp) throws IOException, ParseException {
		try{
			Query query = qp.parse(QueryParser.escape(queryInput));	 

			TopDocs hits = searcher.search(query, 10000,Sort.RELEVANCE);

			
			int c=0;
			String train_postId="";

			for(ScoreDoc sd:hits.scoreDocs){

					Document d = searcher.doc(sd.doc);
					
			//if you want to get value for an index put the index name as a parameter
					String true_id= d.getField("id").stringValue();
					String content=d.getField("content").stringValue();
					
					
					String trueLabel=true_id.split("_")[0];
					train_postId=true_id.split("_")[1];
					c++;
					
					if (trueLabel.equals(searchTrueLabel)){
						//System.out.println(c+"\t"+ true_id+"\t"+content+"\t input:"+queryInput);
						
						return new HitObject(Integer.parseInt(train_postId), content, c, searchTrueLabel);
					}

			}	
		}catch(Exception e){
			return new HitObject(-100, "", -100, "");
		}
		
		
		
		return new HitObject(-100, "", -100, "");
	}

	public static void CreateIndex(String indexFile, String trainFile, String textType) throws IOException{
		

		Directory dir = FSDirectory.open(Paths.get(indexFile));

		IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());//set the analyzer that break text to token

		config.setSimilarity(new ClassicSimilarity());

		System.out.println("Similarity Algorithm: "+config.getSimilarity());

		IndexWriter writer = new IndexWriter(dir, config);

		writer.deleteAll(); //delete all content of existing index

		List<Document> docs = new ArrayList<Document>();

		

		


		//you can create a loop to add multiple documents
		
		
		//after that add those documents. this should be outside the loop
		BufferedReader br = new BufferedReader(new FileReader(trainFile));
	       
	    String line = "";//br.readLine();
		while ( (line=br.readLine()) != null) {
				
				// read next line
				
				String [] arr =line.split("\t");
				if (arr.length!=6){
					continue;
				}
				
				//System.out.println(arr);
				String truelabel=arr[0];
				String id=arr[1]; // postid
				String title=arr[2];
				String tags=arr[3];
				String bodyKeywords=arr[4];
				String createtime=arr[5];
				tags=tags.replace("<", "");
				tags=String.join(" ", Arrays.asList(tags.split(">"))).trim();
				
				String content=tags;
				if (textType.equals("tags")){
					content=tags;
				}else if (textType.equals("title")){
					content=title;
				}else if (textType.equals("body")){
					content=bodyKeywords;
				}
				
				//System.out.println(truelabel+"\t"+id+"\t"+title+"\t"+tags);
				
				Document document = new Document();

				//create field you want to store. each field contain different values

				FieldType contentType = new FieldType();
				contentType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
				contentType.setStored(true);
				contentType.setTokenized(true);

								
				FieldType idType = new FieldType();
				idType.setIndexOptions(IndexOptions.DOCS_AND_FREQS);
				idType.setStored(true);		

				document.add(new Field("content",content,contentType));
				document.add(new Field("id",truelabel+"_"+id,idType));
				
				docs.add(document);
		}
					
				
		br.close();		

		writer.addDocuments(docs);

		writer.commit();

		writer.close();



	}

}
