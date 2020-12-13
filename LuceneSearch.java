package testlucene;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.Version;

public class LuceneSearch {

	public static void main(String[] args) throws IOException, ParseException {
		// TODO Auto-generated method stub
       System.out.println("aa");
       
       /*StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);

       // 1. create the index
       Directory index = new RAMDirectory();

       IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_30, analyzer);

       String fileName="stackoverflow_r_true_id_title_tags_train"; 
       
       IndexWriter w = new IndexWriter(index, config);
       
       BufferedReader br = new BufferedReader(new FileReader(fileName));
       
       String line = br.readLine();
		while (line != null) {
			//System.out.println(line);
			// read next line
			
			String [] arr =line.split("\t");
			if (arr.length!=4){
				continue;
			}
				
			//System.out.println(arr);
			//addDoc(w, arr[2], arr[0]+"_"+arr[1]);
			Document doc = new Document();
	        doc.add(new TextField("title", arr[2], Field.Store.YES));

	        // use a string field for isbn because we don't want it tokenized
	        doc.add(new StringField("isbn", arr[0]+"_"+arr[1], Field.Store.YES));
	        w.addDocument(doc);
			line = br.readLine();
			
		}
	   br.close();
       
       
       
       //addDoc(w, "Lucene for Dummies", "55320055Z");
       //addDoc(w, "Managing Gigabytes", "55063554A");
       //addDoc(w, "The Art of Computer Science", "9900333X");
       w.close();*/
       
       String testFile="stackoverflow_r_true_id_title_tags_test";
       BufferedReader brTest = new BufferedReader(new FileReader(testFile));
       
      
       //IndexReader reader = DirectoryReader.open(index);
	   //IndexSearcher searcher = new IndexSearcher(reader);
       //QueryParser qParser=new QueryParser(Version.LUCENE_30, "title", analyzer);
       
       String testline = brTest.readLine();
       
	   while (testline != null) {
			//System.out.println(line);
			// read next line
		   
		   StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);

	       // 1. create the index
	       Directory index = new RAMDirectory();

	       IndexWriterConfig config = new IndexWriterConfig(Version.LUCENE_30, analyzer);

	       String fileName="stackoverflow_r_true_id_title_tags_train"; 
	       
	       IndexWriter w = new IndexWriter(index, config);
	       
	       BufferedReader br = new BufferedReader(new FileReader(fileName));
	       
	       String line = br.readLine();
			while (line != null) {
				//System.out.println(line);
				// read next line
				
				String [] arr =line.split("\t");
				if (arr.length!=4){
					continue;
				}
					
				//System.out.println(arr);
				//addDoc(w, arr[2], arr[0]+"_"+arr[1]);
				Document doc = new Document();
		        doc.add(new TextField("title", arr[2], Field.Store.YES));

		        // use a string field for isbn because we don't want it tokenized
		        doc.add(new StringField("isbn", arr[0]+"_"+arr[1], Field.Store.YES));
		        w.addDocument(doc);
				line = br.readLine();
				
			}
		   br.close();
	       
	       
	       
	       //addDoc(w, "Lucene for Dummies", "55320055Z");
	       //addDoc(w, "Managing Gigabytes", "55063554A");
	       //addDoc(w, "The Art of Computer Science", "9900333X");
	       w.close();
		   
			
			String [] arr =testline.split("\t");
			if (arr.length!=4){
				continue;
			}
			
			String queryTitle = arr[2];
			String testPostId=arr[1];
			String testTrueLabel=arr[0];
			
			
			
			IndexReader reader = DirectoryReader.open(index);
			IndexSearcher searcher = new IndexSearcher(reader);
		    QueryParser qParser=new QueryParser(Version.LUCENE_30, "title", analyzer);
			
			
			Query q = qParser.parse(queryTitle);
			
			int hitsPerPage = 10000;
		    //IndexReader reader = DirectoryReader.open(index);
		    //IndexSearcher searcher = new IndexSearcher(reader);
			
		    TopDocs docs = searcher.search(q, hitsPerPage);
		    ScoreDoc[] hits = docs.scoreDocs;

		       // 4. display results
		    //System.out.println("Found " + hits.length + " hits.");
		    int flag=0;
		    for(int i=0;i<hits.length;++i) {
		      int docId = hits[i].doc;
		      Document d = searcher.doc(docId);
		      String true_postId = d.get("isbn");
		      String[] arr1=true_postId.split("_");
		      String trainTrueLabel = arr1[0];
		      if (testTrueLabel.equals(trainTrueLabel)){
		    	  System.out.println("found\t"+(i + 1) + ". " + d.get("isbn") + "\t" + d.get("title"));
		    	  flag=1;
		    	  break;
		      }
		      //System.out.println((i + 1) + ". " + d.get("isbn") + "\t" + d.get("title"));
		           //System.out.println((i + 1) + ". " + d.get("title"));
		    }
		    
		    if (flag==0){
		    	System.out.println("not found\t"+testPostId+"\t"+queryTitle);	
		    }
		    
		    

		       // reader can only be closed when there
		       // is no need to access the documents any more.
		   //reader.close();
			
		   testline = brTest.readLine();
		   
		   reader.close();
		   index.close();
		   analyzer.close();
		   
		  
		   
		   
			
		}
	   brTest.close();
	   
	   
	   //reader.close();
	   
	   
	  
       
       
       /*// 2. query
       String querystr = "ggplot not properly displaying";

       // the "title" arg specifies the default field to use
       // when no field is explicitly specified in the query.
       Query q = new QueryParser(Version.LUCENE_30, "title", analyzer).parse(querystr);

       // 3. search
       int hitsPerPage = 10000;
       IndexReader reader = DirectoryReader.open(index);
       IndexSearcher searcher = new IndexSearcher(reader);
       TopDocs docs = searcher.search(q, hitsPerPage);
       ScoreDoc[] hits = docs.scoreDocs;

       // 4. display results
       System.out.println("Found " + hits.length + " hits.");
       for(int i=0;i<hits.length;++i) {
           int docId = hits[i].doc;
           Document d = searcher.doc(docId);
           System.out.println((i + 1) + ". " + d.get("isbn") + "\t" + d.get("title"));
           //System.out.println((i + 1) + ". " + d.get("title"));
       }

       // reader can only be closed when there
       // is no need to access the documents any more.
       reader.close();*/
	}
	
	private static void addDoc(IndexWriter w, String title, String isbn) throws IOException {
        Document doc = new Document();
        doc.add(new TextField("title", title, Field.Store.YES));

        // use a string field for isbn because we don't want it tokenized
        doc.add(new StringField("isbn", isbn, Field.Store.YES));
        w.addDocument(doc);
    }
	
	

}
