package testlucene;

public class HitObject {
	
	public int postId;
	public String content;
	public int hitIndex;
	public String testTrueLabel;
	
	public HitObject(int postId, String content, int hitIndex,String testTrueLabel){
		this.postId=postId;
		this.content=content;
		this.hitIndex=hitIndex;
		this.testTrueLabel=testTrueLabel;
	}
}
