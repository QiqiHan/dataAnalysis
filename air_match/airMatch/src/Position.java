/**
 * Created by H77 on 2017/12/5.
 */
public class Position {
    //Position应该包括时间和位置和前驱节点
    public int x;
    public int y;
    public int time; //精确到分钟
    public double wind;
    public Position pre;


    public Position(int x, int y, int time ,double wind) {
        this.x = x;
        this.y = y;
        this.time = time;
        this.wind = wind;
    }
}
